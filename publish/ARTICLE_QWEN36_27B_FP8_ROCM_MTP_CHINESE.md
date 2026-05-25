# 在 AMD ROCm DSW 上部署 Qwen3.6-27B-FP8：vLLM 服务、MTP 解码加速与 8K 正确性验证

## 摘要

本文记录我在 ModelScope DSW AMD GPU 实例上，对 `Qwen/Qwen3.6-27B-FP8` 进行 ROCm/vLLM 部署、MTP 投机解码加速、8K needle retrieval 正确性验证和 FP8 KV cache 实验的过程。

这次实验的重点不是只证明“模型能启动”，而是把部署过程整理成一套可复现的研究 baseline：服务可启动、短输出可用、near-8K 长上下文检索通过、MTP 在 decode-heavy 场景下有稳定收益，并且相关脚本、日志摘要和报告都能在实例重启后恢复。

本轮最关键的严格对照结果来自强制 1024-token decode 实验：所有配置都生成完整 `1024,1024` completion tokens。在相同 prompt、相同 `max_tokens=1024`、相同单请求测试口径下，baseline median decode throughput 为 `21.46 tok/s`，`num_speculative_tokens=4` 为 `54.48 tok/s`，约 `2.54x`。

需要说明的是，这不是生产 serving benchmark，也不是与其他硬件的直接横向评测。它更适合作为 AMD ROCm 上研究 Qwen3.6-27B-FP8、vLLM、MTP、FP8 KV cache 和后续量化优化的工程基线。

## 选题背景

Qwen3.6-27B 是一个规模适中的 MoE/大模型推理研究对象：参数规模足够大，能暴露 ROCm/vLLM 长上下文、KV cache、投机解码等真实工程问题；同时又不像超大模型那样一开始就被权重体积和调度复杂度完全淹没。

我这轮选择官方 FP8 版本，是因为它更贴近推理部署场景。研究问题主要有三个：

1. 在 ModelScope DSW AMD GPU 环境中，`Qwen/Qwen3.6-27B-FP8` 能否通过 vLLM ROCm 路径稳定启动？
2. MTP 投机解码在 AMD ROCm 上是否真的带来 decode 性能收益？
3. 8K 长上下文和 FP8 KV cache 是否能通过 correctness gate，而不是只看服务进程存在？

## 环境与目录设计

本次实验环境：

| 项目 | 记录 |
|---|---|
| 平台 | ModelScope DSW AMD GPU |
| Python | 3.12 |
| PyTorch | 2.10.0+git... |
| HIP | 7.2.53211 |
| vLLM | 0.20.1+rocm721 |
| GPU | 1 张 AMD GPU，显存约 192GB |
| 模型 | Qwen/Qwen3.6-27B-FP8 |

复现目录分成两类：

```bash
# 复现资料目录：脚本、日志摘要、报告、prompt、结果表
export REPRO_DIR=/mnt/workspace/qwen27b_rocm_mtp_repro_20260525

# 模型缓存目录：权重体积大，可以按需重新下载
export MODEL_CACHE=/home/qwen_model_cache
```

我没有把模型权重放进本地轻量备份。原因很简单：模型权重体积大，可以重新下载；脚本、报告、参数、日志摘要和测试结果体积小，但决定实验能不能复现，所以优先保存这些资料。

## 启动 baseline 服务

baseline 使用 vLLM OpenAI API 服务：

```bash
export VLLM_USE_MODELSCOPE=true

vllm serve Qwen/Qwen3.6-27B-FP8 \
  --served-model-name qwen3.6-27b-fp8-amd-8k \
  --host 0.0.0.0 \
  --port 8000 \
  --download-dir /home/qwen_model_cache \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --disable-uvicorn-access-log
```

服务启动后先检查 `/v1/models`，再做短输出：

```bash
curl -s http://127.0.0.1:8000/v1/models
```

这个步骤的意义是把问题分层：先确认服务层和模型加载可用，再进入 MTP、8K 和 KV cache 实验。

## MTP 投机解码配置

vLLM 中启用 Qwen MTP 路径时，核心参数是：

```bash
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":4}'
```

我没有一开始只跑一个配置，而是扫了 `num_speculative_tokens=1/2/3/4`。这样可以观察 MTP token 数量变化对 decode throughput 的影响。

## 第一组：128-token decode 小测试

这一组用于快速确认 MTP 是否生效：

| 配置 | median tok/s | mean tok/s | median latency |
|---|---:|---:|---:|
| baseline | 21.63 | - | 5.82s |
| mtp_t1 | 34.85 | 31.02 | 3.67s |
| mtp_t2 | 49.33 | 48.23 | 2.59s |
| mtp_t3 | 62.00 | 52.06 | 2.06s |
| mtp_t4 | 65.84 | 55.08 | 1.94s |

这一轮最优是 `mtp_t4`，相对 baseline median decode throughput 约 `3.04x`。

这组结果说明 MTP 路径确实生效，但 128-token 只能算小样本，还不能直接作为文章里的核心性能结论。

## 第二组：near-8K needle retrieval 正确性验证

长上下文不能只看模型有没有返回 token。我使用 needle retrieval：在长上下文里插入隐藏 key，要求模型只返回 key。

MTP t4 near-8K 结果：

| case | repeats | needle | total tokens | completion tokens | 结果 |
|---|---:|---|---:|---:|---|
| 8k-near-safe | 100 | violet-5082 | 8002 | 145 | PASS |

这说明在当前 8K 配置下，MTP t4 不只是能生成，还能通过语义检索 gate。

## 第三组：长输出 decode 与 FP8 KV cache

在 512-token decode 测试中：

| 配置 | latency | decode tok/s | completion tokens |
|---|---:|---:|---:|
| baseline_512 | 23.41s | 21.87 | 512 |
| mtp_t2_512 | 10.42s | 49.13 | 512 |
| mtp_t3_512 | 8.23s | 62.20 | 512 |
| mtp_t4_512 | 8.24s | 62.13 | 512 |

这说明 MTP 的收益不只出现在 128-token 小测试里，在更长 decode 中仍然明显。

随后我测试了 `--kv-cache-dtype fp8`：

| 配置 | latency | decode tok/s | completion tokens |
|---|---:|---:|---:|
| kv_baseline_512 | 26.97s | 18.98 | 512 |
| kv_mtp_t4_512 | 8.89s | 57.59 | 512 |

FP8 KV near-8K correctness：

| case | status | prompt tokens | total tokens | completion tokens | 结果 |
|---|---:|---:|---:|---:|---|
| kv_baseline_8k_near_safe | 200 | 7858 | 8006 | 148 | PASS |
| kv_mtp_t4_8k_near_safe | 200 | 7861 | 8053 | 192 | PASS |

FP8 KV baseline 在这组 512-token 测试里比默认 KV 慢一些，这说明 FP8 KV 并不一定天然更快，它更多涉及显存、kernel 和调度之间的权衡。但它能启动，并且能通过 near-8K correctness gate。

## 第四组：强制 1024-token decode 严格对照

上一轮 `max_tokens=1024` 中，有些配置因为 prompt 自然结束，只生成了 603 tokens。为了避免这个问题，我重新设计了 prompt：要求模型持续生成合成 benchmark records，直到被 `max_tokens` 截断。

这一轮所有配置都生成完整 `1024,1024` completion tokens：

| 配置 | repeat | median latency | median tok/s | mean tok/s | full 1024 |
|---|---:|---:|---:|---:|---|
| baseline | 2 | 47.73s | 21.46 | 21.46 | yes |
| mtp_t2 | 2 | 24.14s | 42.41 | 42.41 | yes |
| mtp_t3 | 2 | 21.04s | 48.68 | 48.68 | yes |
| mtp_t4 | 2 | 18.80s | 54.48 | 54.48 | yes |

相对 baseline 的 decode throughput 加速：

| 配置 | speedup |
|---|---:|
| mtp_t2 | 1.98x |
| mtp_t3 | 2.27x |
| mtp_t4 | 2.54x |

相对 baseline 的 median latency 降低：

| 配置 | latency reduction |
|---|---:|
| mtp_t2 | 49.41% |
| mtp_t3 | 55.92% |
| mtp_t4 | 60.62% |

这是目前最适合写进文章的 MTP 性能证据，因为它避免了不同配置生成长度不一致的问题。

## 复现资料

本轮沉淀的轻量复现资料如下：

```text
qwen27b_rocm_mtp_repro_20260525/
  README.md
  notes/
    official_references.md
  scripts/
    start_qwen36_27b_fp8_8k_baseline.sh
    start_qwen36_27b_fp8_8k_mtp.sh
    run_decode_bench.py
    run_needle_gate.py
    run_mtp_sweep.sh
    run_round3_long_decode_and_kv.sh
    run_round4_forced_1024_decode.sh
  reports/
    round1_status_20260525.md
    round2_mtp_sweep_and_8k_gate_20260525.md
    round3_long_decode_and_kv_20260525.md
    round4_forced_1024_decode_20260525.md
```

本地轻量备份：

```text
D:/model/artifacts/qwen27b_rocm_mtp_repro_20260525_light_repro_latest.zip
```

## 结论

这轮实验已经把 Qwen3.6-27B-FP8 在 AMD ROCm DSW 上的研究链路跑通到一个比较完整的状态：

1. vLLM ROCm 服务可以启动；
2. 短输出和 `/v1/models` 可用；
3. MTP `qwen3_next_mtp` 路径可以启用；
4. near-8K needle retrieval 通过；
5. FP8 KV cache 可启动并通过 near-8K correctness gate；
6. 强制 1024-token decode 中，MTP t4 相对 baseline 达到约 `2.54x` decode throughput。

当前最稳的结论不是“已经达到生产 serving 最优性能”，而是：在这套 AMD ROCm/vLLM 环境中，Qwen3.6-27B-FP8 已经形成了可运行、可验证、可复现的 MTP 解码加速研究 baseline。下一步可以继续做并发吞吐、32K/更长上下文、AMD Quark 量化，以及 llama.cpp/GGUF 工程对照。

## 参考资料

- Qwen3.6-27B-FP8 模型页：https://huggingface.co/Qwen/Qwen3.6-27B-FP8
- vLLM MTP 文档：https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/
- ROCm vLLM 优化文档：https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html

