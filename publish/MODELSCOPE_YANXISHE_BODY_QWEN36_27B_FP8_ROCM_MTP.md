## 摘要

本文记录我在 ModelScope DSW AMD GPU 实例上，对 `Qwen/Qwen3.6-27B-FP8` 进行 ROCm/vLLM 部署、MTP 投机解码、near-8K 长上下文正确性验证、FP8 KV cache 验证和小并发 serving 压测的工程实践。

这次实验的目标不是只证明“模型能启动”，而是整理出一套可复现的 AMD ROCm 大模型推理研究 baseline：服务可启动、短输出可用、near-8K needle retrieval 通过、MTP 在 decode-heavy 场景中有稳定收益，并且脚本、日志摘要、prompt、图表和报告都能在实例重启后恢复。

最干净的单请求结果来自强制 1024-token decode：所有配置都生成完整 `1024,1024` completion tokens。在相同 prompt、相同 `max_tokens=1024` 的口径下，baseline median decode throughput 为 `21.46 tok/s`，MTP `num_speculative_tokens=4` 为 `54.48 tok/s`，约 `2.54x`。

进一步的小并发压测中，MTP t4 在 `concurrency=1/2/4` 下仍然保持优势，aggregate completion throughput 分别达到 `54.28 / 103.73 / 189.73 tok/s`，相对 baseline 约 `2.52x / 2.79x / 2.58x`。所有请求均成功，且都生成完整 1024 completion tokens。

需要说明的是，这不是生产 serving benchmark，也不是不同硬件之间的横向排名。它更适合作为 AMD ROCm 上研究 Qwen3.6-27B-FP8、vLLM、MTP、FP8 KV cache 和后续量化优化的工程基线。

![图一：实验链路与证据闭环](https://raw.githubusercontent.com/lyydfys/qwen36-27b-fp8-rocm-mtp-repro/master/figures/modelscope_yanxishe/figure_00_pipeline_cn.png)

## 选题背景

我选择 Qwen3.6-27B-FP8，是因为它适合做 AMD ROCm 大模型推理研究：模型规模足够大，能暴露 vLLM 服务、长上下文、KV cache、投机解码和并发调度中的真实问题；同时又不像超大模型那样一开始就被权重体积和多机调度复杂度完全淹没。

这轮实验我重点关注四个问题：

1. `Qwen/Qwen3.6-27B-FP8` 能否在 ModelScope DSW AMD GPU 环境中通过 vLLM ROCm 稳定启动；
2. Qwen MTP 路径 `qwen3_next_mtp` 在 AMD ROCm 上是否真的带来 decode 加速；
3. 8K 长上下文和 FP8 KV cache 是否能通过 correctness gate；
4. 单请求加速能否延伸到小并发 serving 场景。

## 环境与目录设计

本次实验环境如下：

| 项目 | 记录 |
|---|---|
| 平台 | ModelScope DSW AMD GPU |
| Python | 3.12 |
| PyTorch | 2.10.0+git... |
| HIP | 7.2.53211 |
| vLLM | 0.20.1+rocm721 |
| GPU | 1 张 AMD GPU，显存约 192GB |
| 模型 | Qwen/Qwen3.6-27B-FP8 |

我把复现资料和模型权重分开管理：

```bash
export REPRO_DIR=/mnt/workspace/qwen27b_rocm_mtp_repro_20260525
export MODEL_CACHE=/home/qwen_model_cache
```

`REPRO_DIR` 保存脚本、报告、prompt、日志摘要和图表；`MODEL_CACHE` 保存模型权重。模型权重体积大，可以按需重新下载；脚本和结果记录体积小，但决定实验能不能恢复，所以优先持久化保存。

## vLLM baseline 启动

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

服务启动后先检查：

```bash
curl -s http://127.0.0.1:8000/v1/models
```

这个步骤用于确认服务层和模型加载可用，再进入 MTP、8K 和 KV cache 实验。

## MTP 配置

启用 Qwen MTP 路径时，核心参数是：

```bash
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":4}'
```

我没有只跑一个配置，而是先扫了 `num_speculative_tokens=1/2/3/4`，再把最优配置 t4 用在长输出和并发测试中。

## MTP tokens sweep

128-token decode 小测试用于确认 MTP 是否生效：

| 配置 | median tok/s | mean tok/s | median latency |
|---|---:|---:|---:|
| baseline | 21.63 | - | 5.82s |
| mtp_t1 | 34.85 | 31.02 | 3.67s |
| mtp_t2 | 49.33 | 48.23 | 2.59s |
| mtp_t3 | 62.00 | 52.06 | 2.06s |
| mtp_t4 | 65.84 | 55.08 | 1.94s |

这一轮最优是 `mtp_t4`，相对 baseline median decode throughput 约 `3.04x`。不过 128-token 只能作为快速验证，不能单独作为核心结论。

## near-8K correctness gate

长上下文不能只看模型有没有返回 token。我采用 needle retrieval：在长上下文中插入隐藏 key，要求模型只返回这个 key。

MTP t4 near-8K 结果：

| case | repeats | needle | total tokens | completion tokens | 结果 |
|---|---:|---|---:|---:|---|
| 8k-near-safe | 100 | violet-5082 | 8002 | 145 | PASS |

这说明当前 8K 配置下，MTP t4 不只是能生成，还能通过语义检索 gate。

## FP8 KV cache

我也测试了 `--kv-cache-dtype fp8`：

| 配置 | latency | decode tok/s | completion tokens |
|---|---:|---:|---:|
| kv_baseline_512 | 26.97s | 18.98 | 512 |
| kv_mtp_t4_512 | 8.89s | 57.59 | 512 |

FP8 KV near-8K correctness：

| case | status | prompt tokens | total tokens | completion tokens | 结果 |
|---|---:|---:|---:|---:|---|
| kv_baseline_8k_near_safe | 200 | 7858 | 8006 | 148 | PASS |
| kv_mtp_t4_8k_near_safe | 200 | 7861 | 8053 | 192 | PASS |

FP8 KV baseline 在这组 512-token 测试里比默认 KV 慢一些。这说明 FP8 KV 不一定天然更快，它更像是显存、kernel 和调度之间的权衡。但它能启动，并且能通过 near-8K correctness gate。

## 强制 1024-token decode 严格对照

上一轮 `max_tokens=1024` 中，有些配置因为 prompt 自然结束，只生成了 603 tokens。为了避免这个问题，我重新设计了 prompt：要求模型持续生成合成 benchmark records，直到被 `max_tokens` 截断。

这一轮所有配置都生成完整 `1024,1024` completion tokens：

![图二：强制 1024-token decode 严格对照](https://raw.githubusercontent.com/lyydfys/qwen36-27b-fp8-rocm-mtp-repro/master/figures/modelscope_yanxishe/figure_01_forced_1024_decode_cn.png)

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

这是目前最适合作为单请求性能结论的数据，因为它避免了不同配置生成长度不一致的问题。

## 小并发 serving 压测

为了观察服务化场景下 MTP 是否仍然有效，我继续做了小并发压测。测试仍然使用强制 1024-token prompt。

![图三：小并发 serving 压测结果](https://raw.githubusercontent.com/lyydfys/qwen36-27b-fp8-rocm-mtp-repro/master/figures/modelscope_yanxishe/figure_02_concurrency_serving_cn.png)

| label | concurrency | requests | success rate | aggregate tok/s | p50 latency | p95 latency | full 1024 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 1 | 2 | 1.0 | 21.51 | 47.61s | 47.64s | yes |
| baseline | 2 | 2 | 1.0 | 37.23 | 55.00s | 55.00s | yes |
| baseline | 4 | 4 | 1.0 | 73.42 | 55.79s | 55.79s | yes |
| mtp_t4 | 1 | 2 | 1.0 | 54.28 | 18.87s | 18.97s | yes |
| mtp_t4 | 2 | 2 | 1.0 | 103.73 | 19.74s | 19.74s | yes |
| mtp_t4 | 4 | 4 | 1.0 | 189.73 | 20.78s | 21.58s | yes |

按 aggregate completion throughput 计算：

| concurrency | baseline tok/s | MTP t4 tok/s | speedup |
|---:|---:|---:|---:|
| 1 | 21.51 | 54.28 | 2.52x |
| 2 | 37.23 | 103.73 | 2.79x |
| 4 | 73.42 | 189.73 | 2.58x |

这说明 MTP t4 的收益没有停留在单请求场景，在这个小并发测试里仍然稳定存在。需要注意的是，这仍然不是生产压测；请求数量较少，prompt 也比较统一，更适合作为 serving research baseline。

## 复现资料

本轮保留的轻量复现资料如下：

```text
qwen27b_rocm_mtp_repro_20260525/
  README.md
  notes/
    official_references.md
  scripts/
    start_qwen36_27b_fp8_8k_baseline.sh
    start_qwen36_27b_fp8_8k_mtp.sh
    run_decode_bench.py
    run_concurrency_bench.py
    run_needle_gate.py
    run_mtp_sweep.sh
    run_round3_long_decode_and_kv.sh
    run_round4_forced_1024_decode.sh
    run_round5_concurrency_serving.sh
  reports/
    round2_mtp_sweep_and_8k_gate_20260525.md
    round3_long_decode_and_kv_20260525.md
    round4_forced_1024_decode_20260525.md
    round5_concurrency_serving_20260525.md
  figures/
    round4_forced_1024_decode.png
    round5_concurrency_serving.png
```

模型权重没有放进轻量备份。保留脚本、日志摘要、结果表和图表，才是后续复现最关键的部分。

## 结论

这轮研究已经形成了一条比较完整的 AMD ROCm / vLLM / Qwen3.6-27B-FP8 baseline：

1. vLLM ROCm 服务可以启动；
2. Qwen MTP `qwen3_next_mtp` 路径可以启用；
3. near-8K needle retrieval 通过；
4. FP8 KV cache 可启动并通过 near-8K correctness gate；
5. 强制 1024-token 单请求中，MTP t4 达到约 `2.54x` decode throughput；
6. 小并发 `concurrency=1/2/4` 中，MTP t4 仍有约 `2.52x / 2.79x / 2.58x` aggregate throughput 提升。

我认为这套结果的价值在于：它把“AMD ROCm 上能不能跑 Qwen3.6-27B-FP8”推进到了“能不能验证正确性、能不能量化 MTP decode 收益、能不能在小并发 serving 下保持收益”。后续可以继续扩展到 16K/32K、更多并发、不同 prompt 分布、AMD Quark 量化和 llama.cpp/GGUF 工程对照。

## 参考资料

- Qwen3.6-27B-FP8 模型页：https://huggingface.co/Qwen/Qwen3.6-27B-FP8
- vLLM MTP 文档：https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/
- ROCm vLLM 优化文档：https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html


