# 在 AMD ROCm DSW 上部署 Qwen3.6-27B-FP8：vLLM、MTP 解码加速与小并发压测

## 摘要

本文记录一次在 ModelScope DSW AMD GPU 实例上完成的 Qwen3.6-27B-FP8 推理实践。实验重点不是单纯证明模型可以启动，而是围绕 vLLM ROCm 服务、Qwen MTP 投机解码、near-8K 长上下文正确性验证、FP8 KV cache 和小并发 serving 压测，整理一套可复现、可复查、可继续扩展的 AMD GPU 大模型推理 baseline。

这轮实验中，我把模型权重和复现资料分开管理：模型权重体积大，可以按需重新下载；脚本、参数、日志摘要、图表和报告体积小，但决定实验能否复现，因此优先保存。最终沉淀了启动脚本、benchmark 脚本、near-8K correctness gate、强制 1024-token decode 对照、小并发压测结果，以及 GitHub 和 Hugging Face Space 两个公开入口。

核心结论如下：

- vLLM ROCm 服务可用于 Qwen3.6-27B-FP8 的研究型部署；
- near-8K MTP gate 和 FP8 KV cache gate 均有通过记录；
- 在强制 1024-token decode 口径下，baseline median decode throughput 为 21.46 tok/s，MTP t4 为 54.48 tok/s，约 2.54x；
- 在 concurrency=1/2/4 的小并发压测中，MTP t4 aggregate completion throughput 相对 baseline 约为 2.52x / 2.79x / 2.58x；
- 当前结果适合作为 ROCm/vLLM/MTP 推理优化 baseline，不应被解读为生产 serving benchmark 或跨硬件排名。

公开入口：

- GitHub 仓库：https://github.com/lyydfys/qwen36-27b-fp8-rocm-mtp-repro
- Hugging Face Space：https://huggingface.co/spaces/lyydfys/qwen36-27b-fp8-rocm-mtp-report
- Gallery Notebook 源文件：https://github.com/lyydfys/qwen36-27b-fp8-rocm-mtp-repro/tree/master/publish/gallery

## 为什么选择 Qwen3.6-27B-FP8

Qwen3.6-27B-FP8 适合做 AMD ROCm 大模型推理实践，原因有三个。

第一，模型规模足够大。27B 级别模型会真实暴露显存、KV cache、prefill、decode 和并发调度问题，不会停留在“启动一个小模型”的演示层面。

第二，FP8 和 MTP 都和推理效率有关。FP8 让显存和带宽压力下降，MTP speculative decoding 则尝试通过一次预测多个 token 来改善 decode-heavy 场景的吞吐。二者都适合在 AMD GPU / ROCm 环境下做工程验证。

第三，vLLM 对 ROCm 的支持正在快速演进。只记录“能不能启动”价值有限，更有意义的是建立 correctness gate 和性能测试脚本，后续可以继续比较不同 ROCm、vLLM、AITER、MTP tokens、KV cache 和量化配置。

## 实验环境与目录设计

本次实验在 ModelScope DSW AMD GPU 实例上完成，环境摘要如下：

| 项目 | 记录 |
| --- | --- |
| 平台 | ModelScope DSW AMD GPU |
| Python | 3.12 |
| PyTorch | 2.10.0+git... |
| HIP | 7.2.53211 |
| vLLM | 0.20.1+rocm721 |
| GPU | 单张 AMD GPU，约 192GB 显存 |
| 模型 | Qwen/Qwen3.6-27B-FP8 |

目录设计采用“权重和复现资料分离”的方式：

```bash
export REPRO_DIR=/mnt/workspace/qwen27b_rocm_mtp_repro_20260525
export MODEL_CACHE=/home/qwen_model_cache
export MODEL_DIR=$MODEL_CACHE/Qwen/Qwen3.6-27B-FP8
```

其中：

- `MODEL_CACHE` 用来放模型权重；
- `REPRO_DIR` 用来放脚本、报告、日志摘要、prompt 和图表；
- 本地备份和 GitHub 仓库不保存模型权重，只保存可复现资料。

这种设计的原则很简单：权重文件体积大，必要时可以重新下载；脚本、参数、日志摘要和报告体积小，但决定实验能否恢复，所以需要优先持久化。

## vLLM ROCm 服务启动思路

实验中我使用 vLLM OpenAI API server 作为服务层。baseline 和 MTP 两类服务的启动参数保持尽量一致，只改变 MTP 相关配置，便于对照。

baseline 启动脚本的核心思路如下：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen36-27b-fp8-baseline \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --disable-uvicorn-access-log
```

MTP t4 服务在 baseline 基础上增加 speculative decoding 配置：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen36-27b-fp8-mtp-t4 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --speculative-config '{"model":"qwen_mtp","num_speculative_tokens":4}' \
  --disable-uvicorn-access-log
```

实际脚本中还包含日志文件、端口检查和恢复说明。启动后先做 `/v1/models` 和短输出检查，再进入长上下文和性能测试。

## 正确性验证：near-8K needle gate

性能测试之前必须先做 correctness gate。长上下文测试不能只看模型是否返回 token，而要确认模型能从长文本中取回指定信息。

本次使用 near-8K needle retrieval：构造接近 8K token 的上下文，在其中插入隐藏 key，然后要求模型只返回该 key。只有命中隐藏 key，才认为该配置通过。

关键结果如下：

| case | prompt tokens | total tokens | completion tokens | 结果 |
| --- | ---: | ---: | ---: | --- |
| MTP t4 near-8K needle | 7857 | 8002 | 145 | PASS |
| FP8 KV baseline near-8K | 7858 | 8006 | 148 | PASS |
| FP8 KV + MTP t4 near-8K | 7861 | 8053 | 192 | PASS |

这一步的意义是把“服务能启动”提升到“长上下文语义可验证”。如果没有 correctness gate，后面的吞吐数字就容易失去可信度。

## MTP tokens sweep

为了观察 MTP speculative decoding 的收益，我先做了一轮 128-token decode sweep：

| 配置 | median tok/s | 相对 baseline |
| --- | ---: | ---: |
| baseline | 21.63 | 1.00x |
| MTP t1 | 34.85 | 1.61x |
| MTP t2 | 49.33 | 2.28x |
| MTP t3 | 62.00 | 2.87x |
| MTP t4 | 65.84 | 3.04x |

这一轮说明 MTP 方向有效，但 128-token 生成太短，容易受首 token、调度和请求波动影响。因此我继续做了更严格的强制 1024-token decode 对照。

## 强制 1024-token decode 对照

强制 1024-token decode 是本轮最干净的单请求性能结果。所有配置都生成完整 `1024` completion tokens，避免了“某个配置提前结束导致吞吐虚高”的问题。

| 配置 | repeat | median latency | median decode tok/s | 是否完整 1024 |
| --- | ---: | ---: | ---: | --- |
| baseline | 2 | 47.73s | 21.46 | yes |
| MTP t2 | 2 | 24.14s | 42.41 | yes |
| MTP t3 | 2 | 21.04s | 48.68 | yes |
| MTP t4 | 2 | 18.80s | 54.48 | yes |

在这个口径下，MTP t4 相对 baseline 的 median decode throughput 约为 `2.54x`。这是本文最适合引用的性能结论，因为输入、输出长度和测试脚本都更一致。

## 小并发 serving 压测

单请求加速成立后，我继续做了小并发 serving 对照。测试仍然使用强制 1024-token 输出，观察 MTP t4 在 concurrency=1/2/4 下是否还能保持收益。

| 配置 | concurrency | requests | success rate | aggregate tok/s | p50 latency | 是否完整 1024 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 1 | 2 | 1.0 | 21.51 | 47.61s | yes |
| baseline | 2 | 2 | 1.0 | 37.23 | 55.00s | yes |
| baseline | 4 | 4 | 1.0 | 73.42 | 55.79s | yes |
| MTP t4 | 1 | 2 | 1.0 | 54.28 | 18.87s | yes |
| MTP t4 | 2 | 2 | 1.0 | 103.73 | 19.74s | yes |
| MTP t4 | 4 | 4 | 1.0 | 189.73 | 20.78s | yes |

换算后，MTP t4 在 concurrency=1/2/4 下分别约为 `2.52x / 2.79x / 2.58x`。这说明 MTP 收益不是只存在于孤立单请求中，在小并发场景里仍然能保持方向一致。

## FP8 KV cache 观察

FP8 KV cache 在长上下文场景中主要价值是降低 KV cache 显存压力。本轮实验中，FP8 KV 路径可以启动，并且 near-8K gate 有通过记录。

不过在小规模 decode 测试里，FP8 KV baseline 吞吐低于默认 KV 路径。因此我没有把它写成“必然更快”，而是把它作为一个 runtime/kernel tradeoff 来看：它更可能在更长上下文、更高并发或显存压力更大的场景中体现价值，需要后续继续测试。

## 复现资料与公开入口

为了让实验不是一次性结果，我整理了如下文件：

```text
scripts/
  start_qwen36_27b_fp8_8k_baseline.sh
  start_qwen36_27b_fp8_8k_mtp.sh
  run_mtp_sweep.sh
  run_needle_gate.py
  run_round4_forced_1024_decode.sh
  run_round5_concurrency_serving.sh

reports/
  round2_mtp_sweep_and_8k_gate_20260525.md
  round3_long_decode_and_kv_20260525.md
  round4_forced_1024_decode_20260525.md
  round5_concurrency_serving_20260525.md
  qwen36_27b_rocm_mtp_research_report_20260525.md

publish/gallery/
  qwen36_27b_fp8_rocm_vllm_mtp_gallery.ipynb
  requirements.txt
```

公开入口如下：

- GitHub 仓库：https://github.com/lyydfys/qwen36-27b-fp8-rocm-mtp-repro
- Hugging Face Space：https://huggingface.co/spaces/lyydfys/qwen36-27b-fp8-rocm-mtp-report
- Gallery Notebook 源文件：https://github.com/lyydfys/qwen36-27b-fp8-rocm-mtp-repro/tree/master/publish/gallery

其中 Gallery Notebook 的默认 Cell 不下载模型、不启动完整 vLLM 服务，只做环境探测、实验数据复核、加速比计算、图表生成和复现清单输出。完整 live check 被放在可选 Cell 中，需要手动设置环境变量开启。

## 边界与下一步

这次实验应被理解为一个可复现的 ROCm/vLLM/MTP 研究 baseline，而不是生产级 serving benchmark，也不是不同硬件之间的排名。

当前结果已经说明：

- AMD ROCm 环境中可以围绕 Qwen3.6-27B-FP8 建立 vLLM 推理实验链路；
- MTP 在 decode-heavy 场景中有明确收益；
- correctness gate 对长上下文实验非常必要；
- FP8 KV cache 可用，但需要在更长上下文和更高显存压力下继续验证；
- 小并发 serving 结果支持继续往真实 serving 场景扩展。

下一步计划包括：

1. 扩展 16K / 32K context correctness gate；
2. 增加更稳定的并发请求数和输入输出 token 形状；
3. 比较默认 KV 与 FP8 KV 在长上下文和高并发下的差异；
4. 引入 AMD Quark 等量化工具做权重量化实验；
5. 用 llama.cpp / GGUF 路线做工程部署对照，但不把它和 vLLM 直接做同算法性能排名。

对我来说，这轮实验的价值在于把“模型能不能启动”推进到了“如何验证 correctness、如何设计公平 decode benchmark、MTP 在 ROCm 上是否有稳定收益、后续量化和 serving 优化应该从哪里继续”的阶段。这个 baseline 后续可以继续复用，而不是每次都从零开始排查环境和启动参数。

## 参考资料

- Qwen3.6-27B-FP8 模型卡：https://huggingface.co/Qwen/Qwen3.6-27B-FP8
- vLLM MTP 文档：https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/
- ROCm vLLM 优化文档：https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- GitHub 复现仓库：https://github.com/lyydfys/qwen36-27b-fp8-rocm-mtp-repro
- Hugging Face Space 报告页：https://huggingface.co/spaces/lyydfys/qwen36-27b-fp8-rocm-mtp-report
