# Resume Project Description

## Chinese

**AMD ROCm 上的 Qwen3.6-27B-FP8 vLLM 部署与 MTP 解码加速研究**

在 ModelScope DSW AMD GPU 环境中完成 `Qwen/Qwen3.6-27B-FP8` 的 vLLM ROCm
部署、MTP 投机解码、near-8K 长上下文正确性验证和 FP8 KV cache 实验。构建了可复现脚本、环境记录、needle retrieval gate、强制 1024-token decode
benchmark 与小并发 serving 压测链路。严格 1024-token 单请求测试中，MTP
`num_speculative_tokens=4` 将 median decode throughput 从 `21.46 tok/s`
提升到 `54.48 tok/s`，约 `2.54x`。项目沉淀为 GitHub 仓库、HF Space 展示页和中文技术文章，可作为 AMD GPU 大模型推理优化研究 baseline。

**关键词**：AMD ROCm、vLLM、Qwen3.6、FP8、MTP、speculative decoding、KV cache、LLM serving benchmark

## English

**Qwen3.6-27B-FP8 Deployment and MTP Decode Acceleration on AMD ROCm**

Built a reproducible vLLM ROCm deployment baseline for `Qwen/Qwen3.6-27B-FP8`
on a ModelScope DSW AMD GPU instance. Implemented MTP speculative decoding
experiments, near-8K needle retrieval correctness gates, FP8 KV cache validation,
forced 1024-token decode benchmarks, and small-concurrency serving tests. In the
strict 1024-token single-request benchmark, MTP with `num_speculative_tokens=4`
improved median decode throughput from `21.46 tok/s` to `54.48 tok/s`
(`2.54x`). Packaged scripts, reports, charts, and publication materials into a
GitHub repository and Hugging Face Space, excluding model weights.

**Keywords**: AMD ROCm, vLLM, Qwen3.6, FP8, MTP, speculative decoding, KV cache, LLM serving benchmark

