本 Notebook 是一份基于 ModelScope DSW AMD GPU 实例的 Qwen3.6-27B-FP8 推理实践，重点复核 vLLM ROCm 部署、Qwen MTP 投机解码、near-8K correctness gate、FP8 KV cache 和小并发 serving 压测结果。

为了保证 Gallery 自动化运行稳定，默认 Cell 不下载模型权重，也不启动完整 vLLM 大模型服务，而是完成环境探测、实验数据复核、加速比计算、图表生成和复现清单输出。完整 vLLM live check 被放在可选 Cell 中，需要手动设置环境变量后执行。

核心结果：在强制 1024-token decode 口径下，baseline median decode throughput 为 21.46 tok/s，MTP t4 为 54.48 tok/s，约 2.54x；小并发 concurrency=1/2/4 下，MTP t4 aggregate completion throughput 相对 baseline 约 2.52x/2.79x/2.58x。
