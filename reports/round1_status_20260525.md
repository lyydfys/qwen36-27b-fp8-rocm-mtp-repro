# Round 1 Status - Qwen3.6-27B-FP8 on AMD ROCm

Date: 2026-05-25

## Result

Round 1 completed a working vLLM 8K baseline and an MTP comparison for
`Qwen/Qwen3.6-27B-FP8` on the ModelScope DSW AMD ROCm instance.

## Environment

- Remote workspace: `/mnt/workspace`
- Remote repro dir: `/mnt/workspace/qwen27b_rocm_mtp_repro_20260525`
- Model cache: `/home/qwen_model_cache`
- Python: 3.12
- PyTorch: `2.10.0+git...`
- HIP: `7.2.53211`
- vLLM: `0.20.1+rocm721`
- GPU: 1 AMD GPU, about 192GB VRAM

## Completed

1. Environment probe completed.
2. Current vLLM contains both `qwen3_next_mtp` and generic `mtp` strings.
3. `Qwen/Qwen3.6-27B-FP8` downloaded successfully, 66 checkpoint shards.
4. FP8 8K baseline service started successfully.
5. `/v1/models` returned HTTP 200.
6. Short chat returned HTTP 200; with a larger max token budget, final content includes `5`.
7. Baseline 128-token completions decode benchmark completed.
8. MTP service started with `qwen3_next_mtp`; logs showed `llm_base_proposer`.
9. MTP 128-token completions decode benchmark completed.
10. MTP needle retrieval smoke test passed at 2k-ish and 6k-ish prompt sizes.

## Performance

| Config | Model | max_model_len | speculative | latency | completion tokens | decode tok/s |
|---|---|---:|---|---:|---:|---:|
| baseline | `Qwen/Qwen3.6-27B-FP8` | 8192 | off | 5.831s | 128 | 21.952 |
| MTP | `Qwen/Qwen3.6-27B-FP8` | 8192 | `qwen3_next_mtp`, 2 tokens | 4.506s | 128 | 28.407 |

MTP vs baseline:

- Latency improved by about 22.7%.
- Decode throughput improved by about 29.4%.

## Correctness Smoke Test

| Case | prompt tokens | total tokens | latency | result |
|---|---:|---:|---:|---|
| 2k-ish | 2529 | 2785 | 7.968s | PASS |
| 6k-ish | 5940 | 6196 | 11.444s | PASS |

The first needle attempt with `max_tokens=32` failed because Qwen started with
`<think>` and did not reach the final answer. Retrying with `max_tokens=256`
passed. Future correctness gates should record prompt tokens, output limit, and
reasoning behavior together.

## Next

1. Sweep `num_speculative_tokens=1/2/3/4`.
2. Add a stricter 8K needle retrieval gate near the model length limit.
3. Test FP8 KV cache.
4. Evaluate AMD Quark FP8/KV quantization.
5. Add llama.cpp GGUF as an engineering-shape comparison only.

