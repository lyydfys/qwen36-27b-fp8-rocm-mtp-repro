# Qwen3.6-27B-FP8 on AMD ROCm: vLLM Deployment, MTP Decode Acceleration, and 8K Correctness

Date: 2026-05-25

## Executive Summary

This report summarizes a reproducible engineering study of
`Qwen/Qwen3.6-27B-FP8` on a ModelScope DSW AMD GPU instance using vLLM ROCm.

The study built a baseline service, enabled Qwen MTP speculative decoding,
validated near-8K needle retrieval, tested FP8 KV cache, and finally ran a
strict forced 1024-token decode comparison.

The cleanest result is the forced 1024-token decode benchmark:

| configuration | repeat | median latency | median decode tok/s | full 1024 |
|---|---:|---:|---:|---|
| baseline | 2 | 47.726203s | 21.455850 | yes |
| MTP t2 | 2 | 24.144029s | 42.413046 | yes |
| MTP t3 | 2 | 21.039062s | 48.675316 | yes |
| MTP t4 | 2 | 18.795832s | 54.481873 | yes |

MTP t4 improved median decode throughput by about `2.54x` compared with the
baseline under this single-request decode-heavy workload.

The small-concurrency serving benchmark showed the same direction:

| concurrency | baseline aggregate tok/s | MTP t4 aggregate tok/s | speedup |
|---:|---:|---:|---:|
| 1 | 21.505711 | 54.276744 | 2.52x |
| 2 | 37.232449 | 103.729872 | 2.79x |
| 4 | 73.418453 | 189.734199 | 2.58x |

## Environment

| item | value |
|---|---|
| Platform | ModelScope DSW AMD GPU |
| Python | 3.12 |
| PyTorch | 2.10.0+git... |
| HIP | 7.2.53211 |
| vLLM | 0.20.1+rocm721 |
| GPU | 1 AMD GPU, about 192GB VRAM |
| Model | Qwen/Qwen3.6-27B-FP8 |

## Storage Policy

Remote reproducibility directory:

```text
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525
```

Model cache:

```text
/home/qwen_model_cache
```

The local backup intentionally excludes model weights. It only stores scripts,
reports, prompts, environment notes, and summarized evidence.

## Experiment Matrix

| round | goal | key result |
|---|---|---|
| Round 1 | vLLM ROCm baseline | service and short generation worked |
| Round 2 | MTP token sweep + near-8K gate | MTP t4 reached about 3.04x on 128-token decode; near-8K gate passed |
| Round 3 | long decode + FP8 KV cache | MTP remained beneficial at 512 tokens; FP8 KV cache started and passed near-8K gate |
| Round 4 | strict forced 1024-token decode | MTP t4 reached 54.48 tok/s vs baseline 21.46 tok/s |
| Round 5 | small-concurrency serving | MTP t4 kept about 2.52x-2.79x aggregate throughput speedup |

## Correctness Gates

Near-8K MTP t4 gate:

| case | repeats | total tokens | completion tokens | result |
|---|---:|---:|---:|---|
| 8k-near-safe | 100 | 8002 | 145 | PASS |

FP8 KV near-8K gates:

| case | prompt tokens | total tokens | completion tokens | result |
|---|---:|---:|---:|---|
| kv_baseline_8k_near_safe | 7858 | 8006 | 148 | PASS |
| kv_mtp_t4_8k_near_safe | 7861 | 8053 | 192 | PASS |

## Performance Findings

### 128-token MTP sweep

| label | median tok/s | speedup vs baseline |
|---|---:|---:|
| baseline | 21.631446 | 1.00x |
| mtp_t1 | 34.846241 | 1.61x |
| mtp_t2 | 49.332937 | 2.28x |
| mtp_t3 | 62.000804 | 2.87x |
| mtp_t4 | 65.839519 | 3.04x |

### 512-token long decode

| label | latency | decode tok/s | completion tokens |
|---|---:|---:|---:|
| baseline_512 | 23.411724s | 21.869385 | 512 |
| mtp_t2_512 | 10.422068s | 49.126525 | 512 |
| mtp_t3_512 | 8.232156s | 62.195126 | 512 |
| mtp_t4_512 | 8.241094s | 62.127671 | 512 |

### FP8 KV cache

| label | latency | decode tok/s | completion tokens |
|---|---:|---:|---:|
| kv_baseline_512 | 26.970590s | 18.983641 | 512 |
| kv_mtp_t4_512 | 8.889985s | 57.592896 | 512 |

FP8 KV cache was functional and correctness-safe in this setup, but the small
decode benchmark showed lower baseline throughput than the default KV path.
This should be treated as a runtime/kernel tradeoff rather than a final
conclusion about FP8 KV cache.

### Forced 1024-token decode

| label | repeat | median latency | median tok/s | mean tok/s | all full 1024 |
|---|---:|---:|---:|---:|---|
| baseline | 2 | 47.726203s | 21.455850 | 21.455850 | true |
| mtp_t2 | 2 | 24.144029s | 42.413046 | 42.413046 | true |
| mtp_t3 | 2 | 21.039062s | 48.675316 | 48.675316 | true |
| mtp_t4 | 2 | 18.795832s | 54.481873 | 54.481873 | true |

This is the preferred performance table for publication because all rows reached
the same completion length.

### Small-concurrency serving

| label | concurrency | requests | success rate | aggregate tok/s | p50 latency | p95 latency | all full 1024 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 1 | 2 | 1.0 | 21.505711 | 47.614726s | 47.640604s | true |
| baseline | 2 | 2 | 1.0 | 37.232449 | 55.004424s | 55.004868s | true |
| baseline | 4 | 4 | 1.0 | 73.418453 | 55.785482s | 55.787349s | true |
| mtp_t4 | 1 | 2 | 1.0 | 54.276744 | 18.865812s | 18.970771s | true |
| mtp_t4 | 2 | 2 | 1.0 | 103.729872 | 19.742277s | 19.742702s | true |
| mtp_t4 | 4 | 4 | 1.0 | 189.734199 | 20.777241s | 21.583426s | true |

The small-concurrency benchmark is still compact, but it shows that the MTP t4
gain is not limited to an isolated single-request measurement.

## Evidence Inventory

Remote evidence:

```text
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/mtp_tokens_sweep_*.jsonl
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/needle_gate_mtp_t4_8k_near*.jsonl
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/long_decode_512_1024_*.jsonl
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/kv_fp8_experiment_*.jsonl
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/forced_1024_decode_20260525_152019.jsonl
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/forced_1024_decode_summary_20260525_152019.txt
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/round5_concurrency_serving_20260525_155522.jsonl
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/round5_concurrency_serving_summary_20260525_155522.txt
```

Local lightweight backup:

```text
D:/model/artifacts/qwen27b_rocm_mtp_repro_20260525_light_repro_latest.zip
```

## Boundaries

This study should be described as:

- A single-request decode-heavy benchmark.
- A reproducible ROCm/vLLM research baseline.
- A correctness-gated MTP and FP8 KV cache experiment.

It should not be described as:

- A production serving benchmark.
- A hardware-to-hardware ranking.
- A final throughput result under realistic multi-user concurrency.

## Recommended Next Steps

1. Run concurrent serving tests with fixed request counts and input/output token
   shapes.
2. Test 16K and 32K context after the 8K baseline is fully stable.
3. Compare default KV vs FP8 KV under memory-pressure scenarios.
4. Add AMD Quark quantization experiments.
5. Add llama.cpp/GGUF as an engineering deployment comparison, not as a
   same-algorithm performance comparison.
