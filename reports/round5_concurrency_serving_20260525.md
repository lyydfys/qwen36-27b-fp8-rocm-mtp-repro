# Round 5 - Small-Concurrency Serving Benchmark

Date: 2026-05-25

## Scope

Round 5 extends the single-request forced 1024-token decode benchmark into a
small-concurrency serving test. The goal is to check whether the MTP t4 gain
still appears when multiple requests are in flight.

This is still a compact research benchmark, not a production serving benchmark.

Remote working directory:

```text
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525
```

Remote evidence files:

```text
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/round5_concurrency_serving_20260525_155522.jsonl
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/round5_concurrency_serving_summary_20260525_155522.txt
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/round5_forced_1024_prompt_20260525_155522.txt
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/scripts/run_concurrency_bench.py
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/scripts/run_round5_concurrency_serving.sh
```

## Test Design

Common setup:

- Model: `Qwen/Qwen3.6-27B-FP8`
- Backend: vLLM ROCm
- Endpoint: `/v1/completions`
- `max_model_len=8192`
- `max_tokens=1024`
- `temperature=0`
- Concurrency: `1/2/4`
- Requests: `2/2/4`
- Prompt: forced long synthetic benchmark records
- Compared configs: baseline vs MTP t4

Every successful request produced the full requested 1024 completion tokens.

## Results

| label | concurrency | requests | success rate | wall time | aggregate tok/s | p50 latency | p95 latency | p50 decode tok/s | full 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 1 | 2 | 1.0 | 95.230517s | 21.505711 | 47.614726s | 47.640604s | 21.505958 | yes |
| baseline | 2 | 2 | 1.0 | 55.005783s | 37.232449 | 55.004424s | 55.004868s | 18.616684 | yes |
| baseline | 4 | 4 | 1.0 | 55.789789s | 73.418453 | 55.785482s | 55.787349s | 18.356030 | yes |
| mtp_t4 | 1 | 2 | 1.0 | 37.732551s | 54.276744 | 18.865812s | 18.970771s | 54.279758 | yes |
| mtp_t4 | 2 | 2 | 1.0 | 19.743589s | 103.729872 | 19.742277s | 19.742702s | 51.868384 | yes |
| mtp_t4 | 4 | 4 | 1.0 | 21.588095s | 189.734199 | 20.777241s | 21.583426s | 49.293278 | yes |

Speedup by aggregate completion throughput:

| concurrency | baseline tok/s | MTP t4 tok/s | speedup |
|---:|---:|---:|---:|
| 1 | 21.505711 | 54.276744 | 2.52x |
| 2 | 37.232449 | 103.729872 | 2.79x |
| 4 | 73.418453 | 189.734199 | 2.58x |

## Interpretation

MTP t4 retained a clear advantage under small concurrency:

- `concurrency=1`: aggregate throughput improved from `21.51 tok/s` to
  `54.28 tok/s`.
- `concurrency=2`: aggregate throughput improved from `37.23 tok/s` to
  `103.73 tok/s`.
- `concurrency=4`: aggregate throughput improved from `73.42 tok/s` to
  `189.73 tok/s`.

Latency also remained much lower with MTP t4. For example, at `concurrency=4`,
baseline p50 latency was `55.79s`, while MTP t4 p50 latency was `20.78s`.

All rows had `success_rate=1.0` and `all_full=Yes`, so this is a clean serving
comparison for the article/report.

## Boundary

This benchmark uses a small number of deterministic synthetic requests and is
best treated as a research serving baseline. It does not replace a production
benchmark with larger request counts, varied prompt lengths, streaming behavior,
and multi-user traffic patterns.

