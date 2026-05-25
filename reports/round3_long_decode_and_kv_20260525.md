# Round 3 - Long Decode and FP8 KV Cache

Date: 2026-05-25

## Scope

This round completed the two follow-up experiments after the MTP token sweep:

- Long-output decode tests with `max_tokens=512/1024`.
- FP8 KV cache tests, including decode speed and a near-8K needle gate.

Remote working directory:

```text
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525
```

Model cache:

```text
/home/qwen_model_cache
```

The model cache is intentionally not included in local or persistent lightweight
backups.

## Long Decode Experiment

Common setup:

- Model: `Qwen/Qwen3.6-27B-FP8`
- vLLM backend: ROCm build
- Endpoint: `/v1/completions`
- Prompt: `Count from 1 to 100, separated by spaces:`
- `temperature=0`

Observed results:

| label | latency | decode tok/s | completion tokens |
|---|---:|---:|---:|
| baseline_512 | 23.411724s | 21.869385 | 512 |
| baseline_1024 | 27.501256s | 21.926271 | 603 |
| mtp_t2_512 | 10.422068s | 49.126525 | 512 |
| mtp_t2_1024 | 12.133587s | 49.696763 | 603 |
| mtp_t3_512 | 8.232156s | 62.195126 | 512 |
| mtp_t3_1024 | 9.424770s | 63.980340 | 603 |
| mtp_t4_512 | 8.241094s | 62.127671 | 512 |
| mtp_t4_1024 | 15.732794s | 65.086979 | 1024 |

Interpretation:

- MTP remains beneficial beyond the earlier 128-token microbenchmark.
- For `max_tokens=512`, baseline was about `21.87 tok/s`; MTP t2/t3/t4
  reached about `49.13/62.20/62.13 tok/s`.
- For this prompt, several `max_tokens=1024` requests ended naturally at 603
  completion tokens, so the 1024-row comparison is useful but not perfectly
  controlled. A stricter next run should use a prompt that forces longer output.
- `mtp_t4_1024` did generate 1024 completion tokens and reached `65.09 tok/s`.

## FP8 KV Cache Experiment

Common setup:

- Added `--kv-cache-dtype fp8` to the vLLM launch path.
- Tested both baseline and MTP t4.
- Ran a 512-token decode test and a near-8K needle retrieval gate.

Observed decode results:

| label | latency | decode tok/s | completion tokens |
|---|---:|---:|---:|
| kv_baseline_512 | 26.970590s | 18.983641 | 512 |
| kv_mtp_t4_512 | 8.889985s | 57.592896 | 512 |

Observed near-8K correctness gate:

| case | status | prompt tokens | total tokens | completion tokens | result |
|---|---:|---:|---:|---:|---|
| kv_baseline_8k_near_safe | 200 | 7858 | 8006 | 148 | PASS |
| kv_mtp_t4_8k_near_safe | 200 | 7861 | 8053 | 192 | PASS |

Interpretation:

- FP8 KV cache starts successfully on the tested ROCm/vLLM stack.
- FP8 KV cache preserved the near-8K needle retrieval gate in both baseline and
  MTP t4 configurations.
- In this small 512-token decode test, FP8 KV baseline was slower than the
  default KV baseline from the long-decode experiment (`18.98 tok/s` vs
  `21.87 tok/s`). This should be treated as a kernel/runtime tradeoff result,
  not as a final judgment on FP8 KV cache.
- FP8 KV + MTP t4 remained strong at `57.59 tok/s`.

## Evidence Files

Remote evidence patterns:

```text
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/long_decode_512_1024_*.jsonl
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/kv_fp8_experiment_*.jsonl
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/logs/round3_driver_*.log
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/logs/round3_launch_fixed_*.log
```

Local lightweight backup directory:

```text
D:/model/artifacts/qwen27b_rocm_mtp_repro_20260525
```

## Current Conclusion

The two requested experiments are complete:

1. Long decode confirms that Qwen3.6-27B-FP8 MTP on AMD ROCm remains beneficial
   beyond the 128-token benchmark.
2. FP8 KV cache can start, pass near-8K correctness, and still work with MTP t4.

The best current research baseline for decode-heavy workloads remains MTP t3/t4.
For a publication-grade performance table, the next controlled run should use a
prompt that forces the same 1024-token completion length across baseline and MTP.

