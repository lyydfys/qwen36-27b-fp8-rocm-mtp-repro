# Round 4 - Forced 1024-Token Decode Comparison

Date: 2026-05-25

## Scope

Round 3 showed that MTP remained useful for longer decode, but the original
`max_tokens=1024` prompt sometimes ended naturally at 603 completion tokens.
Round 4 fixes that by using a prompt that explicitly asks the model to keep
generating synthetic benchmark records until it is cut off by `max_tokens`.

This gives a stricter decode comparison because every measured row produced the
same number of completion tokens.

Remote working directory:

```text
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525
```

Remote evidence files:

```text
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/forced_1024_decode_20260525_152019.jsonl
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/forced_1024_decode_summary_20260525_152019.txt
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/forced_1024_prompt_20260525_152019.txt
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/scripts/run_round4_forced_1024_decode.sh
```

Model weights remain under:

```text
/home/qwen_model_cache
```

The model cache is intentionally not included in lightweight backups.

## Test Design

Common setup:

- Model: `Qwen/Qwen3.6-27B-FP8`
- Backend: vLLM ROCm
- Endpoint: `/v1/completions`
- `max_model_len=8192`
- `max_tokens=1024`
- `temperature=0`
- `repeat=2`
- KV cache: default path, not FP8 KV cache

Prompt design:

- Generate synthetic benchmark records.
- Keep increasing record IDs.
- Do not summarize.
- Do not stop early.
- Continue until the max token limit cuts off the response.

This prompt was selected to avoid the natural early stop seen in Round 3.

## Results

All rows reached the full requested decode length:

```text
completion_tokens = 1024,1024
all_full_1024 = True
```

| label | repeat | median latency | median decode tok/s | mean decode tok/s | full 1024 |
|---|---:|---:|---:|---:|---|
| baseline | 2 | 47.726203s | 21.455850 | 21.455850 | yes |
| mtp_t2 | 2 | 24.144029s | 42.413046 | 42.413046 | yes |
| mtp_t3 | 2 | 21.039062s | 48.675316 | 48.675316 | yes |
| mtp_t4 | 2 | 18.795832s | 54.481873 | 54.481873 | yes |

Speedup over baseline median decode throughput:

| label | median tok/s | speedup vs baseline |
|---|---:|---:|
| baseline | 21.455850 | 1.00x |
| mtp_t2 | 42.413046 | 1.98x |
| mtp_t3 | 48.675316 | 2.27x |
| mtp_t4 | 54.481873 | 2.54x |

Latency reduction compared with baseline:

| label | median latency | latency reduction |
|---|---:|---:|
| mtp_t2 | 24.144029s | 49.41% |
| mtp_t3 | 21.039062s | 55.92% |
| mtp_t4 | 18.795832s | 60.62% |

## Interpretation

This is the cleanest MTP decode evidence so far in the Qwen3.6-27B-FP8 ROCm
study:

- The prompt no longer exits early.
- Every configuration produced full 1024-token completions.
- MTP t4 remained the best setting in this test.
- The observed decode throughput improved from `21.46 tok/s` to `54.48 tok/s`.
- The observed median latency dropped from `47.73s` to `18.80s`.

This remains a single-request decode-heavy benchmark, not a production serving
benchmark. It is suitable as a controlled research baseline for the article and
report, especially when combined with the earlier near-8K needle gate and FP8 KV
cache correctness checks.

## Next Report Use

For the article/report, the clean conclusion should be:

> Under a forced 1024-token decode workload, Qwen3.6-27B-FP8 on the tested AMD
> ROCm/vLLM DSW environment showed a clear MTP gain. With
> `num_speculative_tokens=4`, median decode throughput improved from
> `21.46 tok/s` to `54.48 tok/s`, about `2.54x`, while all runs generated the
> full 1024 completion tokens.

