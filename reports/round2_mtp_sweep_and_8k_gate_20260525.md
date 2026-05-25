# Round 2 - MTP Token Sweep and Near-8K Needle Gate

Date: 2026-05-25

## Scope

This round extends the first Qwen3.6-27B-FP8 ROCm/vLLM baseline with:

- A repeated baseline decode benchmark.
- MTP `num_speculative_tokens=1/2/3/4` sweep.
- A stricter near-8K needle retrieval gate.

All remote artifacts are under:

```text
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525
```

Model weights are cached under:

```text
/home/qwen_model_cache
```

The model cache is intentionally not included in the local backup.

## Repeated Baseline

Model:

```text
Qwen/Qwen3.6-27B-FP8
```

Service name:

```text
qwen3.6-27b-fp8-amd-8k
```

Result:

| label | repeat | median latency | median decode tok/s |
|---|---:|---:|---:|
| baseline_repeat | 3 | 5.817664s | 21.631446 |

## MTP Sweep

MTP method:

```text
qwen3_next_mtp
```

Common benchmark:

- Endpoint: `/v1/completions`
- Prompt: `Count from 1 to 100, separated by spaces:`
- `max_tokens=128`
- `temperature=0`
- 3 repeats for each configuration

| label | median tok/s | mean tok/s | median latency |
|---|---:|---:|---:|
| mtp_t1 | 34.846241 | 31.024518 | 3.673280s |
| mtp_t2 | 49.332937 | 48.226040 | 2.594615s |
| mtp_t3 | 62.000804 | 52.063673 | 2.064489s |
| mtp_t4 | 65.839519 | 55.082900 | 1.944121s |

Compared with the repeated baseline median `21.631446 tok/s`, the best observed
configuration in this round was `mtp_t4`.

Approximate median decode speedup:

```text
65.839519 / 21.631446 = 3.04x
```

This is a small decode-only benchmark, not a full serving benchmark. It is still
strong evidence that Qwen3.6-27B-FP8 MTP is active and beneficial on the tested
AMD ROCm DSW environment.

## Near-8K Needle Gate

Current service:

```text
qwen3.6-27b-fp8-amd-8k-mtp-t4
```

Strict gate result:

| case | repeats | needle | total tokens | completion tokens | result |
|---|---:|---|---:|---:|---|
| 8k-near-safe | 100 | violet-5082 | 8002 | 145 | PASS |

Additional smoke cases from the same round:

| case | repeats | total tokens | result |
|---|---:|---:|---|
| 2k | 40 | 3298 | PASS |
| 6k | 95 | 7660 | PASS |
| 8k-near | 120 | null | FAIL / rejected near limit |

The rejected `8k-near:120` case is useful as a boundary marker: with
`max_model_len=8192` and a non-trivial output budget, pushing the prompt too
close to 8K can be rejected before generation. The safe near-8K case reached
`total_tokens=8002` and successfully retrieved the hidden key.

## Evidence Files

Remote evidence patterns:

```text
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/logs/mtp_tokens_sweep_driver_*.log
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/mtp_tokens_sweep_*.jsonl
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/mtp_tokens_sweep_summary_*.txt
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/baseline_repeat_decode_*.jsonl
/mnt/workspace/qwen27b_rocm_mtp_repro_20260525/reports/needle_gate_mtp_t4_8k_near*.jsonl
```

Local lightweight backup:

```text
D:/model/artifacts/qwen27b_rocm_mtp_repro_20260525
```

## Next Research Steps

1. Repeat the best `mtp_t4` result on a second prompt style to check whether the
   speedup is prompt-specific.
2. Run a longer generation benchmark, for example `max_tokens=512`, because MTP
   should mainly help decode-heavy workloads.
3. Test FP8 KV cache and record memory impact.
4. Evaluate AMD Quark FP8/KV quantization.
5. Add a llama.cpp GGUF comparison only as an engineering deployment comparison,
   not as a same-algorithm quantization comparison.

