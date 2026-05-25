# Qwen3.6-27B-FP8 on AMD ROCm with vLLM and MTP

Reproducible engineering notes for deploying `Qwen/Qwen3.6-27B-FP8` on a
ModelScope DSW AMD GPU instance with vLLM ROCm, validating near-8K correctness,
and measuring Qwen MTP speculative decoding.

This repository intentionally excludes model weights. It contains scripts,
reports, prompts, result summaries, and publication drafts only.

Public links:

- GitHub: https://github.com/lyydfys/qwen36-27b-fp8-rocm-mtp-repro
- Hugging Face Space: https://huggingface.co/spaces/lyydfys/qwen36-27b-fp8-rocm-mtp-report

## Key Result

Under a forced 1024-token single-request decode workload, all configurations
generated full `1024,1024` completion tokens.

| configuration | median latency | median decode tok/s | speedup |
|---|---:|---:|---:|
| baseline | 47.73s | 21.46 | 1.00x |
| MTP t2 | 24.14s | 42.41 | 1.98x |
| MTP t3 | 21.04s | 48.68 | 2.27x |
| MTP t4 | 18.80s | 54.48 | 2.54x |

Near-8K correctness gates also passed for MTP t4 and FP8 KV cache paths.

Small-concurrency serving results using the same forced 1024-token prompt:

| concurrency | baseline aggregate tok/s | MTP t4 aggregate tok/s | speedup |
|---:|---:|---:|---:|
| 1 | 21.51 | 54.28 | 2.52x |
| 2 | 37.23 | 103.73 | 2.79x |
| 4 | 73.42 | 189.73 | 2.58x |

## Environment Snapshot

| item | value |
|---|---|
| Platform | ModelScope DSW AMD GPU |
| Python | 3.12 |
| PyTorch | 2.10.0+git... |
| HIP | 7.2.53211 |
| vLLM | 0.20.1+rocm721 |
| GPU | 1 AMD GPU, about 192GB VRAM |
| Model | Qwen/Qwen3.6-27B-FP8 |

## Directory Layout

```text
scripts/    Reproducible launch and benchmark scripts
reports/    Round-by-round reports and research summary
publish/    Chinese article draft
notes/      Official references
figures/    Generated charts
```

## Reproduction Notes

Remote DSW paths used in the study:

```bash
export REPRO_DIR=/mnt/workspace/qwen27b_rocm_mtp_repro_20260525
export MODEL_CACHE=/home/qwen_model_cache
```

The model cache is separate from the reproducibility directory. The scripts and
reports are small and should be persisted; model weights can be downloaded again.

## Important Scripts

- `scripts/start_qwen36_27b_fp8_8k_baseline.sh`
- `scripts/start_qwen36_27b_fp8_8k_mtp.sh`
- `scripts/run_mtp_sweep.sh`
- `scripts/run_needle_gate.py`
- `scripts/run_round3_long_decode_and_kv.sh`
- `scripts/run_round4_forced_1024_decode.sh`
- `scripts/run_round5_concurrency_serving.sh`

## Reports

- `reports/round2_mtp_sweep_and_8k_gate_20260525.md`
- `reports/round3_long_decode_and_kv_20260525.md`
- `reports/round4_forced_1024_decode_20260525.md`
- `reports/round5_concurrency_serving_20260525.md`
- `reports/qwen36_27b_rocm_mtp_research_report_20260525.md`

## Scope Boundary

This is a reproducible research baseline and single-request / small-concurrency
serving study. It is not a production serving benchmark and should not be used
as a hardware ranking without matching hardware, concurrency, prompts, versions,
and measurement scripts.

## References

- Qwen3.6-27B-FP8: https://huggingface.co/Qwen/Qwen3.6-27B-FP8
- vLLM MTP documentation: https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/
- ROCm vLLM optimization: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
