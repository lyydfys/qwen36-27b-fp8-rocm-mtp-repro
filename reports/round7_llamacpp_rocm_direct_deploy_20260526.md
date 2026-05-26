# Round 7: llama.cpp ROCm Q4_K_M Direct Deployment Gate

Date: 2026-05-26

## Goal

Move the llama.cpp branch from "GGUF quantized files exist" to "the quantized
model can be deployed directly on AMD ROCm with a repeatable server path".

The target artifact is:

```text
/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525/models/qwen36-27b-fp8-q4_k_m.gguf
```

Remote size: about `16G`.

## Environment

- Platform: ModelScope DSW AMD GPU
- GPU runtime: ROCm, llama.cpp built with HIP support
- Observed device line: `ROCm0 : (196288 MiB, 195960 MiB free)`
- llama.cpp runtime path:
  `/mnt/workspace/toolchains/llama.cpp`
- Persistent repro path:
  `/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525`

## CLI Gate

The Q4_K_M GGUF model passed a direct `llama-cli` generation gate with ROCm:

```bash
./build-rocm/bin/llama-cli \
  -m /mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525/models/qwen36-27b-fp8-q4_k_m.gguf \
  -p '1+1=' \
  -n 8 \
  -ngl 99 \
  -fa off \
  --no-mmap \
  -c 512 \
  -t 16 \
  --temp 0 \
  --no-display-prompt
```

Observed result:

- Return code: `RC:0`
- Prompt throughput: about `73.7 tok/s`
- Generation throughput: about `33.4 tok/s`

This is the first gate proving that the quantized GGUF file is not only
convertible, but also executable on ROCm with llama.cpp.

## Server Gate

The OpenAI-compatible server was started with:

```bash
./build-rocm/bin/llama-server \
  -m /mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525/models/qwen36-27b-fp8-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 18080 \
  -ngl 99 \
  -fa off \
  --no-mmap \
  -c 2048 \
  -t 16
```

Final server report:

```text
/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525/reports/round7_direct_deploy_final_gate_20260526_113425.txt
```

Key evidence:

- `/health`: HTTP `200`
- `/v1/models`: HTTP `200`
- model metadata included `n_ctx=2048`, `n_ctx_train=262144`,
  `n_embd=5120`, `n_params=27320697856`, and size `16799719424`
- API stability: `API_CASES_OK=5/5`

API case summary:

| case | endpoint | status | elapsed | result |
|---|---|---:|---:|---|
| `chat_short_1` | `/v1/chat/completions` | 200 | 1.105s | pass |
| `chat_short_2` | `/v1/chat/completions` | 200 | 0.574s | pass |
| `chat_short_3` | `/v1/chat/completions` | 200 | 0.575s | pass |
| `chat_256_tokens` | `/v1/chat/completions` | 200 | 8.697s | pass |
| `native_completion` | `/completion` | 200 | 1.669s | pass |

This verifies both the OpenAI-compatible chat route and the native llama.cpp
completion route.

## Restart Gate

To avoid treating an already-running process as deployment success, a clean
restart gate was also run:

1. Kill the existing `llama-server` process.
2. Start the server again with the same Q4_K_M ROCm parameters.
3. Wait for `/health`.
4. Re-run `/v1/models`, one OpenAI-compatible chat request, and one native
   completion request.

Restart report:

```text
/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525/reports/round7_restart_gate_20260526_120835.txt
```

Observed restart evidence:

- `HEALTH_READY=1`
- `STARTUP_SECONDS=6`
- `/v1/models`: HTTP `200`
- `chat_after_restart`: HTTP `200`, elapsed `1.200s`
- `completion_after_restart`: HTTP `200`, elapsed `0.804s`
- `RESTART_API_OK=2/2`
- `DONE_ROUND7_RESTART_GATE`

## What This Means

The llama.cpp path is now a direct-deployment baseline for the Q4_K_M GGUF
artifact on AMD ROCm:

- the quantized GGUF file exists and is small enough for practical reuse;
- `llama-cli` can generate on ROCm;
- `llama-server` can expose an OpenAI-compatible local API;
- `/health`, `/v1/models`, chat completions, and native completions all pass;
- the server can be stopped, restarted, and verified again with the same
  deployment parameters.

This is stronger than a minimal conversion gate. It is a practical deployment
gate for local llama.cpp serving on AMD ROCm.

## Boundary

This round does not yet claim that Q4_K_M is the best quantization point or that
it preserves full model quality. It also does not test long-context llama.cpp
serving beyond the current `-c 2048` server gate.

The current result should be described as:

```text
Qwen3.6-27B-FP8 has a working Q4_K_M GGUF llama.cpp ROCm deployment baseline,
including CLI generation, OpenAI-compatible server verification, native
completion verification, and clean restart recovery.
```

Next research gates:

- compare Q4_K_M with Q5_K_M and Q8_0 quality/performance;
- add a small deterministic accuracy suite;
- test 4K/8K context under llama.cpp after the 2K server gate;
- package a publication-ready GGUF deployment note without including model
  weights in this repository.
