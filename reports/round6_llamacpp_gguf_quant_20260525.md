# Round 6: llama.cpp ROCm GGUF Quantization Probe

Date: 2026-05-25

## Goal

Build a deployable llama.cpp route for `Qwen/Qwen3.6-27B-FP8` on AMD ROCm.
The deployable artifact format is GGUF, because llama.cpp runs GGUF directly.
Quark, AWQ, and GPTQ remain useful comparison branches, but they are not the
first runtime target for llama.cpp.

## Remote Environment

- Platform: ModelScope DSW AMD GPU
- GPU target: `gfx942`
- ROCm/HIP: `7.2.53211`
- VRAM: about 192GB
- Persistent repro directory:
  `/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525`
- Model cache:
  `/mnt/workspace/.cache/modelscope/models/Qwen/Qwen3.6-27B-FP8`

Large model files were kept on the remote instance. This repository stores only
scripts and summarized evidence.

## What Worked

1. Built upstream `ggml-org/llama.cpp` from a zip snapshot with ROCm enabled:
   `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx942`.
2. Confirmed ModelScope cached model config:
   `model_type=qwen3_5`, architecture `Qwen3_5ForConditionalGeneration`,
   quantization method `fp8`, format `e4m3`.
3. Found a conversion pitfall: the ModelScope visible path is a symlink
   `Qwen3.6-27B-FP8 -> Qwen3___6-27B-FP8`.
4. Found a second conversion pitfall: the weight index references
   `layers-*.safetensors` and `mtp.safetensors`; a runtime tree must preserve
   those original names. HF-style `model-xxxxx-of-N.safetensors` links alone are
   not enough.
5. After fixing the runtime symlink tree, llama.cpp dry-run recognized
   `866` tensors and estimated Q8_0 output size around `29.0G`.
6. Generated a real Q8_0 GGUF:
   `/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525/models/qwen36-27b-fp8-q8_0.gguf`
   with size about `28G`; conversion status `rc=0`, elapsed about `423s`.
7. Generated a real Q4_K_M GGUF from Q8_0 with `--allow-requantize`:
   `/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525/models/qwen36-27b-fp8-q4_k_m.gguf`
   with size about `16G`; quantization status `rc=0`.

## Current Runtime Blocker

The GGUF files are generated, but the minimum llama.cpp inference gate has not
passed yet.

Observed behavior:

- Q8_0 `llama-cli` test with `-fa on` did not finish in the observation window.
- Q4_K_M `llama-cli` test with `-fa off`, `-n 4`, `-c 512`, `-t 16` also did not
  finish in the observation window.
- During these attempts, VRAM was allocated, but `rocm-smi` showed GPU use near
  zero. Logs remained empty until process exit, so the current evidence points
  to a load/init or early execution stall rather than a confirmed generation
  result.
- The DSW browser token expired before the final load-only status could be read.

This means the work has reached "real GGUF quantized artifacts generated" but
has not yet reached "llama.cpp short generation validated".

## Reproduction Scripts Added

- `scripts/llamacpp_build_rocm.sh`
- `scripts/llamacpp_prepare_runtime_tree.sh`
- `scripts/llamacpp_convert_gguf_quant.sh`
- `scripts/llamacpp_run_min_gate.sh`

## Next Step

After refreshing the DSW login, resume from the existing persistent directory
and run:

```bash
export REPRO_DIR=/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525
bash "$REPRO_DIR/scripts/run_llama_q4_load_only.sh"
cat "$REPRO_DIR/reports/llama_q4_load_only_status.txt"
tail -n 120 "$REPRO_DIR/logs/llama_q4_load_only.log"
```

If load-only times out, rerun with:

```bash
MODEL_GGUF="$REPRO_DIR/models/qwen36-27b-fp8-q4_k_m.gguf" \
TIMEOUT_SECONDS=180 \
bash scripts/llamacpp_run_min_gate.sh
```

Then compare:

- `-ngl 0` CPU-only load-only gate
- `-ngl 99 -fa off`
- `-ngl 99 -fa auto`
- `--no-mmap` versus mmap default
- current upstream llama.cpp versus AMD ROCm fork if needed
