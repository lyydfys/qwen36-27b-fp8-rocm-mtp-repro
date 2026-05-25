#!/usr/bin/env bash
set -euo pipefail

REPRO_DIR=${REPRO_DIR:-/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525}
LLAMA_DIR=${LLAMA_DIR:-/mnt/workspace/toolchains/llama.cpp}
RUNTIME_DIR=${RUNTIME_DIR:-$REPRO_DIR/runtime_hf_symlink}

mkdir -p "$REPRO_DIR/models" "$REPRO_DIR/logs" "$REPRO_DIR/reports"

Q8_OUT=${Q8_OUT:-$REPRO_DIR/models/qwen36-27b-fp8-q8_0.gguf}
Q4_OUT=${Q4_OUT:-$REPRO_DIR/models/qwen36-27b-fp8-q4_k_m.gguf}

cd "$LLAMA_DIR"

python3 convert_hf_to_gguf.py "$RUNTIME_DIR" \
  --outtype q8_0 \
  --outfile "$REPRO_DIR/reports/dryrun_symlink_q8.gguf" \
  --dry-run \
  |& tee "$REPRO_DIR/logs/convert_dryrun_symlink_q8_$(date +%Y%m%d_%H%M%S).log"

start=$(date +%s)
python3 convert_hf_to_gguf.py "$RUNTIME_DIR" \
  --outtype q8_0 \
  --outfile "$Q8_OUT" \
  |& tee "$REPRO_DIR/logs/convert_q8_0_symlink_$(date +%Y%m%d_%H%M%S).log"
rc=${PIPESTATUS[0]}
end=$(date +%s)
{
  echo "rc=$rc"
  echo "elapsed_sec=$((end - start))"
  ls -lh "$Q8_OUT"
} | tee "$REPRO_DIR/reports/convert_q8_0_status.txt"

"$LLAMA_DIR/build-rocm/bin/llama-quantize" \
  --allow-requantize \
  "$Q8_OUT" \
  "$Q4_OUT" \
  Q4_K_M \
  > "$REPRO_DIR/logs/quantize_q4_k_m.log" 2>&1

{
  echo "rc=$?"
  ls -lh "$Q4_OUT"
} | tee "$REPRO_DIR/reports/quantize_q4_k_m_status.txt"
