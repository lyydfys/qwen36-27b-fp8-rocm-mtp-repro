#!/usr/bin/env bash
set -euo pipefail

REPRO_DIR=${REPRO_DIR:-/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525}
LLAMA_DIR=${LLAMA_DIR:-/mnt/workspace/toolchains/llama.cpp}
MODEL_GGUF=${MODEL_GGUF:-$REPRO_DIR/models/qwen36-27b-fp8-q4_k_m.gguf}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-180}

mkdir -p "$REPRO_DIR/logs" "$REPRO_DIR/reports"

log="$REPRO_DIR/logs/llama_min_gate.log"
status="$REPRO_DIR/reports/llama_min_gate_status.txt"

cd "$LLAMA_DIR"
echo "start=$(date '+%F %T')" > "$status"
timeout "$TIMEOUT_SECONDS" ./build-rocm/bin/llama-cli \
  -m "$MODEL_GGUF" \
  -p "2+3=" \
  -n 4 \
  -ngl 99 \
  -fa off \
  -c 512 \
  -t 16 \
  --temp 0 \
  --no-display-prompt \
  > "$log" 2>&1
rc=$?
echo "rc=$rc" >> "$status"
echo "end=$(date '+%F %T')" >> "$status"
