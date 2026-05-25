#!/usr/bin/env bash
set -euo pipefail

REPRO_DIR=${REPRO_DIR:-/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525}
LLAMA_DIR=${LLAMA_DIR:-/mnt/workspace/toolchains/llama.cpp}
MODEL_GGUF=${MODEL_GGUF:-$REPRO_DIR/models/qwen36-27b-fp8-q4_k_m.gguf}
PROMPT=${PROMPT:-"Q: 2+3? A:"}
N_PREDICT=${N_PREDICT:-64}
CTX_SIZE=${CTX_SIZE:-512}
THREADS=${THREADS:-16}
GPU_LAYERS=${GPU_LAYERS:-99}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-900}

mkdir -p "$REPRO_DIR/logs" "$REPRO_DIR/reports"

log="$REPRO_DIR/logs/direct_q4_cli.log"
status="$REPRO_DIR/reports/direct_q4_cli_status.txt"

cd "$LLAMA_DIR"
echo "start=$(date '+%F %T')" > "$status"
echo "model=$MODEL_GGUF" >> "$status"
echo "prompt=$PROMPT" >> "$status"

set +e
timeout "$TIMEOUT_SECONDS" ./build-rocm/bin/llama-cli \
  -m "$MODEL_GGUF" \
  -p "$PROMPT" \
  -n "$N_PREDICT" \
  -ngl "$GPU_LAYERS" \
  -fa off \
  --no-mmap \
  -c "$CTX_SIZE" \
  -t "$THREADS" \
  --temp 0 \
  --no-display-prompt \
  --single-turn \
  --no-cnv \
  > "$log" 2>&1
rc=$?
set -e

echo "rc=$rc" >> "$status"
echo "end=$(date '+%F %T')" >> "$status"
tail -n 80 "$log" >> "$status"
exit "$rc"
