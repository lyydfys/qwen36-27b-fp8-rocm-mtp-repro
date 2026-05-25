#!/usr/bin/env bash
set -euo pipefail

REPRO_DIR=${REPRO_DIR:-/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525}
LLAMA_DIR=${LLAMA_DIR:-/mnt/workspace/toolchains/llama.cpp}
MODEL_GGUF=${MODEL_GGUF:-$REPRO_DIR/models/qwen36-27b-fp8-q4_k_m.gguf}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18080}
CTX_SIZE=${CTX_SIZE:-2048}
THREADS=${THREADS:-16}
GPU_LAYERS=${GPU_LAYERS:-99}

mkdir -p "$REPRO_DIR/logs" "$REPRO_DIR/reports"

log="$REPRO_DIR/logs/llama_server_q4_rocm.log"
pid_file="$REPRO_DIR/reports/llama_server_q4_rocm.pid"
status="$REPRO_DIR/reports/llama_server_q4_rocm_status.txt"

cd "$LLAMA_DIR"
pkill -f "llama-server.*qwen36-27b-fp8-q4_k_m.gguf" 2>/dev/null || true

echo "start=$(date '+%F %T')" > "$status"
echo "host=$HOST" >> "$status"
echo "port=$PORT" >> "$status"
echo "model=$MODEL_GGUF" >> "$status"

nohup ./build-rocm/bin/llama-server \
  -m "$MODEL_GGUF" \
  --host "$HOST" \
  --port "$PORT" \
  -ngl "$GPU_LAYERS" \
  -fa off \
  --no-mmap \
  -c "$CTX_SIZE" \
  -t "$THREADS" \
  > "$log" 2>&1 &

echo $! > "$pid_file"
echo "pid=$(cat "$pid_file")" >> "$status"
echo "log=$log" >> "$status"
