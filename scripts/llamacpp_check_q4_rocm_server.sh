#!/usr/bin/env bash
set -euo pipefail

REPRO_DIR=${REPRO_DIR:-/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18080}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-900}

mkdir -p "$REPRO_DIR/logs" "$REPRO_DIR/reports"

base_url="http://$HOST:$PORT"
status="$REPRO_DIR/reports/llama_server_q4_rocm_check_status.txt"
models_json="$REPRO_DIR/reports/llama_server_q4_rocm_models.json"
completion_json="$REPRO_DIR/reports/llama_server_q4_rocm_completion.json"

echo "start=$(date '+%F %T')" > "$status"

deadline=$((SECONDS + TIMEOUT_SECONDS))
until curl -fsS "$base_url/health" >> "$status" 2>&1; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "health_timeout=1" >> "$status"
    exit 124
  fi
  sleep 5
done

curl -fsS "$base_url/v1/models" > "$models_json"
curl -fsS "$base_url/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Answer with one digit only: 2+3="}],"max_tokens":8,"temperature":0}' \
  > "$completion_json"

echo "models_json=$models_json" >> "$status"
echo "completion_json=$completion_json" >> "$status"
echo "end=$(date '+%F %T')" >> "$status"
