#!/usr/bin/env bash
set -euo pipefail

REPRO_DIR=${REPRO_DIR:-/mnt/workspace/qwen27b_rocm_mtp_repro_20260525}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.6-27B-FP8}
MODEL_CACHE=${MODEL_CACHE:-/home/qwen_model_cache}
METHOD=${METHOD:-qwen3_next_mtp}
TOKENS_LIST=${TOKENS_LIST:-"1 2 3 4"}
REPEAT=${REPEAT:-3}

mkdir -p "$REPRO_DIR/logs" "$REPRO_DIR/reports"
SUMMARY="$REPRO_DIR/reports/mtp_tokens_sweep_$(date +%Y%m%d_%H%M%S).jsonl"
echo "SUMMARY=$SUMMARY"

wait_ready() {
  local model_name="$1"
  for i in $(seq 1 120); do
    if curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/tmp/qwen_vllm_models.json 2>/dev/null; then
      echo "ready after ${i} checks for $model_name"
      cat /tmp/qwen_vllm_models.json | head -c 500
      echo
      return 0
    fi
    sleep 5
  done
  echo "server did not become ready for $model_name" >&2
  return 1
}

stop_vllm() {
  pkill -f "vllm serve $MODEL_ID" 2>/dev/null || true
  sleep 8
}

for spec_tokens in $TOKENS_LIST; do
  model_name="qwen3.6-27b-fp8-amd-8k-mtp-t${spec_tokens}"
  log="$REPRO_DIR/logs/vllm_qwen36_fp8_8k_mtp_t${spec_tokens}_$(date +%Y%m%d_%H%M%S).log"
  echo "=== starting $model_name ==="
  stop_vllm
  SPEC_TOKENS="$spec_tokens" \
    SPEC_METHOD="$METHOD" \
    SERVED_MODEL_NAME="$model_name" \
    MODEL_ID="$MODEL_ID" \
    MODEL_CACHE="$MODEL_CACHE" \
    REPRO_DIR="$REPRO_DIR" \
    nohup bash "$REPRO_DIR/scripts/start_qwen36_27b_fp8_8k_mtp.sh" > "$log" 2>&1 &
  pid=$!
  echo "PID=$pid LOG=$log"
  wait_ready "$model_name"
  python3 "$REPRO_DIR/scripts/run_decode_bench.py" \
    --model "$model_name" \
    --label "mtp_t${spec_tokens}" \
    --repeat "$REPEAT" \
    --output "$SUMMARY"
  echo "=== finished $model_name ==="
done

echo "=== sweep done ==="
cat "$SUMMARY"

