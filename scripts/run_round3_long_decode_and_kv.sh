#!/usr/bin/env bash
set -uo pipefail

REPRO_DIR=${REPRO_DIR:-/mnt/workspace/qwen27b_rocm_mtp_repro_20260525}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.6-27B-FP8}
MODEL_CACHE=${MODEL_CACHE:-/home/qwen_model_cache}
METHOD=${METHOD:-qwen3_next_mtp}
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-true}

mkdir -p "$REPRO_DIR/logs" "$REPRO_DIR/reports"

LONG_OUT="$REPRO_DIR/reports/long_decode_512_1024_$(date +%Y%m%d_%H%M%S).jsonl"
KV_OUT="$REPRO_DIR/reports/kv_fp8_experiment_$(date +%Y%m%d_%H%M%S).jsonl"
ROUND3_LOG="$REPRO_DIR/logs/round3_driver_$(date +%Y%m%d_%H%M%S).log"

echo "LONG_OUT=$LONG_OUT"
echo "KV_OUT=$KV_OUT"
echo "ROUND3_LOG=$ROUND3_LOG"

stop_vllm() {
  pkill -f "vllm serve $MODEL_ID" 2>/dev/null || true
  sleep 8
}

wait_ready() {
  local model_name="$1"
  for i in $(seq 1 150); do
    if curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/tmp/qwen_round3_models.json 2>/dev/null; then
      echo "ready after ${i} checks for $model_name"
      cat /tmp/qwen_round3_models.json | head -c 500
      echo
      return 0
    fi
    sleep 5
  done
  echo "ERROR: server did not become ready for $model_name" >&2
  return 1
}

start_service() {
  local served_name="$1"
  local spec_tokens="$2"
  local kv_dtype="$3"
  local log="$REPRO_DIR/logs/vllm_${served_name}_$(date +%Y%m%d_%H%M%S).log"
  stop_vllm
  echo "=== starting $served_name spec=$spec_tokens kv=$kv_dtype ==="
  local cmd=(
    vllm serve "$MODEL_ID"
    --served-model-name "$served_name"
    --host 0.0.0.0
    --port 8000
    --download-dir "$MODEL_CACHE"
    --max-model-len 8192
    --gpu-memory-utilization 0.90
    --max-num-seqs 4
    --max-num-batched-tokens 8192
    --language-model-only
    --reasoning-parser qwen3
    --enable-prefix-caching
    --disable-uvicorn-access-log
  )
  if [[ "$kv_dtype" != "none" ]]; then
    cmd+=(--kv-cache-dtype "$kv_dtype")
  fi
  if [[ "$spec_tokens" != "0" ]]; then
    cmd+=(--speculative-config "{\"method\":\"$METHOD\",\"num_speculative_tokens\":$spec_tokens}")
  fi
  nohup "${cmd[@]}" > "$log" 2>&1 &
  local pid=$!
  echo "PID=$pid LOG=$log"
  if ! wait_ready "$served_name"; then
    echo "{\"event\":\"start_failed\",\"model\":\"$served_name\",\"spec_tokens\":\"$spec_tokens\",\"kv_dtype\":\"$kv_dtype\",\"log\":\"$log\"}" >> "$KV_OUT"
    tail -n 120 "$log" || true
    return 1
  fi
  return 0
}

run_long_decode() {
  local served_name="$1"
  local label="$2"
  for max_tokens in 512 1024; do
    python3 "$REPRO_DIR/scripts/run_decode_bench.py" \
      --model "$served_name" \
      --label "${label}_${max_tokens}" \
      --max-tokens "$max_tokens" \
      --repeat 1 \
      --timeout 900 \
      --output "$LONG_OUT"
  done
}

run_kv_case() {
  local served_name="$1"
  local label="$2"
  python3 "$REPRO_DIR/scripts/run_decode_bench.py" \
    --model "$served_name" \
    --label "${label}_512" \
    --max-tokens 512 \
    --repeat 1 \
    --timeout 900 \
    --output "$KV_OUT"
  python3 "$REPRO_DIR/scripts/run_needle_gate.py" \
    --model "$served_name" \
    --output "$KV_OUT" \
    --max-tokens 192 \
    --cases "${label}_8k_near_safe:100:${label}-5082"
}

{
  echo "=== round3 start $(date) ==="
  echo "=== long decode experiment ==="
  for spec in "baseline:0" "mtp_t2:2" "mtp_t3:3" "mtp_t4:4"; do
    label="${spec%%:*}"
    tokens="${spec##*:}"
    served="qwen3.6-27b-fp8-amd-8k-long-${label}"
    if start_service "$served" "$tokens" "none"; then
      run_long_decode "$served" "$label"
    fi
  done

  echo "=== fp8 kv cache experiment ==="
  for spec in "kv_baseline:0" "kv_mtp_t4:4"; do
    label="${spec%%:*}"
    tokens="${spec##*:}"
    served="qwen3.6-27b-fp8-amd-8k-${label}"
    if start_service "$served" "$tokens" "fp8"; then
      run_kv_case "$served" "$label"
    fi
  done
  echo "=== round3 done $(date) ==="
  echo "LONG_OUT=$LONG_OUT"
  echo "KV_OUT=$KV_OUT"
} 2>&1 | tee -a "$ROUND3_LOG"
