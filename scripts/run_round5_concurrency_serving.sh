#!/usr/bin/env bash
set -uo pipefail

REPRO_DIR=${REPRO_DIR:-/mnt/workspace/qwen27b_rocm_mtp_repro_20260525}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.6-27B-FP8}
MODEL_CACHE=${MODEL_CACHE:-/home/qwen_model_cache}
METHOD=${METHOD:-qwen3_next_mtp}
CONCURRENCIES=${CONCURRENCIES:-"1 2 4"}
REQUEST_MULTIPLIER=${REQUEST_MULTIPLIER:-2}
MAX_TOKENS=${MAX_TOKENS:-1024}
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-true}

mkdir -p "$REPRO_DIR/logs" "$REPRO_DIR/reports"

TS=$(date +%Y%m%d_%H%M%S)
OUT="$REPRO_DIR/reports/round5_concurrency_serving_${TS}.jsonl"
SUMMARY="$REPRO_DIR/reports/round5_concurrency_serving_summary_${TS}.txt"
DRIVER_LOG="$REPRO_DIR/logs/round5_concurrency_serving_driver_${TS}.log"
PROMPT_FILE="$REPRO_DIR/reports/round5_forced_1024_prompt_${TS}.txt"

cat > "$PROMPT_FILE" <<'PROMPT'
You are running a deterministic concurrent serving benchmark.
Generate a long plain-text list of synthetic benchmark records.
Each record must use this exact format:
RECORD-000001 | amd-rocm-vllm-mtp-qwen36 | concurrency-check | token-fill
RECORD-000002 | amd-rocm-vllm-mtp-qwen36 | concurrency-check | token-fill
Continue increasing the record number by 1.
Do not summarize.
Do not explain.
Do not stop early.
Do not write a conclusion.
Keep generating records until the response is cut off by the max token limit.
Begin now:
PROMPT

echo "OUT=$OUT"
echo "SUMMARY=$SUMMARY"
echo "DRIVER_LOG=$DRIVER_LOG"
echo "PROMPT_FILE=$PROMPT_FILE"

stop_vllm() {
  pkill -f "vllm serve $MODEL_ID" 2>/dev/null || true
  sleep 8
}

wait_ready() {
  local model_name="$1"
  for i in $(seq 1 180); do
    if curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/tmp/qwen_round5_models.json 2>/dev/null; then
      echo "ready after ${i} checks for $model_name"
      cat /tmp/qwen_round5_models.json | head -c 500
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
  local log="$REPRO_DIR/logs/vllm_${served_name}_${TS}.log"
  stop_vllm
  echo "=== starting $served_name spec=$spec_tokens ==="
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
  if [[ "$spec_tokens" != "0" ]]; then
    cmd+=(--speculative-config "{\"method\":\"$METHOD\",\"num_speculative_tokens\":$spec_tokens}")
  fi
  nohup "${cmd[@]}" > "$log" 2>&1 &
  local pid=$!
  echo "PID=$pid LOG=$log"
  if ! wait_ready "$served_name"; then
    echo "{\"event\":\"start_failed\",\"model\":\"$served_name\",\"spec_tokens\":\"$spec_tokens\",\"log\":\"$log\"}" >> "$OUT"
    tail -n 160 "$log" || true
    return 1
  fi
  return 0
}

run_concurrency_set() {
  local served_name="$1"
  local label="$2"
  for c in $CONCURRENCIES; do
    local requests=$((c * REQUEST_MULTIPLIER))
    if [[ "$requests" -lt 2 ]]; then
      requests=2
    fi
    echo "=== bench label=$label concurrency=$c requests=$requests ==="
    python3 "$REPRO_DIR/scripts/run_concurrency_bench.py" \
      --model "$served_name" \
      --label "$label" \
      --output "$OUT" \
      --prompt-file "$PROMPT_FILE" \
      --concurrency "$c" \
      --requests "$requests" \
      --max-tokens "$MAX_TOKENS" \
      --timeout 1800
  done
}

summarize() {
  python3 - "$OUT" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
summaries = []
if out.exists():
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "summary" in obj:
            summaries.append(obj["summary"])

lines = []
lines.append(f"SOURCE {out}")
lines.append(
    "label\tconcurrency\trequests\tsuccess_rate\twall_time\taggregate_tok_s\tp50_latency\tp95_latency\tp50_decode_tok_s\tall_full"
)
for row in summaries:
    lines.append(
        "\t".join(
            [
                str(row.get("label")),
                str(row.get("concurrency")),
                str(row.get("requests")),
                str(row.get("success_rate")),
                str(row.get("wall_time_sec")),
                str(row.get("aggregate_completion_tokens_per_sec")),
                str(row.get("p50_latency_sec")),
                str(row.get("p95_latency_sec")),
                str(row.get("p50_decode_tokens_per_sec")),
                str(row.get("all_full_max_tokens")),
            ]
        )
    )

baseline = {r["concurrency"]: r for r in summaries if r.get("label") == "baseline"}
mtp = {r["concurrency"]: r for r in summaries if r.get("label") == "mtp_t4"}
lines.append("")
lines.append("speedup_vs_baseline_by_aggregate_tok_s")
lines.append("concurrency\tbaseline_tok_s\tmtp_t4_tok_s\tspeedup")
for c in sorted(set(baseline) & set(mtp)):
    b = baseline[c].get("aggregate_completion_tokens_per_sec")
    m = mtp[c].get("aggregate_completion_tokens_per_sec")
    speedup = (m / b) if b and m else None
    lines.append(
        f"{c}\t{b}\t{m}\t{speedup:.6f}" if speedup else f"{c}\t{b}\t{m}\tNA"
    )

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
}

{
  echo "=== round5 concurrency serving start $(date) ==="
  echo "CONCURRENCIES=$CONCURRENCIES REQUEST_MULTIPLIER=$REQUEST_MULTIPLIER MAX_TOKENS=$MAX_TOKENS"
  for spec in "baseline:0" "mtp_t4:4"; do
    label="${spec%%:*}"
    tokens="${spec##*:}"
    served="qwen3.6-27b-fp8-amd-8k-concurrency-${label}"
    if start_service "$served" "$tokens"; then
      run_concurrency_set "$served" "$label"
      summarize || true
    fi
  done
  summarize
  echo "=== round5 concurrency serving done $(date) ==="
  echo "OUT=$OUT"
  echo "SUMMARY=$SUMMARY"
} 2>&1 | tee -a "$DRIVER_LOG"

