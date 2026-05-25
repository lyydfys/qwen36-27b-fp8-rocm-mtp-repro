#!/usr/bin/env bash
set -uo pipefail

REPRO_DIR=${REPRO_DIR:-/mnt/workspace/qwen27b_rocm_mtp_repro_20260525}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.6-27B-FP8}
MODEL_CACHE=${MODEL_CACHE:-/home/qwen_model_cache}
METHOD=${METHOD:-qwen3_next_mtp}
REPEAT=${REPEAT:-2}
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-true}

mkdir -p "$REPRO_DIR/logs" "$REPRO_DIR/reports"

TS=$(date +%Y%m%d_%H%M%S)
OUT="$REPRO_DIR/reports/forced_1024_decode_${TS}.jsonl"
SUMMARY="$REPRO_DIR/reports/forced_1024_decode_summary_${TS}.txt"
DRIVER_LOG="$REPRO_DIR/logs/round4_forced_1024_driver_${TS}.log"
PROMPT_FILE="$REPRO_DIR/reports/forced_1024_prompt_${TS}.txt"

cat > "$PROMPT_FILE" <<'PROMPT'
You are running a deterministic decode benchmark.
Generate a long plain-text list of synthetic benchmark records.
Each record must use this exact format:
RECORD-000001 | amd-rocm-vllm-mtp-qwen36 | latency-check | token-fill
RECORD-000002 | amd-rocm-vllm-mtp-qwen36 | latency-check | token-fill
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
    if curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/tmp/qwen_round4_models.json 2>/dev/null; then
      echo "ready after ${i} checks for $model_name"
      cat /tmp/qwen_round4_models.json | head -c 500
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

run_case() {
  local served_name="$1"
  local label="$2"
  python3 "$REPRO_DIR/scripts/run_decode_bench.py" \
    --model "$served_name" \
    --label "$label" \
    --max-tokens 1024 \
    --repeat "$REPEAT" \
    --timeout 1200 \
    --prompt "$(cat "$PROMPT_FILE")" \
    --output "$OUT"
}

summarize() {
  python3 - "$OUT" "$SUMMARY" <<'PY'
import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
rows = []
for line in out.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    if "summary" not in obj and "label" in obj:
        rows.append(obj)

labels = []
for r in rows:
    if r["label"] not in labels:
        labels.append(r["label"])

lines = []
lines.append(f"SOURCE {out}")
lines.append("label\trepeat\tmedian_latency\tmedian_tok_s\tmean_tok_s\tcompletion_tokens\tall_full_1024")
for label in labels:
    group = [r for r in rows if r["label"] == label]
    speeds = [r["decode_tokens_per_sec"] for r in group if isinstance(r.get("decode_tokens_per_sec"), (int, float))]
    latencies = [r["latency_sec"] for r in group if isinstance(r.get("latency_sec"), (int, float))]
    completion_tokens = [r.get("completion_tokens") for r in group]
    line = [
        label,
        str(len(group)),
        f"{statistics.median(latencies):.6f}" if latencies else "NA",
        f"{statistics.median(speeds):.6f}" if speeds else "NA",
        f"{statistics.mean(speeds):.6f}" if speeds else "NA",
        ",".join(str(x) for x in completion_tokens),
        str(all(x == 1024 for x in completion_tokens)),
    ]
    lines.append("\t".join(line))

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
}

{
  echo "=== round4 forced 1024 start $(date) ==="
  echo "REPEAT=$REPEAT"
  for spec in "baseline:0" "mtp_t2:2" "mtp_t3:3" "mtp_t4:4"; do
    label="${spec%%:*}"
    tokens="${spec##*:}"
    served="qwen3.6-27b-fp8-amd-8k-forced1024-${label}"
    if start_service "$served" "$tokens"; then
      run_case "$served" "$label"
      summarize || true
    fi
  done
  summarize
  echo "=== round4 forced 1024 done $(date) ==="
  echo "OUT=$OUT"
  echo "SUMMARY=$SUMMARY"
} 2>&1 | tee -a "$DRIVER_LOG"

