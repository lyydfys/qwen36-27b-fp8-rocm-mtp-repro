#!/usr/bin/env bash
set -euo pipefail

export MODEL_ID=${MODEL_ID:-Qwen/Qwen3.6-27B-FP8}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.6-27b-fp8-amd-8k-mtp}
export MODEL_CACHE=${MODEL_CACHE:-/home/qwen_model_cache}
export REPRO_DIR=${REPRO_DIR:-/mnt/workspace/qwen27b_rocm_mtp_repro_20260525}
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-true}
export SPEC_METHOD=${SPEC_METHOD:-qwen3_next_mtp}
export SPEC_TOKENS=${SPEC_TOKENS:-2}

mkdir -p "$MODEL_CACHE" "$REPRO_DIR/logs" "$REPRO_DIR/reports"
cd /mnt/workspace

exec vllm serve "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --download-dir "$MODEL_CACHE" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --speculative-config "{\"method\":\"$SPEC_METHOD\",\"num_speculative_tokens\":$SPEC_TOKENS}" \
  --disable-uvicorn-access-log

