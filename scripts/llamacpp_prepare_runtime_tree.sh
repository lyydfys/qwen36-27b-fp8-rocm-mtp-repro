#!/usr/bin/env bash
set -euo pipefail

REPRO_DIR=${REPRO_DIR:-/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525}
MODEL_DIR=${MODEL_DIR:-/mnt/workspace/.cache/modelscope/models/Qwen/Qwen3.6-27B-FP8}
RUNTIME_DIR=${RUNTIME_DIR:-$REPRO_DIR/runtime_hf_symlink}

MODEL_REAL=$(realpath "$MODEL_DIR")
rm -rf "$RUNTIME_DIR"
mkdir -p "$RUNTIME_DIR"

cp "$MODEL_REAL"/*.json "$RUNTIME_DIR"/ 2>/dev/null || true
cp "$MODEL_REAL"/*.jinja "$RUNTIME_DIR"/ 2>/dev/null || true
cp "$MODEL_REAL"/tokenizer* "$RUNTIME_DIR"/ 2>/dev/null || true

# Keep the original layer names because the ModelScope weight index references
# layers-*.safetensors and mtp.safetensors.
for f in "$MODEL_REAL"/*.safetensors; do
  ln -sf "$f" "$RUNTIME_DIR/$(basename "$f")"
done

# Also expose Hugging Face style model shard names for tools that look for
# model-00001-of-N.safetensors patterns.
total=$(find "$MODEL_REAL" -maxdepth 1 -name 'layers-*.safetensors' | wc -l)
n=0
for f in $(find "$MODEL_REAL" -maxdepth 1 -name 'layers-*.safetensors' | sort -V); do
  n=$((n + 1))
  ln -sf "$f" "$RUNTIME_DIR/model-$(printf '%05d' "$n")-of-$(printf '%05d' "$total").safetensors"
done

echo "runtime_dir=$RUNTIME_DIR"
echo "model_real=$MODEL_REAL"
echo "safetensors=$(find "$RUNTIME_DIR" -maxdepth 1 -name '*.safetensors' | wc -l)"
