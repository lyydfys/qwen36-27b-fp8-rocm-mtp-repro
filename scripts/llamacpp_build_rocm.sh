#!/usr/bin/env bash
set -euo pipefail

REPRO_DIR=${REPRO_DIR:-/mnt/workspace/qwen36_27b_llamacpp_rocm_quant_20260525}
LLAMA_DIR=${LLAMA_DIR:-/mnt/workspace/toolchains/llama.cpp}
AMDGPU_TARGETS=${AMDGPU_TARGETS:-gfx942}

mkdir -p "$REPRO_DIR/logs" "$REPRO_DIR/reports" "$(dirname "$LLAMA_DIR")"

if [ ! -d "$LLAMA_DIR" ]; then
  echo "llama.cpp source directory not found: $LLAMA_DIR"
  echo "Download or extract ggml-org/llama.cpp into this path first."
  exit 2
fi

cd "$LLAMA_DIR"
cmake -S . -B build-rocm \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
  -DCMAKE_BUILD_TYPE=Release \
  |& tee "$REPRO_DIR/logs/cmake_llama_rocm_$(date +%Y%m%d_%H%M%S).log"

cmake --build build-rocm --config Release -j "$(nproc)" \
  |& tee "$REPRO_DIR/logs/build_llama_rocm_$(date +%Y%m%d_%H%M%S).log"

"$LLAMA_DIR/build-rocm/bin/llama-cli" --version \
  | tee "$REPRO_DIR/reports/llama_cpp_version.txt"
