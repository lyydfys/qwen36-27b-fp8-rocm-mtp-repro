#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-qwen3.6-27b-fp8-amd-8k}

echo '=== /v1/models ==='
curl -sS --max-time 20 http://127.0.0.1:8000/v1/models \
  | python3 -m json.tool \
  | sed -n '1,80p'

echo '=== short chat ==='
curl -sS --max-time 180 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+3? Answer with one number.\"}],\"max_tokens\":16,\"temperature\":0}" \
  | python3 -m json.tool \
  | sed -n '1,120p'

