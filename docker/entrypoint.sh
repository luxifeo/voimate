#!/usr/bin/env bash
# Submission entrypoint: start vLLM for Qwen3.5-4B, wait until it serves, run the pipeline on one split.
#   DATA_ROOT    folder holding the split folders (default /data)
#   SPLIT        split to predict (default private_test)
#   OUTPUT_PATH  submission file (default /output/submission.jsonl)
#   GPU_UTIL     vLLM --gpu-memory-utilization (default 0.85)
# Extra arguments are passed to scripts/predict.py (e.g. --limit 20).
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-/data}"
SPLIT="${SPLIT:-private_test}"
OUTPUT_PATH="${OUTPUT_PATH:-/output/submission.jsonl}"
PORT=8011

vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen3.5-4b \
  --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  --max-num-seqs 64 \
  --seed 0 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' > /tmp/vllm.log 2>&1 &
VLLM_PID=$!

until python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/v1/models', timeout=2)" 2>/dev/null; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM server exited during startup:" >&2
    tail -50 /tmp/vllm.log >&2
    exit 1
  fi
  sleep 5
done
echo "vLLM ready"

mkdir -p "$(dirname "$OUTPUT_PATH")"
python3 /app/scripts/predict.py \
  --root "$DATA_ROOT" \
  --split "$SPLIT" \
  --out "$OUTPUT_PATH" \
  --train-jsonl /app/assets/train.jsonl \
  --index-dir /app/assets/index \
  --base-url "http://localhost:${PORT}/v1" \
  "$@"

kill "$VLLM_PID"
