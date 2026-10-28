#!/usr/bin/env bash
# Start an LLM (default Gemma-4-E2B-it: text + audio, native tool calling) behind an OpenAI-compatible vLLM server.
# Usage: scripts/serve_vllm.sh [gpu_id] [port]
#   CUDA_COMPAT=1     prepend the image's CUDA forward-compat libs to LD_LIBRARY_PATH (host-only workaround for
#                     drivers older than the image's CUDA 13.0, e.g. driver 535); not part of the submission image.
#   MODEL, SERVED_NAME, TOOL_PARSER, REASONING_PARSER ('' = none), MM_LIMIT ('' = none), MAX_LEN, GPU_UTIL, EXTRA_ARGS
#   e.g. MODEL=Qwen/Qwen3.5-4B SERVED_NAME=qwen3.5-4b TOOL_PARSER=qwen3_coder REASONING_PARSER=qwen3 \
#        MM_LIMIT='{"image": 0, "video": 0}' CUDA_COMPAT=1 scripts/serve_vllm.sh 0 8011
set -euo pipefail
GPU="${1:-0}"
PORT="${2:-8011}"
IMAGE="${VLLM_IMAGE:-lifemate-vllm:0.32.0}"
NAME="${VLLM_NAME:-lifemate-vllm}"
MODEL="${MODEL:-google/gemma-4-E2B-it}"
SERVED_NAME="${SERVED_NAME:-gemma-4-e2b}"
TOOL_PARSER="${TOOL_PARSER:-gemma4}"
REASONING_PARSER="${REASONING_PARSER-gemma4}"
GEMMA_MM='{"image": 0, "audio": 1}'
MM_LIMIT="${MM_LIMIT-$GEMMA_MM}"

EXTRA_ENV=()
if [[ "${CUDA_COMPAT:-0}" == "1" ]]; then
  EXTRA_ENV+=(-e "LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib")
fi
ARGS=()
[[ -n "$REASONING_PARSER" ]] && ARGS+=(--reasoning-parser "$REASONING_PARSER")
[[ -n "$MM_LIMIT" ]] && ARGS+=(--limit-mm-per-prompt "$MM_LIMIT")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=($EXTRA_ARGS)

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "\"device=${GPU}\"" --ipc=host \
  -p "${PORT}:8000" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 \
  "${EXTRA_ENV[@]}" \
  "$IMAGE" \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len "${MAX_LEN:-16384}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.75}" \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_PARSER" \
  "${ARGS[@]}"
echo "container $NAME started on port $PORT (GPU $GPU): $MODEL as $SERVED_NAME"
