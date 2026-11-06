#!/usr/bin/env bash
set -euo pipefail
SWIFT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${SWIFT_PYTHON:-}" ]]; then
  if [[ -x "$SWIFT_ROOT/.venv/bin/python" ]]; then
    SWIFT_PYTHON="$SWIFT_ROOT/.venv/bin/python"
  else
    SWIFT_PYTHON="/u701/j692wu/.conda/envs/inference/bin/python"
  fi
fi
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"
if [[ ! -s "$SWIFT_ROOT/model-path.txt" ]]; then
  "$SWIFT_PYTHON" "$SWIFT_ROOT/download.py"
fi
SWIFT_MODEL_PATH="${SWIFT_CHECKPOINT_PATH:-$(cat "$SWIFT_ROOT/model-path.txt")}"
printf '%s\n' "$$" > "$SWIFT_ROOT/server.pid"
exec "$SWIFT_PYTHON" -m vllm.entrypoints.cli.main serve "$SWIFT_MODEL_PATH" \
  --served-model-name swift \
  --host 127.0.0.1 --port "${SWIFT_PORT:-8000}" \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --language-model-only \
  --engram-config '{"cpu_offload":true}' \
  --moe-backend marlin \
  --max-model-len "${SWIFT_CONTEXT:-131072}" \
  --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.95 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --generation-config vllm \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' \
  "$@"
