#!/usr/bin/env bash
# Parameterized copy of swift-local/serve.sh (same flags) for backend / MTP-length experiments.
# Env: BACKEND (marlin|flashinfer_cutlass|cutlass|flashinfer_b12x|b12x|auto), NSPEC (draft tokens),
#      CUDA_ENV (orig|cuda132), SWIFT_CHECKPOINT_PATH (optional local view), SWIFT_PORT.
set -euo pipefail
SWIFT_ROOT=/u701/j692wu/swift-local
PY=/u701/j692wu/.conda/envs/inference/bin/python
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"
if [[ "${CUDA_ENV:-orig}" == "cuda132" ]]; then
  export CUDA_HOME=/usr/local/cuda
  export PATH=/usr/local/cuda/bin:$PATH
fi
MODEL="${SWIFT_CHECKPOINT_PATH:-$(cat "$SWIFT_ROOT/model-path.txt")}"
BACKEND="${BACKEND:-marlin}"
NSPEC="${NSPEC:-3}"
SPEC="$(printf '{"method":"mtp","num_speculative_tokens":%s,"moe_backend":"triton"}' "$NSPEC")"
MOE_ARGS=()
if [[ "$BACKEND" != "auto" ]]; then MOE_ARGS=(--moe-backend "$BACKEND"); fi
echo "serve_exp: BACKEND=$BACKEND NSPEC=$NSPEC CUDA_ENV=${CUDA_ENV:-orig} MODEL=$MODEL CUDA_HOME=${CUDA_HOME:-unset}" >&2
exec "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --served-model-name swift \
  --host 127.0.0.1 --port "${SWIFT_PORT:-8000}" \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --language-model-only \
  --engram-config '{"cpu_offload":true}' \
  "${MOE_ARGS[@]}" \
  --max-model-len "${SWIFT_CONTEXT:-131072}" \
  --max-num-seqs "${MAX_SEQS:-4}" --max-num-batched-tokens 2048 \
  --gpu-memory-utilization "${GPU_UTIL:-0.95}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --generation-config vllm \
  --compilation-config "$(printf '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[%s]}' "${CAPTURE_SIZES:-1,2,4}")" \
  --speculative-config "$SPEC" \
  "$@"
