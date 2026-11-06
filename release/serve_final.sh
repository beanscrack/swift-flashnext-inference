#!/usr/bin/env bash
# Launch Swift-1.5 on gpu-8-vm with one of the validated presets.
#   PRESET=baseline-wide   vLLM Marlin MTP3, only the CUDA-graph capture sizes widened (concurrency fix, no patches)
#   PRESET=final-code      MTP4 + 49,152-row code/English draft vocab + skinny GEMMs + L2 prefetch   (coding agents; fastest)
#   PRESET=final-multi     same with the 126,868-row multilingual draft vocab                       (recommended for mixed-language traffic)
set -euo pipefail
R="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PRESET="${PRESET:-final-multi}"
export CUDA_ENV="${CUDA_ENV:-cuda132}" BACKEND=marlin
case "$PRESET" in
  baseline-wide)
    export NSPEC=3 CAPTURE_SIZES=1,2,3,4,8,12,16 ;;
  final-code|final-multi)
    export NSPEC=4 CAPTURE_SIZES=1,2,3,4,5,10,15,20
    export PYTHONPATH="$R/hooks${PYTHONPATH:+:$PYTHONPATH}"
    if [ "$PRESET" = final-code ]; then export SWIFT_DRAFT_VOCAB_FILE="$R/vocab/swift_draft_vocab_code49152.npy"
    else export SWIFT_DRAFT_VOCAB_FILE="$R/vocab/swift_draft_vocab_multilingual126868.npy"; fi
    export SWIFT_SKINNY=1 SWIFT_SKINNY_PLAN="$R/plan/skinny_plan_wide.json" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_$USER}"
    export SWIFT_L2PF=1 SWIFT_L2PF_MODE=touch_el SWIFT_L2PF_CTAS=48 SWIFT_L2PF_BLOCK=1024 SWIFT_L2PF_BUDGET_MB=48 SWIFT_L2PF_MAX_TOKENS=24 ;;
  *) echo "unknown PRESET=$PRESET" >&2; exit 2 ;;
esac
echo "serve_final: PRESET=$PRESET NSPEC=$NSPEC CAPTURE_SIZES=$CAPTURE_SIZES" >&2
exec bash "$R/serve_exp.sh" "$@"
