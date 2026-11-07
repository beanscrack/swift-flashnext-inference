# Sourced by serve_final.sh and ../run_config.sh: exports the env for PRESET (baseline-wide | final-code | final-multi).
R="${R:-/u701/j692wu/swift-opt/release}"
PRESET="${PRESET:-final-multi}"
export PRESET CUDA_ENV="${CUDA_ENV:-cuda132}" BACKEND="${BACKEND:-marlin}"
case "$PRESET" in
  baseline-wide)
    export NSPEC="${NSPEC:-3}" CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,3,4,8,12,16}" ;;
  final-code|final-multi)
    export NSPEC="${NSPEC:-4}" CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,3,4,5,10,15,20}"
    export PYTHONPATH="$R/hooks${PYTHONPATH:+:$PYTHONPATH}"
    if [ "$PRESET" = final-code ]; then export SWIFT_DRAFT_VOCAB_FILE="${SWIFT_DRAFT_VOCAB_FILE:-$R/vocab/swift_draft_vocab_code49152.npy}"
    else export SWIFT_DRAFT_VOCAB_FILE="${SWIFT_DRAFT_VOCAB_FILE:-$R/vocab/swift_draft_vocab_multilingual126868.npy}"; fi
    export SWIFT_SKINNY="${SWIFT_SKINNY:-1}" SWIFT_SKINNY_PLAN="${SWIFT_SKINNY_PLAN:-$R/plan/skinny_plan_wide.json}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_$USER}"
    export SWIFT_L2PF="${SWIFT_L2PF:-1}" SWIFT_L2PF_MODE="${SWIFT_L2PF_MODE:-touch_el}" SWIFT_L2PF_CTAS="${SWIFT_L2PF_CTAS:-48}" SWIFT_L2PF_BLOCK="${SWIFT_L2PF_BLOCK:-1024}" SWIFT_L2PF_BUDGET_MB="${SWIFT_L2PF_BUDGET_MB:-48}" SWIFT_L2PF_MAX_TOKENS="${SWIFT_L2PF_MAX_TOKENS:-24}" ;;
  *) echo "unknown PRESET=$PRESET" >&2; return 2 2>/dev/null || exit 2 ;;
esac
