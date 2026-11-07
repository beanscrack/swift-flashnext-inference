#!/usr/bin/env bash
# Launch Swift-1.5 with a validated preset (see preset_env.sh). Extra SWIFT_* env vars pass through to the hooks.
set -euo pipefail
R="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$R/preset_env.sh"
echo "serve_final: PRESET=$PRESET NSPEC=$NSPEC CAPTURE_SIZES=$CAPTURE_SIZES" >&2
exec bash "$R/serve_exp.sh" "$@"
