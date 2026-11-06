#!/usr/bin/env bash
set -euo pipefail
SWIFT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${SWIFT_MTP_MODEL_PATH:-}" && -s "$SWIFT_ROOT/model-path.txt" ]]; then
  SWIFT_MTP_MODEL_PATH="$(python3 "$SWIFT_ROOT/prepare_mtp_view.py")"
fi
SWIFT_MTP_CONFIG="$(python3 - "${SWIFT_MTP_MODEL_PATH:-}" <<'PY'
import json, sys
config = {"method": "mtp", "num_speculative_tokens": 3, "moe_backend": "triton"}
if sys.argv[1]:
    config["model"] = sys.argv[1]
    config["quantization"] = "modelopt_fp4"
print(json.dumps(config))
PY
)"
exec bash "$SWIFT_ROOT/serve.sh" \
  --speculative-config "$SWIFT_MTP_CONFIG" \
  "$@"
