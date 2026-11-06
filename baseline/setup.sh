#!/usr/bin/env bash
set -euo pipefail
SWIFT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SWIFT_BOOTSTRAP_PYTHON="${SWIFT_BOOTSTRAP_PYTHON:-python3}"
"$SWIFT_BOOTSTRAP_PYTHON" -m venv "$SWIFT_ROOT/.venv"
"$SWIFT_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$SWIFT_ROOT/.venv/bin/python" -m pip install \
  'vllm==0.31.0' 'transformers==5.17.0' 'huggingface-hub==1.33.0'
"$SWIFT_ROOT/.venv/bin/python" "$SWIFT_ROOT/download.py"
printf 'Start the server with: bash %s/serve.sh\n' "$SWIFT_ROOT"
