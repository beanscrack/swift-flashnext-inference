#!/usr/bin/env bash
set -euo pipefail
SWIFT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SWIFT_BENCH_PYTHON="$SWIFT_ROOT/.bench-venv/bin/python"
SWIFT_MODEL_PATH="$(cat "$SWIFT_ROOT/model-path.txt")"
SWIFT_DATASET_PATH="$(cat "$SWIFT_ROOT/benchmark-dataset-path.txt")"
SWIFT_RESULTS="${SWIFT_BENCH_RESULTS:-$SWIFT_ROOT/context-benchmarks/$(date -u +%Y%m%dT%H%M%SZ)}"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$SWIFT_RESULTS"
printf '%s\n' "$SWIFT_RESULTS" > "$SWIFT_ROOT/context-benchmark-path.txt"
printf 'Results directory: %s\n' "$SWIFT_RESULTS"
for INPUT in 512 4096 16384 32768 65536; do
  printf 'Starting input=%s output=1024 prompts=5 concurrency=1 at %s\n' "$INPUT" "$(date -u +%FT%TZ)"
  "$SWIFT_BENCH_PYTHON" -m sglang.bench_serving \
    --backend vllm \
    --host 127.0.0.1 --port 8000 \
    --model "$SWIFT_MODEL_PATH" --served-model-name swift \
    --tokenizer "$SWIFT_MODEL_PATH" \
    --dataset-name random --dataset-path "$SWIFT_DATASET_PATH" \
    --random-input-len "$INPUT" \
    --random-output-len 1024 \
    --random-range-ratio 1.0 \
    --max-concurrency 1 \
    --num-prompts 5 \
    --flush-cache \
    --output-details \
    --output-file "$SWIFT_RESULTS/context_${INPUT}.jsonl" \
    > "$SWIFT_RESULTS/context_${INPUT}.log" 2>&1
  /usr/bin/python3 - "$SWIFT_RESULTS/context_${INPUT}.jsonl" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text().splitlines()[-1])
if data['completed'] != 5 or data['total_output_tokens'] != 5120:
    raise RuntimeError(f"Unexpected completed/output token counts: {data['completed']}/{data['total_output_tokens']}")
print(json.dumps({key: data.get(key) for key in ('random_input_len', 'completed', 'total_output_tokens', 'mean_ttft_ms', 'mean_tpot_ms', 'output_throughput')}), flush=True)
PY
done
printf 'All five context benchmarks completed.\n'
/usr/bin/python3 "$SWIFT_ROOT/summarize_context_benchmarks.py"
