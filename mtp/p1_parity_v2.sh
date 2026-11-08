#!/usr/bin/env bash
# P1 GPU parity (V2 runner, as in production): eager BF16 server with only the MTP capture hook in standalone mode,
# one greedy request recorded while $D/captures/ARM exists, then parity_v2.py (hidden-level + top-1 parity).
set -u
P=/u701/j692wu/swift-opt; M=$P/mtp; D=$P/runs/p1-parity-v2; mkdir -p "$D/captures"
PY=/u701/j692wu/.conda/envs/inference/bin/python
log() { echo "$(date -u +%FT%TZ) [p1-parity-v2] $*" | tee -a "$D/driver.log"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
[ "$(hostname)" = gpu-1-vm ] || { log "ABORT: not gpu-1-vm"; exit 3; }
grep -q "session swift-opt" /u701/j692wu/qwen/rft/locks/gpu-1-vm.lock || { log "ABORT: lock"; exit 3; }
[ "$(gpu_used)" -lt 500 ] || { log "ABORT: GPU busy"; exit 3; }
CK=$(cat /u701/j692wu/swift-local/model-path.txt)
rm -f "$D/captures/ARM"
export SWIFT_MTP_CAPTURE_DIR="$D/captures" SWIFT_MTP_CAPTURE_STANDALONE=1 SWIFT_MTP_CAPTURE_MAX_CALLS=400
export PYTHONPATH="$M" MAX_SEQS=1 NSPEC=4 CUDA_ENV=cuda132 BACKEND=marlin
unset VLLM_USE_V2_MODEL_RUNNER
setsid nohup bash $P/release/serve_exp.sh --enforce-eager > "$D/server.log" 2>&1 < /dev/null &
SPID=$!; log "server pid $SPID"
t0=$(date +%s); state=waiting
while true; do
  kill -0 $SPID 2>/dev/null || { state=exited; break; }
  [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)" = 200 ] && { state=ready; break; }
  [ $(( $(date +%s) - t0 )) -gt 2700 ] && { state=timeout; break; }
  sleep 10
done
log "startup $state after $(( $(date +%s) - t0 )) s; capture hook lines: $(grep -c swift-mtp-capture "$D/server.log")"
if [ "$state" = ready ]; then
  touch "$D/captures/ARM"
  curl -s -m 900 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
    "model":"swift","temperature":0,"max_tokens":600,
    "chat_template_kwargs":{"enable_thinking":false},
    "messages":[{"role":"user","content":"Write a Python function that parses an INI-style config file into nested dicts, handling comments, quoted values and duplicate sections. Include type hints and a short docstring."}]}' > "$D/response.json"
  rm -f "$D/captures/ARM"
  log "calls captured: $(ls "$D/captures" | grep -c call_); response bytes $(wc -c < "$D/response.json"); usage $(grep -o '"usage":{[^}]*}' "$D/response.json")"
fi
pgid=$(ps -o pgid= -p $SPID | tr -d ' '); [ "$pgid" = "$SPID" ] && kill -TERM -- -$pgid; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 3; done
kill -0 $SPID 2>/dev/null && kill -KILL -- -$pgid; for i in $(seq 1 60); do [ "$(gpu_used)" -lt 500 ] && break; sleep 3; done
log "server stopped; gpu_used=$(gpu_used)"
if [ "$state" = ready ]; then
  cd $M && $PY parity_v2.py --captures "$D/captures" --config "$CK/config.json" \
    --weights $M/view-identity/model-tail.safetensors --device cuda --out "$D/parity.json" > "$D/parity.log" 2>&1
  log "parity rc=$?: $(tail -14 "$D/parity.log" | tr '\n' ' ' | cut -c1-900)"
fi
log "done"
