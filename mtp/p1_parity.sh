#!/usr/bin/env bash
# P1 GPU parity: eager BF16 server with only the MTP capture hook, one greedy request, then parity_check.py.
set -u
P=/u701/j692wu/swift-opt; M=$P/mtp; D=$P/runs/p1-parity; mkdir -p "$D/captures"
PY=/u701/j692wu/.conda/envs/inference/bin/python
log() { echo "$(date -u +%FT%TZ) [p1-parity] $*" | tee -a "$D/driver.log"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
[ "$(hostname)" = gpu-1-vm ] || { log "ABORT: not gpu-1-vm"; exit 3; }
grep -q "session swift-opt" /u701/j692wu/qwen/rft/locks/gpu-1-vm.lock || { log "ABORT: lock"; exit 3; }
[ "$(gpu_used)" -lt 500 ] || { log "ABORT: GPU busy"; exit 3; }
CK=$(cat /u701/j692wu/swift-local/model-path.txt)
export SWIFT_MTP_CAPTURE_DIR="$D/captures" SWIFT_MTP_CAPTURE_SAMPLE_ROUNDS=4 SWIFT_MTP_CAPTURE_MAX_MB=2000
export VLLM_USE_V2_MODEL_RUNNER=0 PYTHONPATH="$M" MAX_SEQS=1 NSPEC=4 CUDA_ENV=cuda132 BACKEND=marlin
setsid nohup bash $P/release/serve_exp.sh --enforce-eager > "$D/server.log" 2>&1 < /dev/null &
SPID=$!; log "server pid $SPID"
t0=$(date +%s); state=waiting
while true; do
  kill -0 $SPID 2>/dev/null || { state=exited; break; }
  [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)" = 200 ] && { state=ready; break; }
  [ $(( $(date +%s) - t0 )) -gt 2700 ] && { state=timeout; break; }
  sleep 10
done
log "startup $state after $(( $(date +%s) - t0 )) s; capture hook lines: $(grep -c 'mtp_capture\|swift_mtp' "$D/server.log")"
if [ "$state" = ready ]; then
  # one greedy coding request, single client, temperature 0, total length well under 2048
  curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
    "model":"swift","temperature":0,"max_tokens":600,
    "chat_template_kwargs":{"enable_thinking":false},
    "messages":[{"role":"user","content":"Write a Python function that parses an INI-style config file into nested dicts, handling comments, quoted values and duplicate sections. Include type hints and a short docstring."}]}' > "$D/response.json"
  log "response bytes $(wc -c < "$D/response.json"); usage $(grep -o '"usage":{[^}]*}' "$D/response.json")"
  ls -la "$D/captures" | tail -5 >> "$D/driver.log"
fi
pgid=$(ps -o pgid= -p $SPID | tr -d ' '); [ "$pgid" = "$SPID" ] && kill -TERM -- -$pgid; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 3; done
kill -0 $SPID 2>/dev/null && kill -KILL -- -$pgid; for i in $(seq 1 60); do [ "$(gpu_used)" -lt 500 ] && break; sleep 3; done
log "server stopped; gpu_used=$(gpu_used)"
if [ "$state" = ready ]; then
  cd $M && $PY parity_check.py --captures "$D/captures" --config "$CK/config.json" \
    --weights $M/view-identity/model-tail.safetensors --vocab full --lm-head-source bf16 --device cuda \
    --out "$D/parity.json" > "$D/parity.log" 2>&1
  log "parity rc=$?: $(tail -5 "$D/parity.log" | tr '\n' ' ' | cut -c1-600)"
fi
log "done"
