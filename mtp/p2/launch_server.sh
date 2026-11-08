#!/usr/bin/env bash
# P2a: launch BF16 final-code release server on gpu-1-vm (modeled on swift-opt/run_config.sh, no benchmark).
# Run ON gpu-1-vm, detached. Writes server.pid (pid == pgid, via setsid).
set -u
OUT=/u701/j692wu/swift-opt/mtp/p2
LOG=$OUT/server/server.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/server/driver.log"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
[ "$(hostname)" = gpu-1-vm ] || { log "ABORT: not on gpu-1-vm"; exit 3; }
grep -q "session swift-opt" /u701/j692wu/qwen/rft/locks/gpu-1-vm.lock || { log "ABORT: lock missing"; exit 3; }
[ "$(gpu_used)" -lt 500 ] || { log "ABORT: GPU busy ($(gpu_used) MiB)"; exit 3; }
ss -ltn | grep -q ':8000 ' && { log "ABORT: port 8000 busy"; exit 3; }
[ "$(free -g | awk '/Mem:/{print $7}')" -gt 150 ] || { log "ABORT: host RAM low"; exit 3; }
unset SWIFT_FP8 SWIFT_DRAFT_FP8
export SWIFT_CHECKPOINT_PATH="$(cat /u701/j692wu/swift-local/model-path.txt)"
export PRESET=final-code CAPTURE_SIZES=1,2,3,4,5,10,15,20,25,30,35,40 MAX_SEQS=8
log "env: PRESET=$PRESET CAPTURE_SIZES=$CAPTURE_SIZES MAX_SEQS=$MAX_SEQS CKPT=$SWIFT_CHECKPOINT_PATH"
setsid nohup bash /u701/j692wu/swift-opt/release/serve_final.sh > "$LOG" 2>&1 < /dev/null &
SPID=$!
echo "$SPID" > "$OUT/server/server.pid"
log "launched server pid=$SPID pgid=$(ps -o pgid= -p $SPID | tr -d ' ')"
