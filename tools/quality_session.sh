#!/usr/bin/env bash
# Adapted from ~/swift-profile/quality/quality_session.sh (swift-opt quality run on gpu-1-vm).
# Differences from the original: lock is gpu-1-vm.lock; SWIFT_CHECKPOINT_PATH is NOT exported (NFS default from
# ~/swift-local/model-path.txt); outputs go to ~/swift-opt/quality/$LABEL and ~/swift-opt/quality/matrix.log;
# stress smoke block removed. Sampling steps (locked n=6, guard n=1, tool probe) are unchanged and still use the
# sampling configs in ~/swift-profile/quality (read-only).
#   quality_session.sh LABEL
P=/u701/j692wu/swift-profile; Q=$P/quality; RFT=/u701/j692wu/qwen/rft; PY=/u701/j692wu/.conda/envs/inference/bin/python
O=/u701/j692wu/swift-opt/quality; LOCK=$RFT/locks/gpu-1-vm.lock
LABEL=$1; D=$O/$LABEL; mkdir -p $D; SPID=""
log() { echo "$(date -u +%FT%TZ) [quality $LABEL] $*" | tee -a $D/session.log; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
avail_gb() { free -g | awk '/Mem:/{print $7}'; }
grep -q "session swift-opt quality" $LOCK 2>/dev/null || { log "ABORT: lock missing"; exit 3; }
[ "$(gpu_used)" -lt 500 ] || { log "ABORT: GPU busy"; exit 3; }
ss -ltn | grep -q ':8000 ' && { log "ABORT: port busy"; exit 3; }
[ "$(avail_gb)" -gt 150 ] || { log "ABORT: RAM low"; exit 3; }
env | grep -E '^(PRESET|SERVER_CMD|SWIFT_|PYTHONPATH|CAPTURE_SIZES|BACKEND|NSPEC|CUDA_ENV|MAX_SEQS)' > $D/env.txt
setsid nohup bash ${SERVER_CMD:-$P/serve_exp.sh} > $D/server.log 2>&1 < /dev/null &
SPID=$!; log "server pid $SPID"
t0=$(date +%s); state=waiting
while true; do
  kill -0 $SPID 2>/dev/null || { state=exited; break; }
  [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health 2>/dev/null)" = 200 ] && { state=ready; break; }
  [ $(( $(date +%s) - t0 )) -gt 2700 ] && { state=timeout; break; }
  sleep 10
done
log "startup $state after $(( $(date +%s) - t0 )) s"
if [ "$state" = ready ]; then
  grep -F '[swift' $D/server.log | cut -c1-300 > $D/patch-log.txt
  nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu,temperature.gpu,clocks_throttle_reasons.active --format=csv -lms 1000 > $D/telemetry.csv 2>/dev/null &
  TPID=$!
  cd $RFT
  curl -s -m 10 http://127.0.0.1:8000/metrics > $D/metrics-before.txt
  t1=$(date +%s)
  $PY sample.py --tasks taskpack/locked.jsonl --config $Q/sampling-locked.json --out $D/locked.jsonl --n 6 --models base > $D/locked.log 2>&1; log "locked sampling rc=$? in $(( $(date +%s) - t1 )) s: $(tail -1 $D/locked.log | cut -c1-200)"
  t2=$(date +%s)
  $PY sample.py --tasks taskpack/evalplus.jsonl --config $Q/sampling-guard.json --out $D/guard.jsonl --n 1 --models base > $D/guard.log 2>&1; log "guard sampling rc=$? in $(( $(date +%s) - t2 )) s: $(tail -1 $D/guard.log | cut -c1-200)"
  t3=$(date +%s)
  $PY tool_probe.py --base-url http://127.0.0.1:8000/v1 --models base=swift --n 4 --out $D/tool-probe.json > $D/tool-probe.log 2>&1; log "tool probe rc=$? in $(( $(date +%s) - t3 )) s: $(tail -2 $D/tool-probe.log | tr '\n' ' ' | cut -c1-240)"
  curl -s -m 10 http://127.0.0.1:8000/metrics > $D/metrics-after.txt
  kill $TPID 2>/dev/null
  log "tracebacks in server log: $(grep -c Traceback $D/server.log); ERROR lines: $(grep -c ' ERROR ' $D/server.log)"
fi
pgid=$(ps -o pgid= -p $SPID 2>/dev/null | tr -d ' ')
if [ -n "$pgid" ] && [ "$pgid" = "$SPID" ]; then kill -TERM -- -$pgid 2>/dev/null; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 3; done; kill -0 $SPID 2>/dev/null && kill -KILL -- -$pgid 2>/dev/null; fi
for i in $(seq 1 60); do [ "$(gpu_used)" -lt 500 ] && break; sleep 3; done
for i in $(seq 1 60); do [ "$(avail_gb)" -gt 150 ] && break; sleep 3; done
log "after stop: gpu_used=$(gpu_used) MiB host_avail=$(avail_gb) GiB"
[ "$(gpu_used)" -lt 500 ] && [ "$(avail_gb)" -gt 150 ] || { log "CLEANUP PROBLEM"; echo "$(date -u +%FT%TZ) HALT: quality $LABEL cleanup problem" >> $O/matrix.log; exit 3; }
echo "$(date -u +%FT%TZ) quality $LABEL finished" >> $O/matrix.log
