#!/usr/bin/env bash
# Finish the P3 online gate: wait for the already-launched BF16 server (PGID 67535), run online_gate.py, stop that server.
set -u
P3=/u701/j692wu/swift-opt/mtp/p3; L=$P3/logs/gate_chain.log; SPID=67535
log() { echo "$(date -u +%FT%TZ) [gate-chain] $*" | tee -a "$L"; }
cmd=$(ps -o cmd= -p $SPID 2>/dev/null)
case "$cmd" in *vllm.entrypoints.cli.main\ serve*) log "server pid $SPID verified";; *) log "ABORT: pid $SPID is not the vLLM server: $cmd"; exit 3;; esac
t0=$(date +%s)
until [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)" = 200 ]; do
  kill -0 $SPID 2>/dev/null || { log "ABORT: server exited during start-up"; exit 3; }
  [ $(( $(date +%s) - t0 )) -gt 2700 ] && { log "ABORT: start-up timeout"; break; }
  sleep 10
done
log "server ready after $(( $(date +%s) - t0 )) s; hooks: $(grep -oE '\[swift[^]]*\] (ACTIVE|patched|armed)' $P3/logs/*server*.log 2>/dev/null | sort -u | tr '\n' ' ')"
cd $P3 && /u701/j692wu/.conda/envs/inference/bin/python online_gate.py > $P3/logs/online_gate.log 2>&1
log "online_gate rc=$?: $(tail -6 $P3/logs/online_gate.log | tr '\n' ' ' | cut -c1-700)"
cmd=$(ps -o cmd= -p $SPID 2>/dev/null)
case "$cmd" in *vllm.entrypoints.cli.main\ serve*)
  kill -TERM -- -$SPID; for i in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 3; done
  kill -0 $SPID 2>/dev/null && kill -KILL -- -$SPID;;
esac
for i in $(seq 1 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 500 ] && break; sleep 3; done
log "server stopped; gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB; port8000=$(ss -ltn | grep -c ':8000 ')"
log "done"
