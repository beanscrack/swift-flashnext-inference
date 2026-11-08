#!/usr/bin/env bash
# Run ONE configuration end to end on gpu-1-vm: start server, verify hooks, benchmark N reps, stop, verify cleanup.
# Adapted from ~/swift-profile/run_config7.sh (gpu-1-vm lock, NFS checkpoint, launches release/serve_final.sh).
#   PRESET=final-code [extra SWIFT_* env] run_config.sh LABEL [REPS]
# Exit codes: 0 done (see status.json), 3 preflight/cleanup problem (GPU not clean) -> caller must stop.
set -u
P=/u701/j692wu/swift-opt
PP=/u701/j692wu/swift-profile
SW=/u701/j692wu/swift-local
LABEL=$1; REPS=${2:-5}
D=$P/runs/$LABEL; mkdir -p "$D"
log() { echo "$(date -u +%FT%TZ) [$LABEL] $*" | tee -a "$D/driver.log"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
avail_gb() { free -g | awk '/Mem:/{print $7}'; }
status() { printf '{"label":"%s","preset":"%s","status":"%s","valid":%s,"detail":"%s"}\n' "$LABEL" "${PRESET:-}" "$1" "$2" "$3" > "$D/status.json"; }

# ---- preflight ----
[ "$(hostname)" = gpu-1-vm ] || { log "ABORT: not on gpu-1-vm"; exit 3; }
grep -q "session swift-opt" /u701/j692wu/qwen/rft/locks/gpu-1-vm.lock 2>/dev/null || { log "ABORT: our lock is missing"; exit 3; }
[ "$(gpu_used)" -lt ${GPU_FREE_MIB:-500} ] || { log "ABORT: GPU busy ($(gpu_used) MiB)"; exit 3; }
ss -ltn | grep -q ':8000 ' && { log "ABORT: port 8000 busy"; exit 3; }
[ "$(avail_gb)" -gt 150 ] || { log "ABORT: host RAM available only $(avail_gb) GiB"; exit 3; }

# ---- launch (own session so the whole process group can be stopped) ----
source $P/release/preset_env.sh || exit 3
export SWIFT_CHECKPOINT_PATH="${SWIFT_CHECKPOINT_PATH:-$(cat "$SW/model-path.txt")}"
SERVE="${SERVE_SCRIPT:-$P/release/serve_final.sh}"
setsid nohup bash "$SERVE" > "$D/server.log" 2>&1 < /dev/null &
SPID=$!
echo "$SPID" > "$D/server.pid"
log "launched server pid $SPID (serve=$SERVE preset=${PRESET:-})"

stop_server() {
  local pgid; pgid=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' ')
  if [ -n "$pgid" ] && [ "$pgid" = "$SPID" ]; then
    kill -TERM -- "-$pgid" 2>/dev/null
    for i in $(seq 1 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 3; done
    kill -0 "$SPID" 2>/dev/null && { log "server ignored TERM, sending KILL to group"; kill -KILL -- "-$pgid" 2>/dev/null; sleep 5; }
  else
    log "server pid $SPID not a group leader (pgid=$pgid); killing pid only"; kill -TERM "$SPID" 2>/dev/null; sleep 10
  fi
  for i in $(seq 1 60); do [ "$(gpu_used)" -lt ${GPU_FREE_MIB:-500} ] && break; sleep 3; done
  for i in $(seq 1 60); do [ "$(avail_gb)" -gt 150 ] && break; sleep 3; done
  log "after stop: gpu_used=$(gpu_used) MiB host_avail=$(avail_gb) GiB"
  if [ "$(gpu_used)" -ge ${GPU_FREE_MIB:-500} ] || [ "$(avail_gb)" -le 150 ]; then log "CLEANUP PROBLEM: GPU/RAM not released"; return 3; fi
  return 0
}

# ---- wait for readiness ----
t0=$(date +%s); state=waiting
while true; do
  if ! kill -0 "$SPID" 2>/dev/null; then state=exited; break; fi
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health 2>/dev/null)
  if [ "$code" = "200" ]; then state=ready; break; fi
  if [ $(( $(date +%s) - t0 )) -gt 2700 ]; then state=timeout; break; fi
  sleep 10
done
log "startup state=$state after $(( $(date +%s) - t0 )) s"
grep -E "NvFp4 MoE backend|Traceback|Error|OutOfMemory|CUDA out of memory" "$D/server.log" | cut -c1-400 > "$D/backend-selection.txt"
cp "$SERVE" "$D/serve.snapshot"

if [ "$state" != "ready" ]; then
  detail=$(grep -m1 -E "ValueError|NotImplementedError|Error" "$D/server.log" | cut -c1-200 | tr -d '"\\')
  status "$state" false "$detail"; log "NOT VALID: $state ($detail)"; stop_server; exit $?
fi

# ---- patch verification (every requested hook must really be active) ----
env | grep -E '^(SWIFT_|PYTHONPATH|CAPTURE_SIZES|GPU_UTIL|BACKEND|NSPEC|CUDA_ENV|PRESET)' > "$D/env.txt"
grep -F '[swift' "$D/server.log" | cut -c1-500 > "$D/patch-log.txt"
need() { # need FLAG_VALUE MARKER NAME
  if [ "$1" = 1 ] && ! grep -qF "$2" "$D/server.log"; then status ready_patch_inactive false "$3 not active"; log "NOT VALID: $3 not active"; stop_server; exit $?; fi
}
[ -n "${SWIFT_DRAFT_VOCAB_FILE:-}" ] && need 1 '[swift-draft-vocab] ACTIVE' 'draft vocab'
need "${SWIFT_SKINNY:-0}" '[swift_skinny] patched' 'skinny'
need "${SWIFT_L2PF:-0}" '[swift_l2pf] armed' 'l2pf'
need "${SWIFT_FP8:-0}" '[swift_fp8] converted' 'fp8'
need "${SWIFT_DRAFT_FP8:-0}" '[swift_draft_fp8] converted' 'draft fp8'
grep -qF 'captured WITHOUT prefetch' "$D/server.log" && log 'WARNING: some graphs were captured WITHOUT prefetch'
log "hooks: $(grep -oE '\[swift[^]]*\] (ACTIVE|patched|armed|converted)' "$D/server.log" | sort -u | tr '\n' ' ')"

# ---- telemetry + benchmark reps ----
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu,temperature.gpu,clocks_throttle_reasons.active --format=csv -lms 500 > "$D/telemetry.csv" 2>/dev/null &
TPID=$!
cp "$SW/experiments/20261021/decode-cases.json" "$D/decode-cases.json"
for r in $(seq 1 "$REPS"); do
  /usr/bin/python3 "$SW/experiment_bench.py" --root "$D" --kind decode --label "rep$r" > "$D/rep$r.out" 2>&1 \
    && log "rep$r: $(tail -1 "$D/rep$r.out" | /usr/bin/python3 -c 'import sys,json;d=json.loads(sys.stdin.read());print("mean_decode_tps=%.2f ttft=%.3f"%(d["mean_decode_tps"],d["mean_ttft_s"]))' 2>/dev/null)" \
    || log "rep$r FAILED (see rep$r.out)"
done
if [ "${OOD_REPS:-0}" -gt 0 ]; then
  mkdir -p "$D/ood"; cp "${OOD_CASES:-$PP/ood/decode-cases.json}" "$D/ood/decode-cases.json"
  for r in $(seq 1 "$OOD_REPS"); do
    /usr/bin/python3 "$SW/experiment_bench.py" --root "$D/ood" --kind decode --label "rep$r" > "$D/ood/rep$r.out" 2>&1 && log "ood rep$r done" || log "ood rep$r FAILED"
  done
fi
if [ "${CONC:-0}" = 1 ]; then
  /usr/bin/python3 $PP/bench_conc.py "$D/conc.json" ${CONC_LEVELS:-1 2 3 4} > "$D/conc.out" 2>&1 && log "conc: $(tr '\n' ' ' < "$D/conc.out" | cut -c1-300)" || log "conc FAILED: $(tail -1 "$D/conc.out" | cut -c1-200)"
fi
if [ -n "${EXTRA_CMD:-}" ]; then log "extra: $EXTRA_CMD"; bash -c "$EXTRA_CMD" > "$D/extra.out" 2>&1 || log "extra FAILED"; fi
kill "$TPID" 2>/dev/null
curl -s -m 10 http://127.0.0.1:8000/metrics > "$D/metrics-final.txt"
if [ "${KEEP_SERVER:-0}" = 1 ]; then status done true "kept"; log "KEEP_SERVER=1: leaving server pid $SPID running"; exit 0; fi
status done true ""
stop_server; exit $?
