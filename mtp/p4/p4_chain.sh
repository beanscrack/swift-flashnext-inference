#!/usr/bin/env bash
# P4 chain: wait for gpu-1 to be free of co-tenants, then p4-identity and p4-mtp-v1 back to back, then analyze.
set -u
P=/u701/j692wu/swift-opt; L=$P/mtp/p4/chain.log
log() { echo "$(date -u +%FT%TZ) [p4-chain] $*" | tee -a "$L"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
COMMON="GPU_FREE_MIB=1500 PRESET=final-code SWIFT_FP8=1 SWIFT_DRAFT_HEAD_FP8=1 SWIFT_FP8_TARGETS=qkvz,gdn_out,qsa_qkv,qsa_o,lm_head,mtp SWIFT_FP8_PLAN=$P/release/plan/fp8_plan_row.json OOD_REPS=2 CONC=1"
t0=$(date +%s); free_since=0
while true; do
  if [ "$(gpu_used)" -lt 1500 ]; then
    [ $free_since = 0 ] && free_since=$(date +%s)
    [ $(( $(date +%s) - free_since )) -ge 180 ] && break   # free for 3 min (not just between two jobs)
  else free_since=0; fi
  [ $(( $(date +%s) - t0 )) -gt 43200 ] && { log "ABORT: gpu not free within 12 h"; exit 3; }
  sleep 60
done
log "gpu free after $(( ($(date +%s) - t0) / 60 )) min"
cd $P
for pair in "p4-identity:$P/mtp/view-identity" "p4-mtp-v1:$P/mtp/view-mtp-v1"; do
  lab=${pair%%:*}; view=${pair#*:}
  log "start $lab"
  env $COMMON CONC_LEVELS="1 2 4 8" SWIFT_CHECKPOINT_PATH=$view \
    EXTRA_CMD="/usr/bin/python3 /u701/j692wu/swift-profile/score/score_logprobs.py $P/runs/$lab/scores.json" \
    bash run_config.sh $lab 5 > runs/$lab.nohup 2>&1 < /dev/null
  rc=$?; log "$lab rc=$rc status=$(cat runs/$lab/status.json 2>/dev/null)"
  [ $rc = 0 ] || { log "ABORT after $lab"; exit 3; }
done
/u701/j692wu/.conda/envs/inference/bin/python $P/mtp/p4/analyze.py --hash > $P/mtp/p4/analyze.out 2>&1
log "analyze rc=$?"; log "done"
