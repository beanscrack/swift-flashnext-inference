# Serving guide (gpu-8-vm production, gpu-1-vm testing)

All launch scripts hardcode cluster paths (`/u701/j692wu/...`). The repo copy of the launcher is `release/`; the
deployed copies are `~/swift-profile/release/` (original, no FP8 hooks) and `~/swift-opt/release/` (adds the FP8
hooks and `SWIFT_DRAFT_HEAD_FP8`). Everything is env-gated, so with no extra variables `~/swift-opt/release` behaves
like `~/swift-profile/release`.

## Configurations

| config | extra env (on top of `PRESET=final-code`) | single-stream coding tok/s | output vs BF16 release |
|---|---|---:|---|
| A. production as deployed | none (`~/swift-profile/release/serve_final.sh`) | ~257 | reference |
| B. FP8 draft only | `SWIFT_FP8=1 SWIFT_FP8_TARGETS=mtp SWIFT_DRAFT_HEAD_FP8=1 SWIFT_FP8_PLAN=<plan>` | not benchmarked | identical (only the draft changes) |
| C. full FP8 | `SWIFT_FP8=1 SWIFT_FP8_TARGETS=qkvz,gdn_out,qsa_qkv,qsa_o,lm_head,mtp SWIFT_DRAFT_HEAD_FP8=1 SWIFT_FP8_PLAN=<plan>` | ~304 | changed (`docs/FP8_RESULTS.md`; not recommended for production yet) |
| C + retrained draft | C plus `SWIFT_CHECKPOINT_PATH=/u701/j692wu/swift-opt/mtp/view-mtp-v1` | ~316 | same as C (draft-only change) |
| A/B + retrained draft | add `SWIFT_CHECKPOINT_PATH=/u701/j692wu/swift-opt/mtp/view-mtp-v1` | not benchmarked; expect ~+4 % | identical to A |

`<plan>` = `/u701/j692wu/swift-opt/release/plan/fp8_plan_row.json`. Speeds are gpu-1-vm, thinking off, 6-prompt
coding bench (`release/README.md` section 1). The retrained draft was trained and A/B-tested with the 49,152-row
`final-code` draft vocab and full FP8 (C) only; `final-multi` with the retrained draft is untested.

"Output identical" means the target model's weights and numerics are untouched; speculative decoding is lossless,
so text differs from the reference only by Marlin MoE run-to-run non-determinism, which exists in every config.

## Launch flags you must keep from the production command

The production command on gpu-8-vm was:

```bash
SWIFT_CONTEXT=262144 SWIFT_CHECKPOINT_PATH=/tmp/swift-cache-j692wu-20261021 PRESET=final-code \
  bash ~/qwen/rft/bg.sh ~/swift-profile/serve.log bash ~/swift-profile/release/serve_final.sh --generation-config auto
```

- `SWIFT_CONTEXT=262144`: `serve_exp.sh` defaults to `--max-model-len 131072`. Leave it out and long agent
  sessions run out of context. One 262K sequence needs ~8.8 GiB of KV cache. If start-up fails in
  `_check_enough_kv_cache_memory`, lower it (e.g. 196608).
- `--generation-config auto`: `serve_exp.sh` hardcodes `--generation-config vllm`; the trailing flag overrides it
  and makes the model's own sampling defaults (temperature, top_p, ...) apply to requests that do not set them.
- `SWIFT_CHECKPOINT_PATH=/tmp/swift-cache-...`: gpu-8-vm's local-disk copy of the checkpoint (fast load). The
  retrained-draft view (`view-mtp-v1`) instead links to the NFS copy, so start-up is slower (~19 min on gpu-1).
  To get a view on the local copy, build one on gpu-8-vm (CPU only; aborts if names/shapes differ, and the local
  copy's revision has not been checked against the tested one):
  `cd ~/swift-opt/mtp && <conda python> make_view.py --mtp p3b/ckpt/best.safetensors --ck /tmp/swift-cache-j692wu-20261021 --out ~/swift-opt/mtp/view-mtp-v1-gpu8`
- `MAX_SEQS` (default 4) and `GPU_UTIL` (default 0.95) are also read by `serve_exp.sh`.

## Example: B + retrained draft at 262K context

```bash
SWIFT_CONTEXT=262144 PRESET=final-code \
SWIFT_FP8=1 SWIFT_FP8_TARGETS=mtp SWIFT_DRAFT_HEAD_FP8=1 \
SWIFT_FP8_PLAN=/u701/j692wu/swift-opt/release/plan/fp8_plan_row.json \
SWIFT_CHECKPOINT_PATH=/u701/j692wu/swift-opt/mtp/view-mtp-v1 \
  bash ~/qwen/rft/bg.sh /u701/j692wu/swift-opt/runs/gpu8-versionB-mtpv1.log \
  bash /u701/j692wu/swift-opt/release/serve_final.sh --generation-config auto
```

## Swapping the production server

Only one server fits on the GPU (~94 GB) and port 8000 is fixed, so the old one must be stopped first. Expect
~20 min of downtime while the new server starts.

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv     # server PID
ps -o pid,pgid,etime,cmd -p <PID> | cut -c1-200                   # its process group; save the cmd for rollback
kill -TERM -- -<PGID>                                             # whole group (bg.sh uses setsid)
watch -n 3 'nvidia-smi --query-gpu=memory.used --format=csv,noheader; ss -ltn | grep -c :8000'
#   wait for ~0 MiB and 0; if not freed after ~2 min: kill -KILL -- -<PGID>
# launch the new config, then:
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health   # 200 when ready
```

Check the server log after start-up:
- `serve_exp: ... MODEL=<expected path>` (confirms which checkpoint/view is loaded)
- `[swift-draft-vocab] ACTIVE`, `[swift_skinny] patched`, `[swift_l2pf] armed`
- with FP8: `[swift_fp8] converted N modules` (101 for config C; the MTP-only modules for B)

Rollback: rerun the saved production command unchanged (no `SWIFT_CHECKPOINT_PATH` to the view = original draft).

## Benchmarking on gpu-1-vm

`tools/run_config.sh LABEL [REPS]` starts a server with the current env, verifies hooks, runs the coding bench,
optional OOD (`OOD_REPS`), concurrency (`CONC=1 CONC_LEVELS="1 2 4 8"`) and an `EXTRA_CMD` (e.g. the logprob
scorer), then stops the server and checks cleanup. `tools/quality_session.sh LABEL` runs the task-level quality
suites. Rules learned the hard way:

- gpu-1-vm is shared. Our lock file (`/u701/j692wu/qwen/rft/locks/gpu-1-vm.lock`) is a convention, not
  enforcement: other users' Slurm jobs and Jupyter kernels do land on it. Check
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` first and never kill other users' processes.
  `GPU_FREE_MIB` (default 500) lets `run_config.sh` accept a small idle co-tenant.
- Never `pkill -f <pattern>` over ssh: the pattern matches the ssh command itself and kills the session.
- Launch long jobs fully detached (`setsid nohup ... > log 2>&1 < /dev/null &`), preferably via a small wrapper
  script, or the ssh call hangs on the inherited pipe.
- Server start-up is 18-20 min; a 5-rep run with OOD, concurrency and scoring is ~30 min.
- tok/s varies ~2-3 % run to run (one-off slow reps happen); compare configs back to back and with several reps.
  Accepted tokens per cycle (`metrics-final.txt`: accepted / drafts + 1) is the steadier signal.
