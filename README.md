# Swift-1.5-Qwen3.8-Flash-Next: local inference and optimizations

Serving `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` (Qwen3.8 Flash Next) with vLLM 0.31.0 on a single
RTX PRO 6000 Blackwell Max-Q (SM120, 96 GB). Single-stream coding decode goes from ~195 tok/s (stock Marlin, MTP3)
to ~256 tok/s with the patches here, with no detectable quality loss in the checks that were run.

## Layout

| path | what |
|---|---|
| `release/` | The deliverable. Launcher (`serve_final.sh`), env-gated import hooks, draft-vocab presets, skinny-GEMM plan, provenance scripts. **Start with `release/README.md`** (full results) and `release/EXPLAINED.md` (why each piece works). |
| `baseline/` | The original un-patched deployment (`serve.sh`, eager vs CUDA-graph comparison, context sweep, latency/utilization scripts) with its README. |
| `docs/FP8_SCOPE.md` | Original scoping note for FP8 (written before implementation; the implemented version and results are in `docs/FP8_RESULTS.md`). |
| `docs/FP8_RESULTS.md` | Weight-only FP8 dense layers (`SWIFT_FP8`): what changed, launch line, throughput results and quality status. |

## Quick start

```bash
# presets: baseline-wide | final-code | final-multi (default)
PRESET=final-multi bash release/serve_final.sh
```

After start-up, check the server log for `[swift-draft-vocab] ACTIVE`, `[swift_skinny] patched` and `[swift_l2pf] armed`.
The L2-prefetch hook has no strict mode and can silently fall back to the unmodified model.

## Summary of results (thinking mode off for every measurement)

| config | single-request tok/s |
|---|---:|
| stock vLLM Marlin, MTP3 | ~195 |
| final-code (MTP4 + 49K draft vocab + skinny GEMMs + L2 prefetch) | ~257 |
| final-multi (same, 126,868-row multilingual draft vocab) | ~240 |

Widening the CUDA-graph capture sizes alone fixes concurrency (2 requests: 71 -> 297 tok/s aggregate). See
`release/README.md` for the held-out multilingual set, quality checks, stability runs and what did not work.

## Caveats

- Patches target vLLM 0.31.0 internals; the skinny plan was tuned on this GPU only.
- Marlin is non-deterministic at temperature 0, and long-context losslessness is unproven (see section 5 of `release/README.md`).
- The model checkpoint and raw run data (server logs, Nsight traces, per-request JSON) are not in this repo; they remain on the cluster. The FP8 driver logs and status files are in `results/runs/`.
- Scripts contain hardcoded cluster paths (`/u701/j692wu/...`) that need adjusting elsewhere. This includes the hooks and scripts under `release/` and the FP8 launch line in `docs/FP8_RESULTS.md`.
- No license has been chosen yet.
