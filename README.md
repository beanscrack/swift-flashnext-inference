# Swift-1.5-Qwen3.8-Flash-Next: local inference and optimizations

Serving `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` (Qwen3.8 Flash Next) with vLLM 0.31.0 on a single
RTX PRO 6000 Blackwell Max-Q (SM120, 96 GB, 300 W). Single-stream coding decode goes from ~195 tok/s (stock Marlin,
MTP3) to ~257 tok/s losslessly, ~304 tok/s with weight-only FP8, and ~316 tok/s with a retrained MTP draft head on
top. Production runs on gpu-8-vm; all experiments ran on gpu-1-vm.

## Results at a glance (thinking off, 6-prompt coding bench, single request)

| stage | tok/s | lossless? | details |
|---|---:|---|---|
| stock vLLM Marlin, MTP3 | ~195 | reference | `baseline/` |
| + 49,152-row draft vocab, MTP4, skinny GEMMs, L2 prefetch (`final-code`) | ~257 | yes (draft/kernels only) | `release/README.md`, `release/EXPLAINED.md` |
| + weight-only FP8 for dense layers, lm_head, MTP and draft head | ~304 | **no** (see quality) | `docs/FP8_RESULTS.md` |
| + retrained MTP draft head (`view-mtp-v1`) | ~316 | yes relative to the FP8 stack (target untouched) | `docs/MTP_RETRAIN_PLAN.md`, `mtp/README.md` |

Concurrency: widening the CUDA-graph capture sizes alone fixes multi-request serving (2 requests: 71 -> 297 tok/s
aggregate); see `release/README.md` section 2.

Quality of full FP8 against the BF16 release (paired): CodeContests 73.8 % vs 75.9 % (-2.1 pp, CI -5.1 to +1.0,
p 0.21), HumanEval+/MBPP+ 84.9 % vs 84.1 %, tool probe 95/96, logprob -0.212 vs ~-0.208. Not significant, but not
recommended for production yet. FP8 on the draft only, and the retrained draft, cannot change outputs.

## Layout

| path | what |
|---|---|
| `release/` | The deliverable: launcher (`serve_final.sh` -> `serve_exp.sh`), presets (`preset_env.sh`), env-gated import hooks (`hooks/`: draft vocab, skinny GEMM, L2 prefetch, FP8), skinny and FP8 plans, draft vocabs, provenance. Start with `release/README.md` and `release/EXPLAINED.md`. |
| `docs/SERVING.md` | **How to serve:** the configurations, the production launch flags to keep (`SWIFT_CONTEXT=262144`, `--generation-config auto`), how to swap the gpu-8-vm server, and benchmarking rules for gpu-1-vm. |
| `docs/FP8_RESULTS.md` | Weight-only FP8 (`SWIFT_FP8`): design, launch line, throughput and quality results. |
| `docs/FP8_SCOPE.md` | The original FP8 scoping note, written before implementation. |
| `docs/MTP_RETRAIN_PLAN.md` | MTP draft retraining: plan, gates, and the full dated progress log P0-P5 with every number. |
| `mtp/` | MTP retraining code and evidence (views, PyTorch draft model, capture hooks, data generation, TTT training, simulators, A/B analysis, verifier outputs). See `mtp/README.md`. |
| `baseline/` | The original un-patched deployment and its measurements. |
| `tools/` | `run_config.sh` (one server config end to end: start, verify hooks, bench, OOD, concurrency, score, stop), `quality_session.sh` (task-level quality suites), `fp8_bench.py`, `make_fp8_plan.py`. |
| `results/runs/` | FP8 benchmark driver logs. |
| `results/quality/` | Logprob scores and task-level quality comparisons (`task_level/`). |
| `results/fp8_gate/` | FP8 kernel accuracy/speed gate. |
| `results/mtp_runs/` | MTP A/B runs (P0 identity, P4, P5): driver logs, env, hooks, metrics, scores. |

## Quick start

```bash
# lossless release presets: baseline-wide | final-code | final-multi (default)
PRESET=final-code bash release/serve_final.sh

# production-style: 262K context, model sampling defaults, retrained draft
SWIFT_CONTEXT=262144 PRESET=final-code SWIFT_CHECKPOINT_PATH=/u701/j692wu/swift-opt/mtp/view-mtp-v1 \
  bash release/serve_final.sh --generation-config auto
```

After start-up, check the server log for `[swift-draft-vocab] ACTIVE`, `[swift_skinny] patched`, `[swift_l2pf] armed`
(and `[swift_fp8] converted N modules` with FP8). The L2-prefetch hook has no strict mode and can silently fall back
to the unmodified model. More configurations and the swap procedure: `docs/SERVING.md`.

## How the speed was found

- **Lossless stack (195 -> 257).**
  - Profiling showed that decode is weight-bandwidth bound: about 15.9 ms per verify+draft cycle.
  - The draft's 248K-row output head was trimmed to the 49K tokens coding output actually uses.
  - The draft depth was raised to 4.
  - vLLM's CuTe skinny-GEMM path was enabled on SM120 for small-M matmuls.
  - The next layer's weights are prefetched into L2 on a side stream.
- **FP8 (257 -> 304).** The model is NVFP4 only in its MoE experts; the dense attention/GDN projections,
  `lm_head` and the MTP layer were still BF16. Storing them as FP8 e4m3 with per-row scales, read by custom
  Triton GEMV kernels, halves those bytes (6.27 -> 3.14 GiB). This changes numerics, so the target is no longer
  bit-equivalent.
- **Retrained draft (304 -> 316).** Acceptance, not kernels, was the next limit: about 3.43 tokens per cycle.
  The draft head was distilled on 6.4 M tokens of the model's own coding output, with all 4 draft steps
  unrolled during training. A pre-validated offline simulator was checked against live acceptance and then
  used to select the checkpoint. A fresh agent verified every phase independently.

## Caveats

- Patches target vLLM 0.31.0 internals; the skinny and FP8 plans were tuned on this GPU only.
- Marlin MoE is non-deterministic at temperature 0, so outputs differ run to run in every configuration.
  Long-context losslessness is unproven (section 5 of `release/README.md`).
- All speed numbers were measured with thinking off. The retrained draft's speed with thinking on was measured
  offline only: 3.09 -> 3.42 tokens/cycle on heldout.
- The model checkpoint, retrained weights (`~/swift-opt/mtp/p3b/ckpt/best.safetensors`, view
  `~/swift-opt/mtp/view-mtp-v1`), captured hidden states and raw server logs stay on the cluster.
- Scripts hardcode cluster paths (`/u701/j692wu/...`).
- No license has been chosen yet.
