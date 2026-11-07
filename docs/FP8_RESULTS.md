# FP8 weight-only dense layers: changes, launch, results

Status: experimental. FP8 changes the numerics of the model and is **not lossless**. Quality has been checked only
with a teacher-forced logprob comparison so far; a task-level comparison (HumanEval+, MBPP+, CodeContests) is in progress.

Tested on gpu-1-vm (RTX PRO 6000 Blackwell Max-Q, 96 GB), vLLM 0.31.0, preset `final-code`, thinking mode off.

## What changed

- **Format:** weight-only FP8 e4m3 storage with per-output-row fp32 scales (`SWIFT_FP8_GROUP=0`; per-128-K-group
  scales are available as an option). Activations stay BF16, accumulation is fp32.
- **Kernels** (`release/hooks/fp8_gemv.py`): Triton GEMV kernels for small M (`w8fma`) and tensor-core dot kernels
  (`w8dot`) for larger M. Per-shape kernel configs come from `release/plan/fp8_plan_row.json`, keyed by `NxK`
  (`16384x2560`, `2560x6144`, `13312x2560`, `248320x2560`) and token count M.
- **Conversion and dispatch** (`release/hooks/swift_fp8_dense.py`): installed by `sitecustomize.py` when `SWIFT_FP8=1`,
  after the skinny hook. Weights are converted at load time, and a self-check runs at the token counts in
  `SWIFT_FP8_WARM_MS`. `SWIFT_FP8_STRICT=1` (default) aborts start-up if nothing is selected or a check fails.
- **Layers converted** (from the `fp8-all` server log):
  - Target model, 97 modules: GDN `in_proj_qkvz` (`qkvz`, 36) and `out_proj` (`gdn_out`, 36); QSA `qkv_proj`
    (`qsa_qkv`, 12) and `o_proj` (`qsa_o`, 12); `lm_head` (1).
  - MTP layer, 4 modules: `qsa_qkv`, `qsa_o`, shared-expert `gate_up` and `down`.
  - Total 101 modules, 6.27 GiB -> 3.14 GiB of weights.
- **Large-M fallback:** above `SWIFT_FP8_MAX_M` (default 32; used for prefill and prompt logprobs), the hook
  dequantizes at most `SWIFT_FP8_DEQ_CHUNK_ROWS` (default 8192) weight rows at a time and calls cuBLAS (`F.linear`).
- **FP8 draft head** (`release/hooks/swift_draft_vocab.py`, `SWIFT_DRAFT_HEAD_FP8=1`): the trimmed 49,152-row
  draft `lm_head` used for MTP proposals is stored in FP8. The target model's head is not affected by this flag;
  it is covered by `lm_head` in `SWIFT_FP8_TARGETS`.
- **HC-up strided dispatch fix** (`release/hooks/swift_skinny_patch.py`, `SWIFT_SKINNY_STRIDED`, default on):
  enables strided-activation dispatch for Triton plan entries.

## How to launch

```bash
PRESET=final-code SWIFT_FP8=1 SWIFT_DRAFT_HEAD_FP8=1 SWIFT_FP8_TARGETS=qkvz,gdn_out,qsa_qkv,qsa_o,lm_head,mtp SWIFT_FP8_PLAN=$PWD/release/plan/fp8_plan_row.json bash release/serve_final.sh
```

After start-up, the server log should contain `[swift_fp8] converted 101 modules` (with `mtp` in the targets),
`[swift-draft-vocab] ACTIVE`, and `[swift_skinny] patched`. The hooks and scripts in `release/` hardcode
`/u701/j692wu/...` paths.

## Results

Decode throughput from the driver logs in `results/runs/<label>/driver.log`. Each row is the mean of the per-rep
`mean_decode_tps` values in that log. Speedup is relative to the baseline row.

| Configuration | Label | Reps | Per-rep mean_decode_tps | Mean tok/s | vs baseline |
|---|---|---:|---|---:|---:|
| Baseline, final-code | `base-final-code` | 3 | 251.24, 254.57, 255.61 | 253.81 | - |
| + FP8 dense (qkvz, gdn_out, qsa_qkv, qsa_o) | `fp8-dense` | 3 | 280.75, 278.98, 280.83 | 280.19 | +10.4% |
| + lm_head & MTP | `fp8-head-mtp` | 3 | 298.33, 298.94, 298.01 | 298.43 | +17.6% |
| + FP8 draft head | `fp8-all` | 5 | 302.76, 304.63, 300.81, 301.34, 309.74 | 303.86 | +19.7% |
| + FP8 draft head (scoring run) | `fp8-all-score` | 3 | 310.26, 306.85, 304.00 | 307.04 | +21.0% |

The `fp8-all` and `fp8-all-score` rows are two separate runs of the same configuration; they were not pooled.

Kernel-level check (`results/fp8_gate/gate1.log`, `gate1.json`): for the 16384x2560 `qkvz` shape at M=1, the FP8
`w8row` kernel took 27.3 us, against 55.6 us for BF16 torch and 51.8 us for the BF16 plan kernel. Its max relative
error was 2.8e-2, against 2.5e-3 for BF16 torch.

### Quality (teacher-forced logprob check)

- FP8 (full configuration, `fp8-all-score` run): mean logprob -0.2124, rank-1 (top-1) agreement 97.53%.
- BF16 Marlin runs: mean logprob -0.2075 to -0.2089, top-1 98.6% to 99.0%.

These figures come from the scoring output of the `fp8-all-score` run (`scores.json`, which stays on the cluster
and is not in this repository), so they cannot be re-derived from the logs committed here. A task-level comparison
(HumanEval+, MBPP+, CodeContests) is in progress and is not reported here.

## Notes

- An earlier `fp8-all` attempt failed at start-up with `UnboundLocalError: w_sub` (`fp8-all.failed1`). The current
  `swift_draft_vocab.py` defines `w_sub` before use. Only the successful run is included in `results/runs/`.
- `fp8/fp8_gemv.py` in the working directory on the cluster is an older duplicate of `release/hooks/fp8_gemv.py`.
  Only the `release/hooks` copy is in this repository.
