# Scope: FP8 weight-only dense layers (and FP8 KV cache) for Swift-1.5 on vLLM 0.31 / one RTX PRO 6000

Written 2026-10-23. Status: **scoped, nothing implemented, no GPU time used.** Inputs: safetensors headers of the checkpoint, the final-stack Nsight profile
(`nsys/final-mtp4/final.sqlite`, 169 cycles), vLLM 0.31 source, and the public repo aiueo52/flash-next-rtxpro6000 (SGLang; Apache-2.0).

## 1. What would change

The NVFP4 checkpoint quantizes only the routed experts. Everything in the `ignore` list of its `quantization_config` (323 entries) stays BF16 and is read in full on
every verify cycle. FP8 here means **weight-only FP8 (e4m3, one fp32 scale per output row), BF16 activations, BF16 math** ("W8A16"): half the bytes, same compute.
It is a bandwidth change, not a compute change. The KV cache is a separate switch (section 5).

Bytes read per verify cycle, from the checkpoint headers (target model; BF16 today):

| group | tensors | MiB | FP8 candidate? |
|---|---|---:|---|
| linear_attn (36 layers): in_proj_qkv 1800, in_proj_z 1080, out_proj 1080, in_proj_a/b 17 | 180 (72 are the tiny a/b) | 3,977 | yes (a/b are tiny, leave BF16) |
| self_attn (12 layers): q 720, o 360, k 30, v 30 | 48 | 1,140 | yes |
| self_attn indexer (QSA top-k selection) | 12 | 37.5 | **no** (the repo also hard-excludes it) |
| shared experts (48 layers) | 144 | 450 | yes |
| lm_head (248,320 x 2,560) | 1 | 1,212.5 | yes, last |
| hyper-connection mixers (attn + mlp, down + up) | 192 | ~1,212 | skip for now (small kernels at ~0.7-0.85 TB/s, latency-bound; not evaluated here) |
| router gate (512 x 2,560) | 48 | 120 | **no** (repo: FP8 lowered acceptance) |

Candidate set without lm_head: 5,567 MiB = 5.84 GB -> 2.92 GB. With lm_head: 3.55 GB saved per cycle in total. The repo reports 3.73 GB after conversion for its set.
The draft (MTP) layer has its own BF16 dense weights and 512 BF16 experts (4.7 GiB); see section 6.

## 2. Size of the prize (estimate, not a measurement)

Final stack, Nsight, per verify cycle (kernel time 14.67 ms under nsys vs 13.7 ms measured): the BF16 skinny-GEMM family (`cute_dsl_skinny_gemm`) takes **4.28 ms (29%)** in
321 launches; Marlin MoE 3.30 ms; Triton `_dot_kernel` 0.99 ms; the three HC kernels about 1.3 ms. The skinny family reads roughly 5.84 GB of dense weights (approximate: the family also
includes the draft layer's dense GEMMs) in 4.28 ms, about 1.4 TB/s effective, i.e. already near the measured ~1.6 TB/s read roof. So the family is bandwidth-bound and halving the bytes should help.
- If the FP8 kernels reach the same efficiency: save about 2.1 ms. At the repo's measured FP8 GEMV efficiency (74-88% of 1.6 TB/s): about 1.7-2.0 ms.
- 13.7 ms -> about 11.7-12.0 ms, i.e. +14-17% tok/s as an upper bound. After haircuts (tiny shapes that do not benefit, conversion/dispatch overhead, imperfect
  overlap with the L2 prefetch) plan for **+8-15%**; lm_head adds about +2%. That would put `final-code` at roughly 275-295 and `final-multi` at 260-275 tok/s on the same benchmark.
- Secondary effect: about 3 GB of VRAM freed (6 GB BF16 -> 3 GB), which goes to the KV pool (about 10.4 GiB today at 0.95 utilization). That recovers most of final-multi's
  10% KV-pool deficit if the conversion runs before the memory profile.

## 3. How it would be built (same style as the existing hooks)

New hook `release/hooks/swift_fp8_dense.py`, installed from `sitecustomize.py` like `swift_skinny_patch.py` (import hook on the Qwen4Exp modules; no edits to vLLM):
1. After weight load, for the selected categories: quantize each BF16 weight to e4m3 with a per-output-row scale, free the BF16 tensor.
2. Forward dispatch: M <= 20 (decode/verify) -> weight-only FP8 GEMV; M > 20 (prefill) -> dequantize into one shared BF16 scratch buffer (largest weight 52 MB) and call the existing BF16 GEMM.
   No resident BF16 copy and **no activation quantization anywhere** (the repo's first version used W8A8, which also quantizes prefill activations; avoid that).
3. Kernel choice, decided by the microbenchmark gate below: (a) vLLM's own FP8 Marlin (`marlin_utils_fp8.prepare_fp8_layer_for_marlin` / `apply_fp8_marlin_linear`, present in the wheel and supported from
   SM75 up); (b) the repo's Triton W8A16 GEMV (`kernels/sglang/srt/layers/quantization/w8a16_gemv.py`, `microbench/w8a16v2/final_v3.py` has the tuned tile table and tuner); (c) a CuTe-DSL FP8 variant of vLLM's skinny GEMM if (a) and (b) disappoint (most work).
   vLLM's CuTe skinny path (`low_latency_gemm.py`) is BF16-only today.
4. Per-category env flags (shared_expert, self_attn, linear_attn, lm_head), strict mode (abort at start-up if it cannot arm), and one log line
   `[swift_fp8] converted N modules, saved X GiB` so a silent no-op is visible (the L2 hook has no strict mode; do not repeat that).
5. Retune: the L2 prefetch budget (48 MB covers twice as many weights at FP8), the skinny plan (converted modules leave it), CUDA-graph capture is unaffected.

## 4. Plan, gates and cost

| phase | work | GPU? | gate to continue |
|---|---|---|---|
| 1 | microbenchmark each real shape ([10240,2560], [6144,2560], [2560,6144], [12288,2560], [640,2560], [2560,640], lm_head) at M = 1..25: current BF16 skinny vs FP8 Marlin vs Triton W8A16; check relative error vs fp32 | yes, ~1 h (needs a GPU claim) | summed saving >= 1.0 ms per cycle |
| 2 | write and unit-test the hook; smoke-start a server; confirm the log line and graph capture | yes, ~11 min per start | server runs, outputs sane |
| 3a | reference-logprob scoring of FP8 vs BF16 outputs (same method as the skinny check) and draft acceptance | yes | no drop beyond noise |
| 3b | needles at 32K/64K/120K (the Gated-DeltaNet state accumulates error, so linear_attn is the risky group) | yes | all correct |
| 3c | CodeContests LOCKED 135 x 6 and HumanEval+/MBPP+, paired against the existing final-multi samples | yes, ~1.5 h per server | CI not worse than about -1.5 pp |
| 4 | rerun the 8-session stress and the speed benchmarks; update FINAL.md and the release folder | yes | no new failures |

Add categories in this order and stop at the first quality signal: shared_expert, self_attn, linear_attn, lm_head. Effort: about 2-3 days of work, about 6-8 GPU hours (5-6 server starts).

## 5. FP8 KV cache (separate, almost free to try)

vLLM 0.31's Qwen4Exp QSA backend already lists `fp8` / `fp8_e4m3` as supported KV dtypes (`models/qwen4_exp/nvidia/qsa.py`: e4m3 bytes in a uint8 cache, dequantized inside its own Triton kernel; the
parent's flash-attn FP8 probe, which fails on SM120, is bypassed). So `--kv-cache-dtype fp8` may work with no code. Expected effect: no decode-speed change at short context (only 12 of 48 layers have a KV cache), but more KV capacity.
It will not double the pool: only the attention-cache part halves, the Gated-DeltaNet state does not. Untested, and the checkpoint has `kv_cache_quant_algo: null`, so scales are not calibrated (default 1.0): needs the needle/CodeContests checks above.

## 6. Other findings from the inventory

- The MTP draft layer's 512 experts are BF16 (4.7 GiB) and run through Triton `fused_moe_kernel` (0.52 ms per cycle, 3.5%; attribution inferred from the launch count of about 8 per cycle = 4 draft forwards x 2 GEMMs). Quantizing them (FP8 weight-only via Marlin MoE, `prepare_fp8_moe_layer_for_marlin`) could save
  about 0.25 ms (+2%). Because the target verifies every drafted token, this changes only the acceptance rate, not the target's output, so it carries no quality risk beyond speed. Lower priority.
- Not worth doing from the repo: singleton-route pruning (changes the target's arithmetic, non-inferiority not established), HC-mixer FP8, the fine-tuned MTP head (needs private-style data), megakernels (the repo also found them slower).

## 7. Open risks

1. Marlin FP8 or the Triton GEMV may not beat our tuned BF16 skinny kernels on the small shapes (the repo measured a 2,560 x 2,560 FP8 GEMV slower than BF16: 9.8 vs 8.0 us). Phase 1 decides this before any hook is written.
2. FP8 rounding changes the target's numerics, so greedy outputs will differ from BF16 (as the skinny GEMMs already do). Not lossless; it must be validated and labelled that way.
3. The repo's own FP8 validation is thin (relative error ~1e-3, needle at 18.5K, unchanged acceptance); it did not publish a task-level FP8-only quality comparison. We would be the first to measure it on Swift.
4. Memory accounting: the conversion must finish before vLLM's memory profile, otherwise the freed 3 GB is not turned into KV pool.
