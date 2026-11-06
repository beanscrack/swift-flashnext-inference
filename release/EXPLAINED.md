# How the Swift stack got from ~195 to ~256 tok/s, and why each piece works (v2, after independent verification)

Setting: vLLM 0.31.0 serving Swift-1.5-Qwen3.8-Flash-Next on one RTX PRO 6000 Blackwell Max-Q (SM120, 96 GB **GDDR7** on a 512-bit bus = 1.79 TB/s spec peak DRAM bandwidth; the memory clock runs at ~13.4 of 14.0 GHz under CUDA load, so the operating peak is ~1.71 TB/s and all %-of-peak figures are relative to the spec value; 128 MiB L2,
300 W cap). vLLM's source is unmodified; the changes are launch flags, import hooks (`sitecustomize`) that also load a few new Triton kernels, and data files (draft vocabularies, a GEMM plan).
"Baseline" = vLLM Marlin MoE with 3-token MTP as originally deployed, 195.1 tok/s = the mean of seven baseline runs including a 1-rep scoring run (the six runs listed in FINAL.md average 194.7; spread 193-197); percentages below use 195.1 unless a run is named (against 194.7 they would be ~0.3-0.5 points higher). "Cycle time" is derived as
tokens-per-verification / tok/s (not read from a profiler). The MTP-off 103.8 and MTP3 197.5 tok/s figures come from the earlier swift-local benchmark, not this campaign.
Items marked (hypothesis) are explanations that were not directly tested.

## 0. What decode speed is made of

Decoding is mostly **bandwidth-bound**: the big kernels (Marlin MoE, the dense GEMMs, the lm_heads) are limited by how fast weights stream out of DRAM, and one cycle moves roughly 17 GB (estimate: the 4.1 GB MoE share
assumes Marlin runs at ~89% of peak; it was not measurable because Nsight Compute counters are blocked). That is a floor of ~9.7 ms of a 15.9 ms cycle (61% of peak); the other third is latency-bound small kernels and kernels below peak.
So the wins fall into three kinds: move fewer bytes, move them nearer peak bandwidth, or keep the GPU busy.

**MTP speculative decoding** (already on): a small draft module guesses the next k tokens; the full model checks the current token plus the k guesses in ONE forward pass (k+1 tokens), so weights are read once for several tokens.
With greedy decoding a guess is kept only if it equals the target's argmax at that position, so the emitted tokens are the target's own greedy choices as computed in the verify pass (whose numerics differ slightly from one-token decoding).
Baseline: no MTP 103.8 tok/s, MTP3 197.5 (about 1.9x), ~70% of drafted tokens accepted, ~3.1 tokens per cycle.

**One cycle** = 1 verify pass + k draft steps + sampling. Baseline Nsight profile (profiled cycle 16.2 ms; 15.9 ms unprofiled): verify graph ~11.5 ms (1,636 kernels: Marlin MoE ~2.6 ms, dense BF16 GEMMs ~6 ms, many small kernels),
main lm_head 0.8 ms, three draft steps ~3.7 ms of which the draft lm_head GEMV is 3 x 0.84 = 2.5 ms (16%).

## 1. Wider CUDA-graph capture sizes (the concurrency fix)

A **CUDA graph** records a fixed sequence of GPU kernel launches (the verify graph has 1,636 kernels; ~1,900 per cycle across the verify and draft graphs) so one CPU call replays it. vLLM captures graphs only for the token counts you list;
a step is padded up to the next captured size, and if none fits it runs eagerly.

The deployment listed `[1,2,4]`. One request under MTP3 verifies 4 tokens (graph exists); two requests verify 8 tokens, for which no graph exists, so the verify pass and the first draft step ran eagerly (the later draft steps still had graphs).
Measured: ~36 tok/s per request and 71 tok/s aggregate at 2 requests, i.e. ~85 ms per eager cycle against ~16 ms graphed. That is consistent with CPU launch overhead, which I did not profile.

Fix: list (k+1) x C tokens for C=1..4 (`[1,2,3,4,8,12,16]` for MTP3, `[1,2,3,4,5,10,15,20]` for MTP4) plus the draft sizes. **Effect with stock vLLM, no patches:** aggregate 71 -> 297 tok/s at C=2 (4.2x) and 137 -> 478 at C=4 (3.5x);
per-request 36 -> 162 tok/s at C=2. Single-request speed is unchanged. Note: the single-request numbers in sections 2-7 were measured with capture sizes that cover C=1, mostly the original `[1,2,4]` or `[1,2,4,5]` for MTP4 (exceptions: the fresh held-out final-multi arm used the release config, the baseline mean includes one run with wide sizes, MTP6 used `[1,2,4,7]`);
all are equivalent at C=1. The concurrency comparison (section 9) uses the widened sizes for both arms.

## 2. Trimming the draft model's output head (+12.8% alone)

**Why:** the draft shares the target's lm_head (248,320 rows x 2,560 BF16 = 1.27 GB). Each draft step multiplies by all rows only to take an argmax: 1.27 GB at ~1.5 TB/s = 0.84 ms; with 3 drafts that is 2.5 ms (16% of the cycle).
Draft tokens are only proposals that the target verifies, so the head needs to contain the tokens the target is likely to say, not all of them.

**What I did:** keep a frequency-ranked subset (49,152 rows = 19.8%): copy those rows once, after weights load and before graph capture, into a contiguous matrix; compute draft logits over the subset, argmax, map the index back to the real id
through an id table. The target's lm_head, sampler and verification are untouched. The subset ranks tokens by frequency in code, docs and other models' code/reasoning outputs, always keeping special, byte, digit and whitespace tokens, sorted ascending so ties resolve to the lowest id as the full argmax does.

**Effect:** draft head 0.84 -> ~0.16 ms per step; cycle 15.93 -> 13.85 ms (-2.08 ms; the byte-count estimate in patch/estimate.py was 2.0 ms); tokens/cycle 3.09 -> 3.05 because acceptance fell 1.5 points against the same-session baseline (68.3% vs 69.8%; just below the 68.8-70.9% range of baseline runs), as expected since a draft can't propose a token outside the subset. 195 -> 220.0 tok/s (+12.8%, 5 reps 218-221.5; per-prompt +8.7% to +16.8% against the same-session baseline of 194.1).
**Losslessness:** on a deterministic MoE backend (CUTLASS, MTP4, short prompts) the patched server matched the unpatched one byte for byte on all 12 outputs x 2 reps (draft head alone, and draft head + L2 prefetch). A later long-context test (10 prompts of 4K-96K tokens: 5 code-review, 5 random-hex lists; draft head + L2 prefetch, CUTLASS) was inconclusive: with controls (each server run twice) the unpatched baseline did not match itself either (0/10 between two reps on the same server), the patched server did not match itself (0/10), and baseline vs patched was 0/10 in both reps. A test whose control fails as often as the comparison shows nothing, so long-context losslessness remains unproven (the cause of the long-context run-to-run differences was not identified).
**The catch is coverage.** The speedup depends on how often the target's token is inside the subset: code 95-100%, English essays 85-87%, Chinese only 56%/22%. On the two Chinese held-out prompts the full stack was 18-34% SLOWER: tokens per cycle fell from 2.9/2.2 to 2.0/1.2
(draft acceptance 64%/40% -> 25%/6%), far more than the ~16% shorter cycle. A multilingual-safe subset (126,868 rows, padded to 126,912: the 49K code vocabulary + 24K Han, 14K Cyrillic, 8K Arabic, all Hangul/kana/Thai/Devanagari/Hebrew, 32K Latin by token id,
plus tokens seen in the target's answers to 176 prompts in 21 languages) makes the head ~0.43 ms per step (computed from bytes, not profiled) instead of 0.16, costing ~6% of speed on coding (+23% vs +31% over baseline) but improving 11 of 12 fresh held-out prompts
(mean tok/s +13.8%, per-prompt mean +12.8%). Head cost scales with rows, so vocabulary size is a direct speed/robustness dial.

## 3. MTP depth 4 (+2.5% on top)

A draft step used to cost ~1.17 ms (mostly the full head), so deeper drafting did not pay: untrimmed MTP4 measured 196.9 vs ~195. After trimming, draft steps cost ~0.46 ms (the first draft step, which runs over all k+1 positions, ~0.71 ms; these are final-stack profile figures).
Per-position acceptance (accepted-prefix counts divided by all cycles, 5-rep MTP4 run) was 0.81/0.66/0.53/0.41: the 4th guess is accepted in ~41% of all cycles, which is ~78% of the cycles in which the first three were accepted, so it adds up to ~0.4 tokens per cycle (measured +0.36: 3.05 -> 3.41)
for +1.26 ms of cycle (the extra draft step plus verifying 5 instead of 4 tokens, which touches more MoE experts): 3.41 tokens / 15.11 ms = 225.6 tok/s (+15.6%). Deeper is not better: MTP6 in the full stack added +0.54 tokens/cycle (+16%) but the cycle grew +17%
(~1.1 ms per extra step, more than the 0.46 ms draft step), giving 253.6 vs 256.4 at MTP4. MTP5 fails an assertion in this configuration (the sparse-attention ring capacity must divide the attention block size).

## 4. vLLM's skinny-GEMM kernels, switched on for SM120 (+10.4% on top)

**Why the dense layers were slow:** attention projections, shared experts, router and hyper-connection weights are BF16: ~7.4 GB per verify pass, ~6.1 ms of exclusive wall time, ~67% of peak bandwidth. In the Nsight profile (M=4) cuBLAS ran SM80-style wmma 16x16/32x32 tensor-op kernels,
several shapes split over K (split counts inferred from the launch grids) and five shapes added a separate reduction kernel. Extreme case: the DeltaNet `in_proj_ba` GEMM ([96, 2560], 0.49 MB) ran as 8 CTAs of one warp for 16.4 us (1.7% of peak) 36 times per cycle = 0.6 ms of latency.

**What I did:** vLLM 0.31 already ships a CuTe-DSL skinny GEMM and a model-level hook (`low_latency_gemm.py`), but the hook returns an empty plan unless the GPU is SM90/SM103 and uses TP=4 shapes. An import hook supplies a TP=1 plan for this GPU. The plan came from a micro-benchmark
that, for every layer shape and batch size, swept kernel configurations against cuBLAS (weights cycled to defeat caching, inside a CUDA graph), after calibrating the practical ceiling with a streaming-read kernel (1,615 GB/s = 90% of peak). vLLM's CuTe kernel is scalar FMA (SIMT) with FP32 accumulation,
8- or 16-byte weight loads, a K reduction inside each CTA (no cross-CTA split-K) and PDL; at M=4-5 the plan also routes HC-up, the main lm_head and (at M=5) shared-expert down to Triton tensor-core kernels. Isolated winners (M=4): qkvz 63 -> 44 us, o_proj 24 -> 18.6, QSA qkv 49 -> 36, in_proj_ba 15.9 -> 2.1, router 6.1 -> 3.4.

**Effect:** cycle 15.11 -> 13.80 ms (-1.31 ms), 225.6 -> 249.2 tok/s (+10.4%), tokens/cycle 3.41 -> 3.44 (acceptance unchanged). The isolated micro-benchmark promised 2.0-2.4 ms; the shortfall is explained by (measured) in-server CuTe kernels running 15-25% slower than benchmarked (qkvz 55 us vs 44 us, from the full-stack profile, which also has the prefetch and a lower clock), largely because the benchmark's weights were partly L2-resident (its qkvz and
sparse-attention timings exceeded DRAM peak, 104-106%), plus the hyper-connection down-projection (0.23 ms of the isolated saving at M=5) running through vLLM's fused kernel, not through the hook; in the final profile 96 of 97 HC-up calls per pass also stayed on cuBLAS (cause not verified).
**Numerics:** FP32 accumulation in a different order from cuBLAS changes bf16 rounding, so outputs diverge (0/12 byte-identical on CUTLASS), presumably at near-tie tokens (not analysed). Evidence of equivalence: Triton kernels' self-check matches cuBLAS's error; teacher-forced scoring under a reference server found the final stack's outputs
as probable as the baseline's (mean logprob -0.2397 vs -0.2425, paired +0.003 +/- 0.002); task-level quality of the final-multi preset vs baseline showed no detectable difference (CodeContests 135 tasks +0.86 pp, 95% CI -2.10 to +3.83; HumanEval+/MBPP+ -0.74 pp, CI -1.66 to +0.18; tool probe 95/96 vs 94/96).

## 5. L2 weight prefetch on a side stream (+2.9% on top)

**Why:** the 128 MiB L2 is large compared with the next big matrix a layer needs (out_proj 31.5 MB). Between a layer's big input GEMM and its out_proj there is a ~26 us (GDN layers; ~37 us for sparse-attention layers) run of small latency-bound kernels while DRAM is mostly idle. A 48-CTA kernel on an auxiliary stream touches
the upcoming weights (out_proj, the MLP hyper-connection matrices, the router: 47.5 MB per layer) with evict-last loads, so the big GEMM reads from L2. It only reads weights, so it cannot change outputs (byte-identical on CUTLASS). The deployed config streams ~1.1 TB/s standalone and ~0.95 TB/s inside the graph.
**Evidence:** in a synthetic 8-layer test out_proj dropped 26.6 -> 14.3 us (timer resolution ~2 us); in the real profile out_proj runs at 11.5 us (2.7 TB/s, above DRAM peak, i.e. L2-served).
**Effect:** alone, cycle 15.07 -> 14.70 ms (+2.5% vs the 3-rep draft-only MTP4 run at 228.7 tok/s, +3.9% vs the 5-rep run at 225.6; 234.5 tok/s); on top of the skinny GEMMs 13.80 -> 13.27 ms (+2.9%, 249.2 -> 256.4). Both are below the micro-benchmark's ~0.64 ms for the deployed config. Why is not established. With the skinny GEMMs the DeltaNet window is 26 us, shorter than the ~35 us baseline window the design assumed and the 41-45 us of the synthetic test; the sparse-attention window (37 us) is not shorter. But the gain
was larger with the skinny GEMMs (0.53 vs 0.37-0.41 ms), so window length does not explain the shortfall. Inside the graph the prefetch kernel itself takes ~50 us per layer (47.5 MB at ~0.95 TB/s), longer than the 26 us window. Hypotheses: contention with neighbouring kernels for DRAM bandwidth, and power throttling (SM clock ~2.2 GHz of 3.09 max;
the throttle flag was set more often in the prefetch runs, 98-99% of busy samples vs 65-87% in the draft-only and skinny-only runs).

## 6. Why Marlin for the MoE (what did NOT help)

The 512-expert NVFP4 MoE stores 4-bit weights. Marlin keeps activations in BF16 and dequantizes weights inside the kernel (weight-only W4A16, per vLLM's own warning text); vLLM's native FP4 backends use FP4 tensor-core instructions, which by default also quantize activations (W4A4; b12x has an optional A16 mode, not tried). Measured end-to-end:
flashinfer_cutlass 173.4 (run at 0.90 memory utilization; it OOMs at 0.95), vLLM cutlass 158.0, b12x 147.7 tok/s against Marlin ~195. Why they are slower is a hypothesis: at 4-5 tokens the work is bandwidth-bound so FP4 tensor cores have little to win while activation quantization and tile
shapes add cost; no native backend was profiled, and b12x's A16 mode (`VLLM_B12X_MOE_FP4_FORCE_A16`) was not tried. Marlin MoE took ~2.7 ms of kernel time per cycle in the baseline verify graph (2.6 ms exclusive wall time); it is 3.2 ms of kernel time for 5 tokens in the final stack (+18% like for like).

## 7. How the pieces combine (percentages vs 195.1; one run named per row)

| stack (single request, coding workload) | tok/s | cycle | gain |
|---|---:|---:|---:|
| baseline Marlin MTP3 | 195.1 | 15.9 ms | - |
| + trimmed draft head (49K), 5 reps | 220.0 | 13.85 ms | +12.8% |
| + MTP4, 5 reps | 225.6 | 15.11 ms | +15.6% |
| + skinny GEMMs, 5 reps | 249.2 | 13.80 ms | +27.7% |
| + L2 prefetch, 5 reps | 256.4 | 13.27 ms | +31.4% |

Gains don't simply add: MTP4 lengthens the cycle while raising tokens per cycle, and the kernel wins overlap. Final-stack profile (2 requests, 169 cycles; profiled cycle 13.7 ms vs 13.3 ms unprofiled): verify graph 10.6 ms (Marlin kernel time 3.2 ms for 5 tokens, skinny-GEMM kernel time 3.9 ms), four draft steps 2.1 ms (0.71 + 3 x 0.46),
main lm_head 0.78 ms, GPU idle ~0.24 ms (1.8%, all in gaps under 10 us). That last number bounds anything a faster CPU-side orchestrator could recover at one request (the baseline's equivalent was ~0.6 ms, 3.6%). (The power cap: the software power-cap throttle flag is set in ~97% of busy samples in the baseline and final runs, 65-87% in the draft-only and skinny-only runs; SM clock ~2.2 GHz of 3.09 max; this does not
by itself show each win is capped.)

## 8. What was tried and dropped

* Fused MoE router: a single-CTA Triton kernel that fuses top-k + expert align/sort (the router GEMM stays cuBLAS) was bit-exact on the GPU but took 203 us per call vs 8.5 us for the three stock C++ kernels it replaced (285 us when the router GEMV was fused in too).
* Megakernel: a survey agent's rough estimate (not measured) was 16-28 engineer-weeks for 1.3-2.5 ms (the survey also reported no public megakernel covering this architecture; the survey output was not archived); separately, the baseline verify graph is only ~3.6% idle between kernels, which is why the payoff is small. The L2 prefetch and skinny kernels are the practical slices.
* MTP5/6 and the native MoE backends, as above. Not tried: torch.compile / compilation modes other than the deployment's mode 0, FP8 KV cache, quantizing the dense BF16 layers (which would change numerics).

## 9. Gains by situation, and limits

Single request: +31% (code vocabulary) or +23% (multilingual). With 4 concurrent short requests the patches add +7% over the already-fixed baseline (likely because the 20-token verify batch is outside the skinny plan, whose M keys are 1-5, 10 and 15, and more experts are touched; not profiled at C>1); the capture-size fix is the larger win
there (3.5-4.2x). Context: served at 131,072 tokens (model configured for 262,144); prefill is ~9K tokens/s (32K cold prompt 3.66 s to first token, 65K 7.39 s; swift-local context benchmark, MTP off); decode speed looks nearly flat with context (inferred for this stack from the long-context totals: ~5-8% slower at 64-96K). This is an optimized vLLM-based stack: vLLM supplies the scheduler, API, KV cache, graph capture and MTP.
It is not a standalone engine, and SGLang was never compared under a controlled configuration.
