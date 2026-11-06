# Swift-1.5-Qwen3.8-Flash-Next on one RTX PRO 6000 (gpu-8-vm): optimization results

All numbers are mean decode tokens/s of single requests unless stated. Benchmark = 6 saved coding prompts x 2 rounds per rep, 512 output tokens,
temperature 0, prefix cache reset per request, vLLM 0.31.0, MTP speculative decoding. **Thinking mode was off for every measurement in this document:** the prompts were built with the chat template's `enable_thinking=False`, 48-64 input tokens, EOS ignored; the quality runs set `enable_thinking: false` and the stress prompts use the same thinking-disabled template. Speed, quality and stability with thinking enabled (the template's default effort is `xhigh`; `medium` and `low` are also accepted, anything else returns HTTP 400) were not measured. Raw runs: `~/swift-profile/runs/*` (driver logs, per-request JSON,
GPU telemetry). Written 2026-10-22, revised after an independent fact-check (120 claims checked against the raw data; the corrections are folded in).
Nothing in `~/swift-local`, the shared conda env or vLLM's source was modified: every change is an env-gated import hook (`sitecustomize` on PYTHONPATH) or a launch flag.

Run-to-run spread matters for the small effects: the Marlin baseline measured 193.2-196.5 across six runs (5, 5, 5, 3, 3 and 1 reps; ~195 used below), and two runs of the same
draft-only MTP4 config differed by 1.4%. Effects of ~3% (MTP4 over MTP3, L2 prefetch) are only ~2x that spread.

## 1. Headline (single request, coding workload)

| config | tok/s | vs ~195 baseline |
|---|---:|---:|
| vLLM Marlin, MTP3 as deployed (193.7, 194.1, 196.5*, 196.1, 193.2**, 194.5 with 5, 5, 5, 3, 3, 1 reps; swift-local reported 197.5) | 193-197 | - |
| + draft lm_head trimmed to 49,152 rows (MTP3), 5 reps | 220.0 | +13% |
| + MTP4, 5 reps (a second 3-rep run: 228.7) | 225.6 | +16% |
| + vLLM's own CuTe skinny-GEMM path enabled on SM120, 5 reps | 249.2 | +28% |
| + side-stream L2 weight prefetch = "final-code", 5 reps | 256.4 | +31% |
| final-code, release config (wide capture sizes, plan_wide), 3 reps / 1 rep | 258.5 / 258.9 | +33% |
| "final-multi" (126,868-row multilingual draft vocab), 3 reps (original config) | 240.8 | +23% |
| final-multi, release config, 3 reps / 1 rep | 238.8 / 242.1 | +22-24% |

(*196.5 was run at `--gpu-memory-utilization 0.90`; **193.2 used the widened capture sizes.) Cycle time (verify + draft cycle): ~15.9 ms -> 13.3 ms (final-code) / 14.4 ms (final-multi).

## 2. Concurrency (the larger practical win for several agents)

With the original CUDA-graph capture sizes `[1,2,4]`, any batch above 4 verify tokens runs eagerly: two simultaneous requests ran at ~36 tok/s each
(71 tok/s aggregate, i.e. slower in total than one graphed request at ~196). Widening the capture sizes to cover (k+1) x C tokens fixes it with no kernels.
Aggregate output tok/s (mean of 2 reps) / mean per-request decode tok/s:

| config | C=1 | C=2 | C=3 | C=4 |
|---|---|---|---|---|
| baseline, original capture [1,2,4] | n/a | 71 / 36 | n/a | 137 / 35 |
| baseline, wide capture [1,2,3,4,8,12,16] (MTP3) | 185 / 196 | 297 / 162 | 393 / 142 | 478 / 127 |
| final-code, wide capture [1,2,3,4,5,10,15,20] (MTP4) | 236 / 253 | 358 / 192 | 446 / 160 | 511 / 137 |
| final-multi, wide capture | 220 / 237 | 332 / 180 | 425 / 153 | 494 / 132 |

The patches' relative gain shrinks with concurrency (final-code over the wide baseline: +27.5%, +20%, +14%, +7%). Two likely contributors, neither profiled at C>1: more experts are touched per step so
MoE weight reads dominate, and at C=4 under MTP4 the verify batch is M=20, which is not in the skinny plan (keys M=1,2,3,4,5,10,15) and falls back to cuBLAS, so the skinny-GEMM share of the gain (~+10% single-stream) is not applied there. To get only the concurrency fix on the existing deployment, change `--compilation-config` in `swift-local/serve.sh`
to use `"cudagraph_capture_sizes":[1,2,3,4,8,12,16]` (MTP3).

## 3. Vocabulary trade-off, with a fresh held-out set

The trimmed draft head can only propose tokens inside its vocabulary, so acceptance depends on coverage.

**Fresh held-out set (12 prompts written after all tuning and never used to build or choose anything; release configs; 3 reps; baseline = as deployed):**

| prompt | baseline | final-code | final-multi |
|---|---:|---:|---:|
| zh_microwave_principle | 157.2 | 111.9 (-29%) | 169.3 (+8%) |
| zh_balcony_gardening_guide | 157.8 | 125.3 (-21%) | 177.8 (+13%) |
| zh_tw_song_tea_culture | 129.1 | 90.7 (-30%) | 140.7 (+9%) |
| ja_git_internals | 168.6 | 121.3 (-28%) | 183.2 (+9%) |
| ko_puppy_adoption | 140.5 | 96.1 (-32%) | 161.2 (+15%) |
| ru_mendeleev_periodic_law | 166.5 | 90.8 (-45%) | 194.5 (+17%) |
| ar_tides_explained | 162.4 | 90.0 (-45%) | 186.2 (+15%) |
| de_eigenbedarf_kuendigung | 162.6 | 163.5 (+1%) | 166.1 (+2%) |
| fr_bande_dessinee_culture | 152.8 | 157.9 (+3%) | 149.8 (-2%) |
| en_harlem_renaissance_essay | 155.2 | 166.7 (+7%) | 175.4 (+13%) |
| en_java_parking_garage | 204.8 | 255.4 (+25%) | 256.0 (+25%) |
| en_sql_steps_windows | 222.8 | 326.5 (+47%) | 293.1 (+32%) |
| **mean of prompts** | 165.0 | 149.7 (**-9.3%**) | 187.8 (**+13.8%**) |

final-code improves only 5/12 prompts (worst -45%; German +0.5% is within noise); final-multi improves 11/12 (worst -2%, French, within noise).

* `final-code` (49,152 rows, built from code, docs and *other models'* code/reasoning outputs: NVIDIA teacher datasets and Qwen3.8-27B; no Swift output) is clearly fastest on code and structured prompts (the coding benchmark, SQL)
  but regresses every non-Latin-script prompt, and on English prose it is not reliably better than final-multi (fresh essay: 166.7 vs 175.4; first set: 182.6/200.6 vs 174.6/195.7). Use it only for code-dominated deployments.
* `final-multi` (126,868 rows) = the 49K code vocabulary + the first 24K Han / 14K Cyrillic / 8K Arabic tokens by token id + all Hangul/kana/Thai/Devanagari/Hebrew tokens + the byte-fragment tokens + every Latin-script token seen in the target's
  greedy answers (temperature 0, 400 tokens) to 176 calibration prompts in 21 languages (4,717 not already included) + the first 32K Latin tokens by id. As a result D covers 98% of the language-balanced calibration token mass (but only ~85% of its Han
  mass and ~95% of its Cyrillic mass). Rebuild with `vocab/make_preset_D.py` (verified byte-identical to the shipped file). Recommended default for mixed-language traffic.
* **Caveat on the first held-out set (8 prompts):** its baseline outputs were used to pick D's Han and Latin quotas (24K, 32K), so D's result there (+18.6% mean, all 8 prompts improved) is in-sample. The two sets' means are not comparable
  (set 1 has 3 code/structured prompts at +24-29%; the fresh set has 7 non-Latin prompts), so the gap between +18.6% and +13.8% mostly reflects workload mix; on like prompt types the in-sample advantage looks small (English essays +13/+12 vs +13; Chinese mean +12 vs +10).
  The fresh set is the clean number (+13.8%). Coverage of the held-out outputs by D: Chinese 95% / 91% (those two prompts informed the quota), English essays 93-94% (the 49K vocab: 85-87%), Spanish 86%, code 96-100%.
* This tokenizer has 54,531 Han tokens; with a by-id prefix about 24K gave ~90% coverage on the hardest held-out Chinese prompt. Covering all scripts this way takes about half the vocabulary.
* Still only 20 held-out prompts and 6 coding prompts in total. Rebuild a vocabulary from your own traffic with `vocab/*.py` and `calib/*` (including `calib/prompts.json`).

## 4. What did not work (and why)

* Native NVFP4 MoE backends are slower than Marlin on this GPU: flashinfer_cutlass 173.4 (at 0.90 utilization), vLLM cutlass 158.0, b12x 147.7 tok/s (Marlin ~195). `auto` selects flashinfer_cutlass; it OOMs in warmup at
  `--gpu-memory-utilization 0.95`. flashinfer_b12x OOMs in the memory-profiling run (allocating its 680 MiB workspace) at both 0.95 and 0.90 (tried only at those two settings) and has no valid result.
* The main lm_head runs at 89% of DRAM peak (measured). Marlin's DRAM efficiency could not be measured (ncu counters are blocked) and was assumed to be similar, so "MoE kernels are not the lever" rests on that assumption plus the native-backend results.
* MTP length (untrimmed): MTP2 178.8, MTP3 ~195, MTP4 196.9. With the trimmed head MTP4 is best (+2.5% over MTP3); MTP6 (253.6 in the full stack) is no better than MTP4; MTP5 is unsupported in this build
  (QSA ring capacity must divide the attention block size).
* Fused MoE router (Opus-built; bit-exact on canonical outputs on the GPU: 20,000/20,000 cases, graph replay equals eager): 203 us per call vs 8.5 us for the three stock kernels it replaced (Nsight), so it was shelved
  (`~/swift-profile/router`). The achievable ceiling was only ~0.4 ms (3%).
* **Megakernel:** a single persistent kernel for this 48-layer GDN + QSA + 512-expert MoE + hyper-connection + MTP model was judged not worth it. The survey agent's rough estimate was 16-28 engineer-weeks for 1.3-2.5 ms
  (an estimate, not a measurement), and the baseline profile shows the verify graph is only ~3.6% idle between kernels, so kernel boundaries are not the loss. The practical slices that were built are the L2 weight prefetch (+3%) and the skinny-GEMM/PDL enablement.
* Profile of the final stack (2 requests, 169 cycles, profiled cycle 13.7 ms vs 13.3 ms unprofiled): verify graph 10.6 ms (Marlin MoE 3.2 ms for 5 tokens), four draft steps ~2.1 ms, eager kernels ~1.2 ms busy (main lm_head 0.78 ms).
  The big kernels sit near their DRAM-bandwidth floors; what remains is the small-kernel chain.

## 5. Validation and its limits

* **Marlin is non-deterministic** at temperature 0 (consecutive reps of the same unpatched Marlin server: 0 or 1 of 12 outputs byte-identical), so token-for-token checks on Marlin are impossible. The **CUTLASS MoE backend is deterministic**
  (12/12 identical across reps and servers), so exactness was tested there (MTP4, 12 outputs of 512 tokens, 2 reps each):
  * draft-vocab patch alone: **12/12 byte-identical to the unpatched baseline** at short context (lossless in practice, as by construction; see the long-context caveat below);
  * draft patch + L2 prefetch: **12/12 byte-identical** (L2 prefetch only reads weights);
  * full stack (adds the skinny GEMMs): **0/12 identical**, diverging early (character 19 to 256). The skinny kernels change bf16 rounding in the target, which presumably flips near-tie tokens (no logit-gap analysis was done at the divergence points).
  Caveats: the "12" outputs are 6 distinct prompts x 2 identical rounds, and this test used the 49K code vocabulary only; the multilingual vocabulary was not exactness-tested (it is lossless by the same construction).
  **Long-context result (controls completed 2026-10-23 00:19 UTC): inconclusive.** The 10 fixed prompts (4K-96K tokens, temperature 0, 1,024 tokens) were run twice on each of two fresh CUTLASS servers (unpatched baseline; draft head + L2 prefetch with the 49K code vocabulary). The unpatched baseline did not reproduce itself: 0/10 byte-identical between its two reps on the same server, and 0/10 against its earlier run (`longctx/lc-cutlass-base`). The patched server was likewise 0/10 against itself, and baseline vs patched was 0/10 in both reps. A test whose control fails as often as the comparison can neither show nor rule out a long-context difference. The cause was not identified; untested candidates are that the second rep hits the prefix cache (Mamba cache mode 'align', 2,048-token scheduled chunks) and so takes a different prefill path, or that the CUTLASS path is not run-to-run deterministic at long context although it was on short prompts. **Long-context losslessness therefore remains unproven.** The stability runs in section 5b (needles to 120K, copy canaries) show no corruption but are not an exactness test. A teacher-forced scoring of the long outputs under a reference server (as done for short prompts) was not run.
* **Quality of the full stack (no detectable loss in reference likelihood):** 156 baseline outputs and 156 final-stack outputs (same prompts; first 256 tokens each, re-tokenized from text, 39,936 tokens per set) were teacher-force scored
  under the deterministic CUTLASS reference server (a different NVFP4 kernel path; its argmax disagrees with the baseline's own tokens on ~3.4% of positions). Baseline outputs: mean logprob -0.2425, rank-1 fraction 96.63%; final-stack outputs: -0.2397, 96.84%.
  Paired by prompt and round, the mean per-sequence difference is +0.003 +/- 0.002 (s.e., n=52 prompt-round pairs from 26 prompts; 95% CI about -0.001 to +0.007 nats/token); per workload: coding -0.004 +/- 0.004, held-out set 1 +0.003 +/- 0.004, fresh set +0.006 +/- 0.003.
  Set B mixes presets: coding outputs from final-code, set 1 from final-multi, fresh set from final-multi. Limits: greedy outputs only, 26 prompts, 256 tokens each, prefill-path scoring (it measures the outputs, not the decode kernels), and likelihood under the same model checks
  consistency, not task quality.
* Skinny GEMM kernel checks: in-server self-check of the Triton kernels against cuBLAS and fp32 shows the same error as cuBLAS (ratio 1.00); the CuTe kernels (most of the gain) were checked against an fp32 reference in the micro-benchmark.
  Per-position draft acceptance over thousands of steps, p0..p3: baseline-draft-only 0.818/0.671/0.534/0.423 and 0.810/0.661/0.527/0.412 (two runs); skinny 0.817/0.668/0.531/0.422; full stack 0.814/0.659/0.521/0.410; final-code (3-rep run) 0.811/0.660/0.521/0.406.
  So acceptance agrees within run-to-run noise (~1 point); this test cannot resolve smaller shifts.
* Identity control (a 248,256-row "trimmed" vocabulary through the patch, 3 reps): 196.7 tok/s, acceptance 71.1% (baseline runs: 68.8-70.9%), consistent with no effect; it is not itself an exactness proof (the CUTLASS test above is).
* The draft-vocab patch was code-reviewed against the vLLM 0.31 call sites by an independent Opus review (no result-changing bug found).
* Hook behavior: the draft-vocab and skinny hooks are strict by default (they abort at start-up if they cannot arm, `SWIFT_DRAFT_VOCAB_STRICT` / `SWIFT_SKINNY_STRICT`); **the L2-prefetch hook has no strict mode** and can degrade to "model left unmodified" or
  "graph captured WITHOUT prefetch", and `sitecustomize.py` swallows install errors. Always check the server log for `[swift-draft-vocab] ACTIVE`, `[swift_skinny] patched` and `[swift_l2pf] armed`.
* The GPU is power-capped: the software power-cap throttle flag was set in ~97% of busy samples for the baseline and final runs (mean draw ~285-293 W of 300 W; instantaneous draw was >= 299 W in ~60% of busy samples, 50-73% per run), so measured gains already include clock effects.
* Patches depend on vLLM 0.31.0 internals (import hooks on `Qwen4Exp*` modules). The skinny plan was tuned on this GPU only; M=20 verify batches fall back to cuBLAS. L2 prefetch needs compilation mode 0 (as used here).
* The release launcher was smoke-tested end to end (`PRESET=final-multi`): started in 682 s, served requests at 213-265 tok/s, all three hooks armed, no tracebacks, a correct chat answer, clean shutdown. The released wide-capture configs were also run on the fresh held-out set and the coding benchmark.

## 5b. Task-level quality and deployment stability (added 2026-10-23)

### Task-level quality: final-multi vs the stock-Marlin baseline (baseline-wide)

Single-turn code tasks plus tool-call checks, served by vLLM with the Marlin MoE backend; both configurations went through the same session script
(`quality/quality_session.sh`). This is **not** an agentic evaluation (no Terminal-Bench / repo-fixing). Both runs used thinking disabled (`enable_thinking: false`; CodeContests at temperature 0.7 with a 4,096-token budget, HumanEval+/MBPP+ greedy with 2,048 tokens), so these results say nothing about quality in thinking mode. Scored files: `quality/scored/`.

| check | baseline-wide | final-multi | paired difference (final - base) |
|---|---:|---:|---|
| LOCKED CodeContests set, 135 tasks x 6 samples (pass rate) | 75.9% | 76.8% | +0.86 pp, 95% CI -2.10 to +3.83, p = 0.62 (30 tasks improved / 24 regressed / 81 tied) |
| HumanEval+ and MBPP+, 542 tasks x 1 sample (pass@1) | 84.1% | 83.4% | -0.74 pp, 95% CI -1.66 to +0.18, p = 0.22 (1 improved / 5 regressed) |
| tool-calling probe (96 scenario checks) | 94/96 | 95/96 | n/a |

Reading: no detectable difference, but the intervals are wide (135 tasks resolve only effects of several points; the HumanEval+/MBPP+ point estimate is
negative and its interval allows a loss of up to about 1.7 pp). Truncation rates on CodeContests were 14.8% vs 14.6%. The final-multi server of this session
was left orphaned after its sampling, guard and tool probe had finished (the session script died on an NFS stale file handle); it was stopped by hand and the samples are unaffected.

### Stability: up to 8 concurrent sessions, prompts of 2K-120K tokens

Harness: `stress/` (`stress.py`, report in `stress/runs/<label>/stress/REPORT.txt`). Both configurations ran the same traffic once (about 39 minutes each, `--max-num-seqs 8`,
`--gpu-memory-utilization 0.95`, CUDA-graph capture sizes widened to cover 8 x (1+k) tokens): ramps at 1, 2, 4 and 8 sessions (4 min each), a 12-minute soak at 8, then 3-minute
stages at 1, 2 and 3 sessions with 25% of prompts at 120K. Prompt mix in the ramps and soak: 35% 2K, 25% 8K, 20% 32K, 12% 64K, 8% 120K; outputs 64-2048 tokens.

| | baseline-wide (stock Marlin MTP3) | final-multi (MTP4 + patches) |
|---|---:|---:|
| requests completed | 611 | 603 |
| errors / timeouts | 0 / 0 | 0 / 0 |
| Traceback / OOM / ERROR lines in server log | none | none |
| VRAM during run | flat at 93,733 MiB | flat at 93,757 MiB |
| KV pool (tokens) | 341,041 | 306,242 (-10%) |
| peak KV usage | 98.1% | 99.6% |
| preemptions (soak / all stages) | 5 / 8 | 72 / 92 |
| soak aggregate throughput at 8 sessions | 101.3 tok/s | 89.8 tok/s |
| median per-request decode, 1 session (mixed traffic) | 177.8 tok/s | 200.3 tok/s |
| p99 inter-token gap with 2+ sessions | ~200-215 ms | ~200-215 ms |
| known-answer canaries under load (temperature 0) | 48, none regressed | 46, none regressed |
| copy canaries / needle replays under load | 12/12 exact, 12/12 correct | 12/12 exact, 11/11 correct |
| needles inside the traffic: 32K / 64K / 120K | 14/14, 17/17, 11/11 | 14/14, 17/17, 11/11 |
| degeneration flags (soft = repetition) | 1 soft in 513 outputs | 0 in 508 outputs |
| GPU hardware throttle | 0% of samples | 0% of samples |

The harness verdict was **WARN for both**, for four reasons, none of them an error or a leak: (1) Triton kernels JIT-compiled for the first time during the run (8 and 9 kernels), each a one-time
latency spike, so a warm-up request set is advisable; (2) preemptions under KV pressure; (3) TTFT tails for 2K-8K prompts at 2-4 sessions, which queue behind other sessions' long prefills; (4) aggregate throughput at 8 sessions
below that at 4 (96 vs 126 tok/s baseline-wide, 104 vs 122 final-multi), with the KV pool nearly full because of the 120K prompts in the mix.

Reading: both stacks were stable in this run (no errors, no memory growth, all canaries and needles correct). final-multi's smaller KV pool shows up as about 14x more preemptions in the soak (72 vs 5) and about 11% lower
soak throughput; for 8 long sessions on one card prefer baseline-wide, or try a slightly higher `--gpu-memory-utilization` or a cap on long sessions (both untested). The ~200 ms inter-token gaps are other sessions'
chunked prefill and occur in both. Limits: one run per configuration, 39 minutes each (no multi-hour soak), Marlin backend only, and the canaries check correctness of specific answers rather than output exactness.

## 6. How to run

```
# PRESET = final-multi (recommended for mixed-language traffic) | final-code (English/code only) | baseline-wide (concurrency fix only)
SWIFT_CHECKPOINT_PATH=/tmp/swift-cache-j692wu-20261021 PRESET=final-multi bash ~/swift-profile/release/serve_final.sh
```

`release/` holds real copies of the hooks, both draft vocabularies, the skinny plan and the launcher (`serve_exp.sh` is the parameterized copy of `swift-local/serve.sh`). `release/provenance/` has copies of the preset-D build scripts (prefixed with their
source directory, e.g. `vocab_make_preset_D.py`), the calibration and fresh-held-out prompts and the benchmark/scoring scripts; the build scripts need their inputs, which live in `~/swift-profile/vocab` and `~/swift-profile/calib`, so run them from there
(the 49K code-vocabulary builders, `vocab/build_counts.py`, `model_counts.py`, `struct_counts.py` and `make_subsets.py`, are not copied; the shipped `.npy` files are the outputs).
Measured start-up with the node-local checkpoint copy was 10-17 min (typically ~11 min); an uncached start from the NFS snapshot was not measured. Use a node-local `TRITON_CACHE_DIR` (default `/tmp/triton_$USER`).
Benchmarks: `swift-local/experiment_bench.py` (single request), `~/swift-profile/bench_conc.py` (concurrency), `compare_exact.py` (token-exactness on CUTLASS), `score2/` (reference scoring); drivers `run_config7.sh` / `matrix7.sh`; summaries `summarize.py`.
