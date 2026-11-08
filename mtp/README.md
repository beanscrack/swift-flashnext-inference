# MTP draft-head retraining: code and evidence

Code, logs and verification outputs for retraining Swift's 1-layer MTP draft head (2026-11-07). The plan, gates and
full progress log are in `../docs/MTP_RETRAIN_PLAN.md`; the A/B run logs are in `../results/mtp_runs/`.
Paths inside the scripts point at the cluster (`/u701/j692wu/swift-opt/mtp/...`, hidden states in gpu-1-vm
`/tmp/mtp_hidden`). Weights, captured hidden states and the generated corpus (`p2/gen.jsonl`, 30 MB) are not in git.

## Result

| | original draft | retrained (`view-mtp-v1`) |
|---|---:|---:|
| offline heldout acceptance, positions 1-4 (simulator) | .812/.655/.531/.431 | .862/.731/.617/.526 |
| offline tokens/cycle | 3.43 | 3.74 |
| online coding tok/s, full FP8 stack (13 reps each) | 303.5 | 316.2 |
| online tokens/cycle (coding bench) | 3.43 | 3.58 |
| target model | - | byte-identical (lossless) |

## What the draft head is

- One QSA decoder layer (gated attention with an indexer, budget 2048) plus a 512-expert top-10 MoE.
- 31 BF16 tensors (2.61 B params), all stored in the checkpoint's `model-tail.safetensors`.
- Input: the target's pre-final-mixer 4-stream hidden state `[T, 10240]` and the next token's embedding.
- Output goes through the target's (trimmed, 49,152-row) `lm_head`, which it shares with the target.
- vLLM runs it recursively for k=4 draft tokens per cycle.

## How it was trained

- **Self-distillation.** The labels are the target's own greedy argmax. 4,186 coding prompts were generated
  by the served model with thinking on and off (3,982 train, 204 heldout).
- **Data.** For every token the hidden state was captured from the production V2 runner, 6.39 M rows in total.
- **TTT (EAGLE-3 style).** All 4 recursive draft steps are unrolled in parallel with an explicit attention
  mask, so training sees its own step-1..3 errors, as serving does.
- **Loss.** Cross-entropy over the draft vocab at all 4 steps.
- **Trainable set.** All 31 tensors are trainable; `embed_tokens` and `lm_head` stay frozen.
- **Optimizer.** AdamW 2e-5, 50 warmup steps, cosine decay, grad-accum 8, 2 epochs (996 steps, 2h53m on one
  RTX PRO 6000).

## Phases and files

| phase | what | files |
|---|---|---|
| P0 | Checkpoint "views": a directory of symlinks to the original checkpoint plus a rebuilt tail shard carrying replacement `mtp.*` tensors (raw bytes, everything else byte-identical). The identity view reproduces baseline speed. | `make_view.py`, `extract_mtp.py`, `verify_view.py`, `stlib.py`, `p0_real.log` |
| P1 | Standalone PyTorch re-implementation of the draft (`MTPDraft`) and parity against vLLM V2 forward calls captured by an import hook (median cosine 0.9995). | `mtp_torch.py`, `test_mtp_torch.py`, `swift_mtp_capture.py`, `sitecustomize.py`, `parity_v2.py`, `row_stats.py`, `p1_parity_v2.sh`; older V1 attempt `parity_check.py`, `p1_parity.sh`; independent reviews `verify_p1/`, `verify_p1b/`; residual investigation `residual/` |
| P2a | Corpus generation from the served model, leak check against eval sets, determinism and token accounting. | `p2/scripts/*.py`, `p2/*.json`, `p2/smoke/` |
| P2b | Hidden-state capture of the target for every corpus token (import hook into the V2 runner; FP8 rows + scales) and a rerun-consistency check. | `p2b/swift_hidden_capture.py`, `p2b/capture_client.py`, `p2b/rerun_rows.py`, `p2b/verify_rerun.json` |
| P3a | TTT forward (`ttt.py`), grouped MoE, offline acceptance simulator (`simulate.py`, round-end anchors = what live drafts see), second independent simulator (`sim_check.py`), and the gate: simulator vs ONLINE acceptance on the same prompts, max gap 0.008. | `p3/*.py`, `p3/online/online_gate.json`, `p3/logs/` |
| P3b | Training, export of the 31 tensors, verification (frozen hashes, export reload, leakage asserts, sequential cross-check), then an independent verifier using only the p3 code. | `p3b/train.py`, `p3b/common.py`, `p3b/moe_fast.py` (training-only MoE with cheaper backward), `p3b/logs/`, `p3b/eval/`, `p3b/verify/`, `p3b/verify_indep/` |
| P4 | View `view-mtp-v1` built and checked, online A/B on gpu-1-vm (identity vs mtp-v1, back to back: coding, OOD, concurrency 1/2/4/8, logprob score, output comparison). | `p4/check_view.py`, `p4/p4_chain.sh`, `p4/analyze.py`, `p4/analysis.json` |
| P5 | Fresh verifier: re-derived P4 from raw files with its own script and re-ran both arms in reverse order (8 reps each). Pooled Welch p = 1.8e-6. | `p5/raw_ab.py`, `p5/partc.py`, `p5/raw_ab_*.json` |

## Reproducing

1. **Build a view for any 31-tensor file:**
   `python make_view.py --mtp <file> --out <dir under swift-opt/mtp>`, then `python verify_view.py <dir>`.
2. **Serve it:** set `SWIFT_CHECKPOINT_PATH=<view dir>` on any launch command (see `../docs/SERVING.md`).
3. **Offline evaluation:** `p3/simulate.py`, with `p3b/common.py` swapping in a checkpoint the way
   `p3b/verify_indep/run_sim_wrapper.py` does. It needs the captured hidden states on gpu-1-vm.

## Known limits and next steps

- **Smaller online gain.** The online gain (3.43 -> 3.58) is about half the offline heldout gain (-> 3.74). The
  speed bench is a different prompt mix, and serving uses the FP8 draft head while the simulator is BF16.
- **Indexer untrained.** The 3 indexer tensors receive no gradient: training uses dense attention, and the
  indexer only matters above 2,048 tokens of context. `k_norm.weight` also rounds back to its BF16 value.
  Long-context acceptance was not measured separately.
- **No separate test split.** The heldout set was also used to select the checkpoint. The pick was the last
  step of a near-monotone curve, so the bias is small.
- **Ideas for v2:**
  - a corpus closer to production traffic;
  - training through the FP8-quantized draft head;
  - sparse-attention-aware training for the indexer;
  - more epochs (acceptance was still rising at step 996).
