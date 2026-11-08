"""Offline acceptance simulator on the heldout split (204 samples), ORIGINAL draft weights.

Per sample (prompt length P, total T = P + C):
  anchors e in [P-1, T-6] (completion anchors whose labels target_argmax[e+1..e+4] exist).
  pred_j(e) = argmax over the 49,152 draft-vocab rows of BF16 lm_head rows @ sample_hidden_j(e)
  correct_j(e) = pred_j(e) == target_argmax[e+1+j]     (out-of-vocab labels never match)
  prefix_j(e) = all of correct_0..correct_j           -> acc_j = mean over anchors of prefix_j
  tokens/cycle = 1 + sum_j acc_j

Round-end variant (online anchors are verify-round ends): starting at e = P-1, each round accepts
a = number of leading correct steps (0..4) of the draft chain at the current anchor, then the next
anchor is e + a + 1 (the target's bonus token is the verified position).  acc_j is taken over the
round anchors only.  Uses the ORIGINAL draft's own acceptance to choose the anchors.
"""
import json
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
from ttt import ttt_forward, K_STEPS  # noqa: E402
from data import load_gen, load_sample  # noqa: E402
from verify_ttt import load_model_and_head  # noqa: E402

from verify_ttt import MODE, DT  # noqa: E402  (also applies the fp32 router patch when MODE=fp32router)

OUT = f"/u701/j692wu/swift-opt/mtp/p3/sim_results_{MODE}.json"
PREDS = f"/u701/j692wu/swift-opt/mtp/p3/sim_preds_{MODE}.npz"


def summarize(prefix_list, n_anchor_list=None):
    """prefix_list: [n_anchors, K] bool array (prefix correctness). Returns acc_j and tokens/cycle."""
    a = np.asarray(prefix_list, dtype=np.float64)
    acc = a.mean(axis=0) if len(a) else np.zeros(K_STEPS)
    return {
        "n": int(len(a)),
        "acc": [float(v) for v in acc],
        "tokens_per_cycle": float(1.0 + acc.sum()),
    }


def main():
    model, lm_rows, vocab = load_model_and_head()
    recs = load_gen("heldout")
    recs = sorted(recs, key=lambda r: r["id"])
    pooled = []          # per-anchor prefix rows
    pooled_mode = defaultdict(list)
    rounds = []          # round-end anchors prefix rows
    rounds_mode = defaultdict(list)
    per_sample = []
    preds_store = {}
    label_match = []     # sanity: target_argmax[P-1+k] == generated[k]
    t0 = time.time()
    for n_done, rec in enumerate(recs):
        hid, x, labels, T, P, C = load_sample(rec["id"])
        hid = hid.to(DT)
        assert T == P + C == rec["P"] + rec["C"]
        assert x[:P].tolist() == rec["prompt_token_ids"], rec["id"]
        assert x[P:].tolist() == rec["completion_token_ids"], rec["id"]
        comp = torch.tensor(rec["completion_token_ids"], device=labels.device)
        label_match.append(float((labels[P - 1 : P - 1 + C] == comp).float().mean()))
        with torch.no_grad():
            S = ttt_forward(model, hid, x)  # [4, T-5, H], anchors e = 0..T-6
            logits_idx = []
            for j in range(K_STEPS):
                lg = F.linear(S[j].to(lm_rows.dtype), lm_rows)  # [T-5, 49152] bf16
                logits_idx.append(lg.argmax(dim=-1))
            idx = torch.stack(logits_idx, 0)  # [4, T-5] draft-vocab index
            pred = vocab[idx]  # token ids
        pred_np = pred.cpu().numpy()
        labels_np = labels.cpu().numpy()
        n_anchor = T - 5
        e_all = np.arange(P - 1, T - 5)  # e in [P-1, T-6]
        corr = np.zeros((len(e_all), K_STEPS), dtype=bool)
        for j in range(K_STEPS):
            corr[:, j] = pred_np[j, e_all] == labels_np[e_all + 1 + j]
        prefix = np.logical_and.accumulate(corr, axis=1)
        pooled.append(prefix)
        pooled_mode[rec["mode"]].append(prefix)
        # round-end chain driven by the original draft's own acceptance
        e = P - 1
        round_prefix = []
        while e <= T - 6:
            c = corr_at(pred_np, labels_np, e)
            pre = np.logical_and.accumulate(c)
            round_prefix.append(pre)
            a = int(pre.sum())
            e = e + a + 1
        if round_prefix:
            rp = np.array(round_prefix)
            rounds.append(rp)
            rounds_mode[rec["mode"]].append(rp)
        preds_store[rec["id"]] = pred_np[:, P - 1 : T - 5].astype(np.int32)  # [4, n_anchors] token ids
        per_sample.append({
            "id": rec["id"], "mode": rec["mode"], "P": P, "C": C,
            "n_anchors": int(len(e_all)), "acc": summarize(prefix)["acc"],
        })
        if n_done % 20 == 0:
            print(f"{n_done+1}/{len(recs)} {time.time()-t0:.0f}s", flush=True)

    def block(rows_list):
        return summarize(np.concatenate(rows_list, axis=0)) if rows_list else None

    res = {
        "n_samples": len(recs),
        "anchor_rule": "e in [P-1, T-6]; labels target_argmax[e+1+j]",
        "overall": block(pooled),
        "by_mode": {m: block(v) for m, v in sorted(pooled_mode.items())},
        "round_end": block(rounds),
        "round_end_by_mode": {m: block(v) for m, v in sorted(rounds_mode.items())},
        "round_end_n_rounds": int(sum(len(r) for r in rounds)),
        "label_vs_generated_match_mean": float(np.mean(label_match)),
        "label_vs_generated_match_min": float(np.min(label_match)),
        "per_sample": per_sample,
        "seconds": time.time() - t0,
    }
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(PREDS, **preds_store)
    print("OVERALL", json.dumps(res["overall"]), flush=True)
    print("ROUND_END", json.dumps(res["round_end"]), flush=True)


def corr_at(pred_np, labels_np, e):
    return np.array([pred_np[j, e] == labels_np[e + 1 + j] for j in range(K_STEPS)], dtype=bool)


if __name__ == "__main__":
    main()
