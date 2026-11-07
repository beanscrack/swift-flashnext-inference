"""VERIFY the TTT forward against a slow sequential reference (vLLM-like, real KV).

For 3 heldout sequences and 20 random anchors each:
  reference(e): step 0 = MTPDraft.forward over prefix rows 0..e (input x_{p+1}), kv kept;
                steps j=1..3 = MTPDraft.forward one row at a time (token x_{e+1+j}, position e+j,
                hidden = previous step's multi_hidden_out, kv appended).
  TTT: ttt.ttt_forward over all anchors of the sequence at once.
Criteria: max over (anchor, step) of ||a-b||_2/||b||_2 on sample_hidden < 2e-2, and identical
argmax over the 49,152 draft-vocab rows (BF16 lm_head rows) for >= 99% of (anchor, step).
"""
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp")
sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
from mtp_torch import MTPDraft  # noqa: E402
from ttt import ttt_forward, K_STEPS  # noqa: E402
from data import load_gen, load_sample, TAIL, CONFIG, VOCAB  # noqa: E402

OUT = "/u701/j692wu/swift-opt/mtp/p3/verify_ttt_bf16.json" if (sys.argv[1] if len(sys.argv) > 1 else "bf16") == "bf16" else "/u701/j692wu/swift-opt/mtp/p3/verify_ttt_fp32router.json"


MODE = sys.argv[1] if len(sys.argv) > 1 else "bf16"
if MODE == "fp32router":
    import router_patch
    router_patch.apply()
DT = torch.float32 if MODE == "fp32router" else torch.bfloat16


def load_model_and_head():
    model = MTPDraft.from_safetensors(TAIL, CONFIG, dtype=DT, device="cuda").eval()
    with safe_open(TAIL, framework="pt", device="cpu") as f:
        lm_head = f.get_tensor("lm_head.weight").to("cuda", torch.bfloat16)
    vocab = torch.from_numpy(np.load(VOCAB).astype(np.int64)).cuda()
    lm_rows = lm_head[vocab].contiguous()  # [49152, 2560] BF16 rows at draft-vocab ids
    return model, lm_rows, vocab


@torch.no_grad()
def reference_anchor(model, multi, x, e):
    """Sequential, vLLM-like chain for one anchor e.  Returns list of K_STEPS [H] sample vectors."""
    ids = x[1 : e + 2]  # input token of row p is x_{p+1}, rows 0..e
    pos = torch.arange(e + 1, device=x.device)[None]
    s0, m0, kv = model(multi[None, : e + 1], ids[None], pos, kv=None)
    outs = [s0[0, e]]
    carried = m0[:, e : e + 1]  # [1,1,D]
    for j in range(1, K_STEPS):
        tok = x[e + 1 + j][None, None]
        p = torch.tensor([[e + j]], device=x.device)
        s, m, kv = model(carried, tok, p, kv=kv)
        outs.append(s[0, 0])
        carried = m
    return outs


def main():
    torch.manual_seed(0)
    model, lm_rows, vocab = load_model_and_head()
    held = [r for r in load_gen("heldout")]
    rng = random.Random(20261107)
    # three heldout sequences of moderate length (prompt+completion) chosen at random
    cand = [r for r in held if r["P"] + r["C"] >= 400]
    seqs = rng.sample(cand, 3)

    per_seq = []
    all_rel = []
    all_match = []
    t_ttt = []
    t_ref = []
    for rec in seqs:
        hid, x, labels, T, P, C = load_sample(rec["id"])
        hid = hid.to(DT)
        assert T == rec["P"] + rec["C"]
        assert x[:P].tolist() == rec["prompt_token_ids"] and x[P:].tolist() == rec["completion_token_ids"]
        anchors = sorted(rng.sample(range(0, T - 5), 20))  # e in [0, T-6]
        anchors_t = torch.tensor(anchors, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        S = ttt_forward(model, hid, x, anchors=None)  # all anchors of the sequence
        torch.cuda.synchronize()
        t_ttt.append(time.time() - t0)
        # map anchor -> row index in the all-anchor output (anchors are 0..T-6)
        S_sel = S[:, anchors_t]  # [K, 20, H]
        rel_rows = []
        match_rows = []
        t0 = time.time()
        for i, e in enumerate(anchors):
            ref = reference_anchor(model, hid, x, e)
            torch.cuda.synchronize()
            for j in range(K_STEPS):
                a = S_sel[j, i].float()
                b = ref[j].float()
                rel = ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
                la = F.linear(S_sel[j, i : i + 1].to(lm_rows.dtype), lm_rows)
                lb = F.linear(ref[j][None].to(lm_rows.dtype), lm_rows)
                ia = int(la.argmax(-1))
                ib = int(lb.argmax(-1))
                rel_rows.append({"e": e, "j": j, "rel": rel})
                match_rows.append({"e": e, "j": j, "same_argmax": ia == ib})
        t_ref.append(time.time() - t0)
        rels = [r["rel"] for r in rel_rows]
        matches = [m["same_argmax"] for m in match_rows]
        all_rel += rels
        all_match += matches
        per_seq.append({
            "id": rec["id"], "mode": rec["mode"], "T": T, "P": P, "anchors": anchors,
            "max_rel": max(rels), "median_rel": float(np.median(rels)),
            "argmax_match": float(np.mean(matches)), "n_pairs": len(matches),
        })
        print(json.dumps(per_seq[-1]), flush=True)

    max_rel = max(all_rel)
    med_rel = float(np.median(all_rel))
    match = float(np.mean(all_match))
    passed = (max_rel < 2e-2) and (match >= 0.99)
    res = {
        "n_seqs": len(seqs), "anchors_per_seq": 20, "n_pairs": len(all_rel),
        "max_rel_err_sample_hidden": max_rel, "median_rel_err": med_rel,
        "argmax_identical_fraction": match, "criteria": {"max_rel<2e-2": max_rel < 2e-2, "argmax>=0.99": match >= 0.99},
        "PASS": bool(passed), "per_seq": per_seq,
        "ttt_seconds_per_seq": t_ttt, "reference_seconds_per_seq": t_ref,
        "note": "both paths teacher-forced with ground-truth tokens; bf16 weights, fp32 attention math",
    }
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print("RESULT", json.dumps({k: v for k, v in res.items() if k != "per_seq"}), flush=True)


if __name__ == "__main__":
    main()
