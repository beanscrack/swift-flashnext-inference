"""Verification (d): copy of p3/sim_check.py (independent sequential reference, no ttt.py) that can
load either the ORIGINAL weights or an exported checkpoint.  Same 20 heldout samples (sorted ids [:20]).

usage: python -I sim_check_ckpt.py <original|ckpt.safetensors> <out.json> <simulate_preds.npz>
"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp")
sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
from mtp_torch import MTPDraft  # noqa: E402
from data import load_gen, TAIL, CONFIG, VOCAB, HID  # noqa: E402

SRC, OUT, PREDS = sys.argv[1], sys.argv[2], sys.argv[3]
N_SAMPLES = 20
K = 4


def main():
    model = MTPDraft.from_safetensors(TAIL, CONFIG, dtype=torch.bfloat16, device="cuda").eval()
    if SRC != "original":
        with safe_open(SRC, framework="pt", device="cpu") as f:
            model.load_mtp_tensors({n: f.get_tensor(n) for n in f.keys()})
    with safe_open(TAIL, framework="pt", device="cpu") as f:
        W = f.get_tensor("lm_head.weight").to("cuda", torch.bfloat16)
    vocab_ids = torch.from_numpy(np.load(VOCAB).astype(np.int64)).cuda()
    rows = W[vocab_ids]
    ref_preds = np.load(PREDS)

    recs = sorted(load_gen("heldout"), key=lambda r: r["id"])[:N_SAMPLES]
    prefix_all, prefix_mode, round_all = [], {}, []
    agree_total, agree_n = 0, 0
    t0 = time.time()
    for rec in recs:
        d = torch.load(f"{HID}/{rec['id']}.pt", map_location="cpu", weights_only=False)
        hid = (d["fp8"].float() * d["scale"][:, None]).to("cuda", torch.bfloat16)
        x = d["input_ids"].long().cuda()
        lab = torch.load(f"{HID}/{rec['id']}.labels.pt", map_location="cpu", weights_only=False)
        labels = lab["target_argmax"].long().cuda()
        P = len(rec["prompt_token_ids"])
        C = len(rec["completion_token_ids"])
        T = P + C
        assert T == hid.shape[0] and labels.shape[0] == T - 1
        assert x.tolist() == rec["prompt_token_ids"] + rec["completion_token_ids"]

        pred = {}
        with torch.no_grad():
            for e in range(P - 1, T - 5):
                ids = x[1 : e + 2]
                pos = torch.arange(e + 1, device="cuda")[None]
                s0, m0, kv = model(hid[None, : e + 1], ids[None], pos, kv=None)
                hs = [s0[0, e]]
                carried = m0[:, e : e + 1]
                for j in range(1, K):
                    s, m, kv = model(carried, x[e + 1 + j][None, None], torch.tensor([[e + j]], device="cuda"), kv=kv)
                    hs.append(s[0, 0])
                    carried = m
                idx = F.linear(torch.stack(hs, 0), rows).argmax(dim=-1)
                pred[e] = vocab_ids[idx].cpu().numpy()
        sp = ref_preds[rec["id"]]
        seq = np.stack([pred[e] for e in range(P - 1, T - 5)], axis=1)
        agree = seq == sp
        agree_total += int(agree.sum())
        agree_n += agree.size

        labs = labels.cpu().numpy()
        for e in range(P - 1, T - 5):
            c = np.array([pred[e][j] == labs[e + 1 + j] for j in range(K)])
            pre = np.logical_and.accumulate(c)
            prefix_all.append(pre)
            prefix_mode.setdefault(rec["mode"], []).append(pre)
        e = P - 1
        while e <= T - 6:
            c = np.array([pred[e][j] == labs[e + 1 + j] for j in range(K)])
            pre = np.logical_and.accumulate(c)
            round_all.append(pre)
            e = e + int(pre.sum()) + 1
        print(rec["id"], rec["mode"], f"{time.time()-t0:.0f}s", flush=True)

    def stats(lst):
        a = np.array(lst, dtype=np.float64)
        acc = a.mean(axis=0)
        return {"n": int(len(a)), "acc": [float(v) for v in acc], "tokens_per_cycle": float(1 + acc.sum())}

    res = {
        "src": SRC, "n_samples": len(recs), "sample_ids": [r["id"] for r in recs],
        "overall": stats(prefix_all),
        "by_mode": {m: stats(v) for m, v in sorted(prefix_mode.items())},
        "round_end": stats(round_all),
        "agreement_with_simulate_py_argmax": agree_total / max(agree_n, 1),
        "seconds": time.time() - t0,
    }
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("RESULT", json.dumps({k: v for k, v in res.items() if k != "sample_ids"}), flush=True)


if __name__ == "__main__":
    main()
