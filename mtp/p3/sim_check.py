"""Second, independent simulator: recomputes acceptance on 20 heldout samples WITHOUT ttt.py.

Every completion anchor e in [P-1, T-6] is evaluated with the sequential vLLM-style chain built only
from mtp_torch.MTPDraft.forward (step 0 over prefix rows 0..e, then steps 1..3 one row at a time with
the real KV).  Its own loader, own argmax, own label check, own round-end walk.  Then it is compared
with simulate.py (per-anchor argmax saved in sim_preds.npz) and its acc_j are reported.
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

OUT = "/u701/j692wu/swift-opt/mtp/p3/sim_check_bf16.json"
PREDS = "/u701/j692wu/swift-opt/mtp/p3/sim_preds_bf16.npz"
N_SAMPLES = 20
K = 4


def main():
    model = MTPDraft.from_safetensors(TAIL, CONFIG, dtype=torch.bfloat16, device="cuda").eval()
    with safe_open(TAIL, framework="pt", device="cpu") as f:
        W = f.get_tensor("lm_head.weight").to("cuda", torch.bfloat16)
    vocab_ids = torch.from_numpy(np.load(VOCAB).astype(np.int64)).cuda()
    rows = W[vocab_ids]
    ref_preds = np.load(PREDS)

    recs = sorted(load_gen("heldout"), key=lambda r: r["id"])[:N_SAMPLES]
    prefix_all, prefix_mode = [], {}
    round_all = []
    agree_total, agree_n, max_abs_diff_sample = 0, 0, []
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

        pred = {}  # e -> [4] token ids (from sequential reference)
        with torch.no_grad():
            for e in range(P - 1, T - 5):  # e in [P-1, T-6]
                ids = x[1 : e + 2]
                pos = torch.arange(e + 1, device="cuda")[None]
                s0, m0, kv = model(hid[None, : e + 1], ids[None], pos, kv=None)
                hs = [s0[0, e]]
                carried = m0[:, e : e + 1]
                for j in range(1, K):
                    s, m, kv = model(carried, x[e + 1 + j][None, None], torch.tensor([[e + j]], device="cuda"), kv=kv)
                    hs.append(s[0, 0])
                    carried = m
                H = torch.stack(hs, 0)  # [4, H]
                idx = F.linear(H, rows).argmax(dim=-1)
                pred[e] = vocab_ids[idx].cpu().numpy()
        # compare with simulate.py predictions (index e - (P-1))
        sp = ref_preds[rec["id"]]  # [4, n]
        seq = np.stack([pred[e] for e in range(P - 1, T - 5)], axis=1)  # [4, n]
        agree = (seq == sp)
        agree_total += int(agree.sum())
        agree_n += agree.size
        max_abs_diff_sample.append(float(1.0 - agree.mean()))

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
        print(rec["id"], rec["mode"], f"anchors={T-5-(P-1)}", f"{time.time()-t0:.0f}s", flush=True)

    def stats(lst):
        a = np.array(lst, dtype=np.float64)
        acc = a.mean(axis=0)
        return {"n": int(len(a)), "acc": [float(v) for v in acc], "tokens_per_cycle": float(1 + acc.sum())}

    res = {
        "n_samples": len(recs),
        "sample_ids": [r["id"] for r in recs],
        "overall": stats(prefix_all),
        "by_mode": {m: stats(v) for m, v in sorted(prefix_mode.items())},
        "round_end": stats(round_all),
        "agreement_with_simulate_py_argmax": agree_total / max(agree_n, 1),
        "agreement_values": agree_n,
        "per_sample_disagreement": max_abs_diff_sample,
        "seconds": time.time() - t0,
    }
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
