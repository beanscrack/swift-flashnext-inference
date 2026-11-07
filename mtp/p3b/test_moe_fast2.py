"""Equivalence + speed of the fast MoE switch (moe_fast.set_fast) on real train samples.

orig vs orig (GPU nondeterminism floor) and orig vs fast: loss and per-parameter gradient relative diffs,
plus forward+backward time per sample."""
import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3b")
sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
from common import load_base  # noqa: E402
from moe_fast import set_fast  # noqa: E402
from data import load_gen, load_sample  # noqa: E402
from ttt import ttt_forward, K_STEPS  # noqa: E402

model, lm_head, lm_rows, vocab = load_base()
model.requires_grad_(False)
named = [(n, p) for n, p in model.named_parameters() if n != "embed_tokens.weight"]
for _, p in named:
    p.requires_grad_(True)
inv = torch.full((lm_head.shape[0],), -1, dtype=torch.long, device="cuda")
inv[vocab] = torch.arange(vocab.numel(), device="cuda")
recs = [r for r in load_gen("train") if r["C"] >= 5][:2]


def run(fast, rec):
    set_fast(fast)
    for _, p in named:
        p.grad = None
    hid, x, labels, T, P, C = load_sample(rec["id"])
    anchors = torch.arange(P - 1, T - 5, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    S = ttt_forward(model, hid, x, anchors=anchors)
    ls = []
    for j in range(K_STEPS):
        tgt = inv[labels[anchors + 1 + j]]
        valid = tgt >= 0
        logits = F.linear(S[j], lm_rows).float()
        ls.append(F.cross_entropy(logits[valid], tgt[valid]))
    loss = torch.stack(ls).mean()
    loss.backward()
    torch.cuda.synchronize()
    return float(loss), time.time() - t0, T


def grads():
    return {n: p.grad.detach().clone() for n, p in named if p.grad is not None}


def reldiff(ga, gb):
    out = {}
    for n in ga:
        a, b = ga[n].float(), gb[n].float()
        out[n] = float((a - b).norm() / b.norm().clamp_min(1e-30))
    return out


res = {}
rec = recs[0]
run(False, rec)  # warm-up
l0, t0_, T = run(False, rec)
g0 = grads()
l1, t1_, _ = run(False, rec)
g1 = grads()
run(True, rec)  # warm-up fast
lf, tf, _ = run(True, rec)
gf = grads()
noise = reldiff(g1, g0)
diff = reldiff(gf, g0)
res.update({
    "T": T, "loss_orig": l0, "loss_orig_repeat": l1, "loss_fast": lf,
    "sec_orig": t0_, "sec_orig_repeat": t1_, "sec_fast": tf,
    "max_rel_grad_orig_vs_orig": max(noise.values()),
    "max_rel_grad_orig_vs_fast": max(diff.values()),
    "n_params_with_grad": len(g0),
    "worst_orig_vs_fast": sorted(diff.items(), key=lambda kv: -kv[1])[:4],
    "worst_orig_vs_orig": sorted(noise.items(), key=lambda kv: -kv[1])[:2],
    "median_rel_orig_vs_fast": sorted(diff.values())[len(diff) // 2],
})
del g0, g1, gf
print("TEST2", json.dumps(res), flush=True)
