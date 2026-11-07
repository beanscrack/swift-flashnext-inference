"""Check moe_fast.grouped_moe_forward_fast against p3 grouped_moe.grouped_moe_forward on real train samples:
same loss, same parameter gradients (bf16 compute), and per-sample forward+backward time."""
import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3b")
sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
import ttt  # noqa: E402
from common import load_base  # noqa: E402
from grouped_moe import grouped_moe_forward  # noqa: E402
from moe_fast import grouped_moe_forward_fast  # noqa: E402
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


def run(fn, rec):
    ttt.grouped_moe_forward = fn
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


out = {}
rec = recs[0]
run(grouped_moe_forward, rec)  # warm-up
l_ref, t_ref, T = run(grouped_moe_forward, rec)
ref_grads = {n: p.grad.detach().to(torch.bfloat16).clone() for n, p in named if p.grad is not None}
l_fast, t_fast, _ = run(grouped_moe_forward_fast, rec)
l_fast2, t_fast2, _ = run(grouped_moe_forward_fast, rec)
rel = {}
for n, p in named:
    if n in ref_grads and p.grad is not None:
        a = p.grad.detach().float()
        b = ref_grads[n].float()
        rel[n] = float((a - b).norm() / b.norm().clamp_min(1e-30))
out = {
    "T": T, "loss_ref": l_ref, "loss_fast": l_fast, "loss_fast_repeat": l_fast2,
    "sec_ref": t_ref, "sec_fast": t_fast, "sec_fast_repeat": t_fast2,
    "max_rel_grad_diff": max(rel.values()), "n_params_compared": len(rel),
    "n_params_with_grad": sum(1 for _, p in named if p.grad is not None),
    "worst": sorted(rel.items(), key=lambda kv: -kv[1])[:3],
    "peak_GiB": torch.cuda.max_memory_allocated() / 2**30,
}
print("TEST", json.dumps(out), flush=True)
