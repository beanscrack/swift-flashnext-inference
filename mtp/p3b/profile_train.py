"""Profile one training sample (forward + backward) with the fast MoE variant."""
import sys
import time

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3b")
sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
import ttt  # noqa: E402
from common import load_base  # noqa: E402
from moe_fast import grouped_moe_forward_fast  # noqa: E402
from data import load_gen, load_sample  # noqa: E402
from ttt import ttt_forward, K_STEPS  # noqa: E402

ttt.grouped_moe_forward = grouped_moe_forward_fast
model, lm_head, lm_rows, vocab = load_base()
model.requires_grad_(False)
named = [(n, p) for n, p in model.named_parameters() if n != "embed_tokens.weight"]
for _, p in named:
    p.requires_grad_(True)
inv = torch.full((lm_head.shape[0],), -1, dtype=torch.long, device="cuda")
inv[vocab] = torch.arange(vocab.numel(), device="cuda")
rec = [r for r in load_gen("train") if r["C"] >= 5][0]


def step(timing):
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
    torch.cuda.synchronize()
    t1 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t2 = time.time()
    for _, p in named:
        p.grad = None
    return T, t1 - t0, t2 - t1


print("warm", step(True), flush=True)
print("warm2", step(True), flush=True)
print("timed", step(True), flush=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    step(True)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25), flush=True)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15), flush=True)
print("PROFILE_DONE", flush=True)
