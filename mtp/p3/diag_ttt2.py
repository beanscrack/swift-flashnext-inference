"""Diagnostic 2: is the TTT-vs-reference gap numerical (router bf16 ties) or logical?
Patches the MoE router to fp32 in BOTH paths (diagnostic only), then compares per step."""
import json, random, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp")
sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
import mtp_torch
import grouped_moe as gm
from mtp_torch import MTPDraft
from ttt import ttt_forward, K_STEPS
from data import load_gen, load_sample, TAIL, CONFIG
from verify_ttt import reference_anchor

def moe_fp32_router(self, x):
    logits = F.linear(x, self.gate.weight).float()
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, self.top_k, dim=-1)
    if self.norm_topk_prob:
        w = w / w.sum(dim=-1, keepdim=True)
    routed = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    for e in range(self.experts.gate_up_proj.shape[0]):
        rows, slot = (idx == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        xe = x[rows]
        gu = F.linear(xe, self.experts.gate_up_proj[e])
        g, u = gu.chunk(2, dim=-1)
        h = F.silu(g) * u
        y = F.linear(h, self.experts.down_proj[e]).float() * w[rows, slot][:, None]
        routed.index_add_(0, rows, y)
    routed = routed.to(x.dtype)
    s = self.shared_expert
    shared = s.down_proj(F.silu(s.gate_proj(x)) * s.up_proj(x))
    shared = torch.sigmoid(self.shared_expert_gate(x).float()).to(x.dtype) * shared
    return routed + shared

mode = sys.argv[1] if len(sys.argv) > 1 else "fp32router"
if mode == "fp32router":
    mtp_torch.MoEBlock.forward = moe_fp32_router
    gm.ROUTER_DTYPE = torch.float32

out = {"mode": mode}
rng = random.Random(7)
recs = sorted([r for r in load_gen("heldout") if 800 <= r["P"] + r["C"] <= 1000], key=lambda r: r["id"])
rec = recs[0]
for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
    model = MTPDraft.from_safetensors(TAIL, CONFIG, dtype=dt, device="cuda").eval()
    hid, x, labels, T, P, C = load_sample(rec["id"])
    hid = hid.to(dt)
    anchors = sorted(rng.sample(range(0, T - 5), 10))
    with torch.no_grad():
        S = ttt_forward(model, hid, x)
        errs = {j: [] for j in range(K_STEPS)}
        for e in anchors:
            ref = reference_anchor(model, hid, x, e)
            for j in range(K_STEPS):
                a = S[j, e].float(); b = ref[j].float()
                errs[j].append(((a - b).norm() / b.norm()).item())
    out[dt_name] = {f"step{j}": {"max": max(v), "median": sorted(v)[len(v)//2]} for j, v in errs.items()}
    print(dt_name, json.dumps(out[dt_name]), flush=True)
    del model; torch.cuda.empty_cache()
with open(f"/u701/j692wu/swift-opt/mtp/p3/diag_ttt2_{mode}.json", "w") as f:
    json.dump(out, f, indent=1)
