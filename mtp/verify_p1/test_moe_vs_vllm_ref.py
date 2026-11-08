"""verify_p1 item 7: mtp_torch decoder-layer MoE vs an independent per-token reference.

The reference reads RAW tensors straight from mtp_original.safetensors (no mtp_torch loader)
and follows vLLM 0.31 semantics:
  router   : Qwen3NextSparseMoeBlock -> FusedTopKRouter (softmax over 512 logits, top-10,
             renormalize = norm_topk_prob=True), routed_scaling_factor = 1
  experts  : gate_up_proj[e] rows [0:640] = gate (w1), [640:1280] = up (w3)  (routed_experts.py:968-982)
             out = down[e] @ (silu(gate) * up)   (SiluAndMul: silu(x[:d]) * x[d:])
  shared   : Qwen2MoeMLP: down(silu(gate_proj x) * up_proj x) * sigmoid(shared_expert_gate x)
  combine  : routed + shared
Run: /u701/j692wu/.conda/envs/inference/bin/python -I -B test_moe_vs_vllm_ref.py  (CPU, fp32)
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(1234)
MTP = "/u701/j692wu/swift-opt/mtp"
sys.path.insert(0, MTP)
import mtp_torch as M  # noqa: E402

CK = os.path.join(open(os.path.expanduser("~/swift-local/model-path.txt")).read().strip(), "config.json")
WF = os.path.join(MTP, "mtp_original.safetensors")
PFX = "mtp.layers.0.mlp."

from safetensors import safe_open  # noqa: E402

with safe_open(WF, framework="pt", device="cpu") as f:
    gate_w = f.get_tensor(PFX + "gate.weight").float()                       # [512, 2560]
    gu = f.get_tensor(PFX + "experts.gate_up_proj").float()                   # [512, 1280, 2560]
    dn = f.get_tensor(PFX + "experts.down_proj").float()                      # [512, 2560, 640]
    sg = f.get_tensor(PFX + "shared_expert.gate_proj.weight").float()         # [640, 2560]
    su = f.get_tensor(PFX + "shared_expert.up_proj.weight").float()           # [640, 2560]
    sd = f.get_tensor(PFX + "shared_expert.down_proj.weight").float()         # [2560, 640]
    seg = f.get_tensor(PFX + "shared_expert_gate.weight").float()             # [1, 2560]

model = M.MTPDraft.from_safetensors(WF, CK, dtype=torch.float32, device="cpu")
moe = model.layers[0].mlp
moe.eval()
H = gate_w.shape[1]
TOPK = 10


def ref_token(x, router_bf16=False):
    logits = gate_w @ x
    if router_bf16:  # vLLM GateLinear tier 5 (F.linear in bf16, out_dtype unset) -> bf16 logits
        logits = (gate_w.bfloat16() @ x.bfloat16()).float() if False else F.linear(x.bfloat16(), gate_w.bfloat16()).float()
    p = torch.softmax(logits, dim=-1)
    topv, topi = torch.topk(p, TOPK)
    w = topv / topv.sum()
    margin = float(topv[-1] - p.topk(TOPK + 1).values[-1])  # gap between 10th and 11th prob
    out = torch.zeros(H)
    for wi, e in zip(w.tolist(), topi.tolist()):
        h2 = gu[e] @ x  # [1280]
        g, u = h2[:640], h2[640:]
        out += wi * (dn[e] @ (F.silu(g) * u))
    sh = F.silu(sg @ x) * (su @ x)
    out += torch.sigmoid(seg @ x)[0] * (sd @ sh)
    return out, sorted(topi.tolist()), margin


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


results = {}
for trial in range(2):
    x = torch.randn(4, H) * (1.0 + trial)
    with torch.no_grad():
        mine = moe(x)
    refs, refs_b, sets_ok, margins = [], [], 0, []
    for t in range(4):
        r, s, mg = ref_token(x[t])
        rb, sb, _ = ref_token(x[t], router_bf16=True)
        refs.append(r)
        refs_b.append(rb)
        margins.append(mg)
        sets_ok += int(s == sorted(torch.topk(torch.softmax(gate_w @ x[t], -1), TOPK).indices.tolist()))
    ref = torch.stack(refs)
    refb = torch.stack(refs_b)
    per_tok = [rel(mine[t], ref[t]) for t in range(4)]
    results[f"trial{trial}"] = {
        "max_rel_err_fp32_router_vs_ref": max(per_tok),
        "per_token_rel": per_tok,
        "max_rel_err_vs_bf16_router_ref": rel(mine, refb),
        "topk_set_match_fp32_router": sets_ok,
        "min_top10_vs_11_prob_margin": min(margins),
        "ref_output_absmax": float(ref.abs().max()),
    }
print(json.dumps(results, indent=2))
