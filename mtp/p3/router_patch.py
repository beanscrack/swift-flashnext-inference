"""Diagnostic-only: fp32 MoE router in both the reference (MoEBlock.forward) and the grouped TTT path."""
import torch
import torch.nn.functional as F
import mtp_torch
import grouped_moe as gm


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


def apply():
    mtp_torch.MoEBlock.forward = moe_fp32_router
    gm.ROUTER_DTYPE = torch.float32
