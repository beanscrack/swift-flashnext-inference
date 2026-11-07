"""Backward-efficient variant of p3/grouped_moe.grouped_moe_forward (identical forward math).

The p3 version indexes gate_up[e] / down[e] per expert inside the autograd graph.  Each index
backward materialises a full [E, ...] gradient (gate_up: 512 x 1280 x 2560 bf16 = 3.3 GB), so one
training forward/backward costs ~2 * E full-size writes per call.  Here the expert stacks are unbound
once per call, so autograd builds one stacked gradient per parameter per call.  Values are the same
tensors (views of the same storage) and the same operations in the same order.
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
from grouped_moe import ROUTER_DTYPE  # noqa: E402


def grouped_moe_forward_fast(moe, x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, "MoEBlock takes 2-D [N, H] input"
    N, H = x.shape
    E = moe.experts.gate_up_proj.shape[0]
    K = moe.top_k
    dev = x.device

    logits = F.linear(x.to(ROUTER_DTYPE), moe.gate.weight.to(ROUTER_DTYPE)).float()
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)
    if moe.norm_topk_prob:
        w = w / w.sum(dim=-1, keepdim=True)

    flat_idx = idx.reshape(-1)
    flat_w = w.reshape(-1)
    tok = torch.arange(N, device=dev).repeat_interleave(K)
    order = torch.argsort(flat_idx, stable=True)
    counts = torch.bincount(flat_idx, minlength=E).tolist()
    tok_s = tok[order]
    xs = x[tok_s]
    ws = flat_w[order]

    gate_ups = moe.experts.gate_up_proj.unbind(0)  # one autograd node per parameter, not per expert
    downs = moe.experts.down_proj.unbind(0)
    ys = []
    start = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        xe = xs[start : start + c]
        gu = F.linear(xe, gate_ups[e])
        g, u = gu.chunk(2, dim=-1)
        h = F.silu(g) * u
        ys.append(F.linear(h, downs[e]).float() * ws[start : start + c, None])
        start += c
    y = torch.cat(ys, dim=0)
    routed = torch.zeros(N, H, dtype=torch.float32, device=dev).index_add_(0, tok_s, y)
    routed = routed.to(x.dtype)

    s = moe.shared_expert
    shared = s.down_proj(F.silu(s.gate_proj(x)) * s.up_proj(x))
    shared = torch.sigmoid(moe.shared_expert_gate(x).float()).to(x.dtype) * shared
    return routed + shared


# ---- switch used by train.py: fast MoE for the training forward only; eval keeps the original code ----
import mtp_torch  # noqa: E402
import ttt  # noqa: E402

_ORIG_BLOCK_FORWARD = mtp_torch.MoEBlock.forward
_ORIG_GROUPED = ttt.grouped_moe_forward


def _block_fast(self, x):
    return grouped_moe_forward_fast(self, x)


def set_fast(on: bool):
    """on=True: MoEBlock.forward (step 0 of ttt_forward) and ttt's grouped MoE use the fast variant."""
    if on:
        mtp_torch.MoEBlock.forward = _block_fast
        ttt.grouped_moe_forward = grouped_moe_forward_fast
    else:
        mtp_torch.MoEBlock.forward = _ORIG_BLOCK_FORWARD
        ttt.grouped_moe_forward = _ORIG_GROUPED
