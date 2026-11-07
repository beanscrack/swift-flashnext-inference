"""Grouped (sort-by-expert) MoE forward, numerically equivalent to mtp_torch.MoEBlock.forward.

The reference loop in mtp_torch.MoEBlock.forward scans all E experts with a boolean mask over
the [N, top_k] assignment table on every call.  Here the N*top_k (token, expert) assignments are
sorted once by expert id, each expert's contiguous slice is multiplied, and the weighted outputs
are scatter-added back to the token rows.  Router, softmax, top-k, renormalisation, dtype casts
and the shared expert are the same operations as in MoEBlock.forward.
"""

import torch
import torch.nn.functional as F

# Router GEMM dtype: MoEBlock.forward (and vLLM) use bf16.  Only diagnostics override this.
ROUTER_DTYPE = torch.bfloat16


def grouped_moe_forward(moe, x: torch.Tensor) -> torch.Tensor:
    """x: [N, H] (2-D, as MoEBlock.forward expects).  Returns [N, H] in x.dtype."""
    assert x.dim() == 2, "MoEBlock takes 2-D [N, H] input"
    N, H = x.shape
    E = moe.experts.gate_up_proj.shape[0]
    K = moe.top_k
    dev = x.device

    # router: identical ops to MoEBlock.forward (bf16 GEMM, fp32 softmax)
    logits = F.linear(x.to(ROUTER_DTYPE), moe.gate.weight.to(ROUTER_DTYPE)).float()
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, K, dim=-1)  # [N, K]
    if moe.norm_topk_prob:
        w = w / w.sum(dim=-1, keepdim=True)

    flat_idx = idx.reshape(-1)  # assignment a = n*K + k
    flat_w = w.reshape(-1)
    tok = torch.arange(N, device=dev).repeat_interleave(K)  # token row of each assignment
    order = torch.argsort(flat_idx, stable=True)
    counts = torch.bincount(flat_idx, minlength=E).tolist()
    tok_s = tok[order]
    xs = x[tok_s]  # rows grouped by expert
    ws = flat_w[order]

    gate_up = moe.experts.gate_up_proj
    down = moe.experts.down_proj
    ys = []
    start = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        xe = xs[start : start + c]
        gu = F.linear(xe, gate_up[e])
        g, u = gu.chunk(2, dim=-1)
        h = F.silu(g) * u
        ys.append(F.linear(h, down[e]).float() * ws[start : start + c, None])
        start += c
    y = torch.cat(ys, dim=0)
    routed = torch.zeros(N, H, dtype=torch.float32, device=dev).index_add_(0, tok_s, y)
    routed = routed.to(x.dtype)

    s = moe.shared_expert
    shared = s.down_proj(F.silu(s.gate_proj(x)) * s.up_proj(x))
    shared = torch.sigmoid(moe.shared_expert_gate(x).float()).to(x.dtype) * shared
    return routed + shared
