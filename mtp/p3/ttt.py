"""Training-time-test (TTT) forward of the MTP draft, all anchors of one sequence in parallel.

Semantics (task spec, verified against the vLLM V2 speculator):
  * Step 0: one causal forward over all T rows of the sequence.  Row p gets hidden h_p (the target
    multi-stream state at p) and input token x_{p+1}; it predicts the token at p+2.
  * Anchor e (0 <= e <= T-6, so that every label target_argmax[e+1+j] exists) has draft rows
      step j (j = 1..3): position e+j, input token x_{e+1+j} (ground truth), hidden = the same
      anchor's step j-1 multi_hidden_out (differentiable).  Its prediction targets the token at e+2+j.
  * Attention of step j for anchor e: keys = step-0 rows at positions <= e, plus the anchor's own
    rows of steps 1..j.  Implemented as one attention over the concatenated keys
        [step0 rows (T) | step1 rows (N) | ... | step j rows (N)]
    with an explicit boolean mask (EAGLE-3 style).  N = number of anchors = T-5.

Only the attention differs from mtp_torch.MTPDraft.forward; projections, norms, RoPE, hyper
connections, MoE (grouped version) and the final mixer are the same modules/functions and the same
weights.  Step 0 is exactly MTPDraft.forward on the full sequence.
"""

import sys

import torch

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp")
sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
from mtp_torch import apply_partial_neox_rope  # noqa: E402
from grouped_moe import grouped_moe_forward  # noqa: E402

K_STEPS = 4  # chain length (steps 0..3), NSPEC=4 in the release preset


def anchor_range(T: int, device="cpu") -> torch.Tensor:
    """Valid anchors e in [0, T-6]: all labels target_argmax[e+1+j], j=0..3, exist (len = T-1)."""
    return torch.arange(0, T - 5, device=device)


def _project_qkv(attn, x, pos):
    """Mirrors QSAAttention.forward lines 373-381 for x [N, H] treated as one sequence."""
    N = x.shape[0]
    q_gate = attn.q_proj(x).view(N, attn.nh, 2 * attn.hd)
    q, gate = q_gate.chunk(2, dim=-1)
    k = attn.k_proj(x).view(N, attn.nkv, attn.hd)
    v = attn.v_proj(x).view(N, attn.nkv, attn.hd)
    q = attn.q_norm(q)
    k = attn.k_norm(k)
    q = apply_partial_neox_rope(q[None], pos[None], attn.rotary_dim, attn.rope_base)[0]
    k = apply_partial_neox_rope(k[None], pos[None], attn.rotary_dim, attn.rope_base)[0]
    return q, k, v, gate


def _masked_attention(attn, q, K, V, mask, gate):
    """q [N,nh,hd]; K,V [S,nkv,hd]; mask [N,S] bool (True = visible); gate [N,nh,hd].

    Same arithmetic as QSAAttention.forward (fp32 scores, scale, softmax, sigmoid output gate).
    """
    rep = attn.nh // attn.nkv
    Kr = K.repeat_interleave(rep, dim=1).float()
    Vr = V.repeat_interleave(rep, dim=1).float()
    scores = torch.einsum("nhd,shd->hns", q.float(), Kr) * attn.scale  # [nh, N, S]
    scores = scores.masked_fill(~mask[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hns,shd->nhd", probs, Vr)
    out = out * torch.sigmoid(gate.float())
    return out.to(q.dtype).reshape(q.shape[0], attn.nh * attn.hd)


def ttt_forward(model, multi: torch.Tensor, x_ids: torch.Tensor, anchors: torch.Tensor = None):
    """TTT forward for one sequence.

    model:  MTPDraft (eval mode, dtype = model dtype)
    multi:  [T, D] target multi-stream hidden (D = HC*H), model dtype
    x_ids:  [T] int64 input token ids of the sequence (x_0..x_{T-1})
    anchors: [N] int64 anchor positions (default anchor_range(T))
    Returns samples [K_STEPS, N, H]: sample_hidden of step j for each anchor (step 0 included).
    """
    T = multi.shape[0]
    dev = multi.device
    if anchors is None:
        anchors = anchor_range(T, dev)
    anchors = anchors.to(dev)
    N = anchors.numel()
    assert int(anchors.max()) <= T - 6, "anchor e needs label target_argmax[e+4] (index <= T-2)"
    layer = model.layers[0]
    attn = layer.self_attn
    H, HC, D = model.cfg.hidden_size, model.cfg.hc_count, model.cfg.mtp_width

    # ---- step 0: full causal forward (identical to MTPDraft.forward on the sequence) ----
    pos0 = torch.arange(T, device=dev)
    ids0 = torch.empty_like(x_ids)
    ids0[:-1] = x_ids[1:]
    ids0[-1] = x_ids[-1]  # row T-1 has no successor; it is never visible to any anchor
    s0, m0, kv0 = model(multi[None], ids0[None], pos0[None], kv=None)
    samples = [s0[0, anchors]]
    carried = m0[0, anchors]  # [N, D] multi_hidden_out of step 0 at the anchor rows

    # ---- steps 1..K-1: explicit mask over [step0 | step1 | ... | step j] keys ----
    K_blocks = [kv0.k[0]]  # [T, nkv, hd]
    V_blocks = [kv0.v[0]]
    mask0 = pos0[None, :] <= anchors[:, None]  # [N, T]: step-0 rows at positions <= e only
    eye = torch.eye(N, dtype=torch.bool, device=dev)  # own-chain rows of steps 1..j
    for j in range(1, K_STEPS):
        tok = x_ids[anchors + 1 + j]  # ground-truth x_{e+1+j}
        p = anchors + j  # RoPE / causal position of this step's row
        emb = model.pre_fc_norm_embedding(model.embed_tokens(tok))
        prev_block = model.fc_embedding(emb)  # [N, H]
        hid = model.pre_fc_norm_hidden(carried)  # [N, D]
        hid = model.fc_hidden(hid.reshape(N, HC, H))
        hidden = hid.reshape(N, D)
        hidden, block_in, inj = layer.attn_hyper_connection.combine_and_mix(hidden, prev_block, None)
        q, k_cur, v_cur, gate = _project_qkv(attn, block_in, p)
        K_blocks.append(k_cur)
        V_blocks.append(v_cur)
        K = torch.cat(K_blocks, dim=0)
        V = torch.cat(V_blocks, dim=0)
        mask = torch.cat([mask0] + [eye] * j, dim=1)  # [N, T + j*N]
        out = _masked_attention(attn, q, K, V, mask, gate)
        attn_out = attn.o_proj(out)
        hidden, block_in2, inj2 = layer.mlp_hyper_connection.combine_and_mix(hidden, attn_out, inj)
        mlp_out = grouped_moe_forward(layer.mlp, block_in2)
        multi_j, sample_j, _ = model.hyper_connection_mixer.combine_and_mix(hidden, mlp_out, inj2)
        samples.append(sample_j)
        carried = multi_j
    return torch.stack(samples, dim=0)
