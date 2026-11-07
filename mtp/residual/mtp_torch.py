"""Standalone PyTorch re-implementation of the Qwen4Exp MTP draft module (vLLM 0.31).

Scope: the single-layer MTP head (``mtp.*`` tensors) used as a training / parity
reference.  The target's ``embed_tokens`` and ``lm_head`` are *not* owned by the
draft proper; ``embed_tokens`` is loaded from the same file when present, and
``lm_head`` rows are passed explicitly to :meth:`MTPDraft.unroll`.

Every mirrored vLLM function carries a ``file:line`` citation (line numbers refer
to the vLLM 0.31 install under
``/u701/j692wu/.conda/envs/inference/lib/python3.1/site-packages/vllm``).

Known limits (see the report RISKS section):
  * Sparse QSA selection is NOT implemented.  When ``past + T <= indexer_budget``
    the selection covers every causal key, so dense causal attention is exact
    (see ``QSAAttention.forward``).  Longer contexts raise ``ValueError``.
  * The indexer weights (``index_qk_proj``, ``q_layernorm``, ``k_layernorm``) are
    loaded but never used on the in-budget path.
  * MoE is a per-expert loop (correct, not fast).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MTP_PREFIX = "mtp."
ATTN_MODE = "fp32"  # residual experiments: fp32 | pbf16 | vllm
ROUTER_MODE = "bf16"  # residual experiments: bf16 | fp32
ROPE_MODE = "bf16"  # residual experiments: bf16 | fused | fp32cs | shift+1 | shift-1
EMBED_KEY = "model.language_model.embed_tokens.weight"

# The 29 non-expert ``mtp.*`` tensors (names relative to ``mtp.``).  Verified
# against the BF16 tensors of the nvidia checkpoint
# (/u701/j692wu/models/nvidia-Qwen3.8-Flash-Next-NVFP4, shards 9-10) and against
# vLLM's loader (models/qwen4_exp/nvidia/mtp.py:_remap_mtp_weight_name, 99-130).
NON_EXPERT_KEYS = (
    "fc_embedding.weight",  # mtp.py:199-204 (ColumnParallelLinear H->H, no bias)
    "fc_hidden.weight",  # mtp.py:205-210
    "pre_fc_norm_embedding.weight",  # mtp.py:236
    "pre_fc_norm_hidden.weight",  # mtp.py:238-240 (size H*HC)
    "hyper_connection_mixer.hc_norm.weight",
    "hyper_connection_mixer.input_mix_weight_down.weight",
    "hyper_connection_mixer.input_mix_weight_up.weight",
    "layers.0.attn_hyper_connection.block_inject_weight.weight",
    "layers.0.attn_hyper_connection.hc_norm.weight",
    "layers.0.attn_hyper_connection.input_mix_weight_down.weight",
    "layers.0.attn_hyper_connection.input_mix_weight_up.weight",
    "layers.0.mlp_hyper_connection.block_inject_weight.weight",
    "layers.0.mlp_hyper_connection.hc_norm.weight",
    "layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
    "layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
    "layers.0.mlp.gate.weight",
    "layers.0.mlp.shared_expert.down_proj.weight",
    "layers.0.mlp.shared_expert.gate_proj.weight",
    "layers.0.mlp.shared_expert.up_proj.weight",
    "layers.0.mlp.shared_expert_gate.weight",
    "layers.0.self_attn.indexer.index_qk_proj.weight",
    "layers.0.self_attn.indexer.k_layernorm.weight",
    "layers.0.self_attn.indexer.q_layernorm.weight",
    "layers.0.self_attn.k_norm.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.v_proj.weight",
)  # 29 tensors

# Expert layouts.  The Swift tail shard named in the brief is not present on
# datasci, so the fused layout (29 + 2 = 31) is an ASSUMPTION; per-expert HF
# names are accepted too (29 + 3*512 = 1565).  Both are checked by name.
EXPERT_FUSED_KEYS = ("layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.experts.down_proj")
_EXPERT_PROJ = ("gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MTPConfig:
    hidden_size: int = 2560
    hc_count: int = 4
    hc_lowrank: int = 320
    num_heads: int = 24
    num_kv_heads: int = 2
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    rope_theta: float = 1e7
    rms_norm_eps: float = 1e-6
    num_experts: int = 512
    top_k: int = 10
    moe_intermediate: int = 640
    shared_intermediate: int = 640
    norm_topk_prob: bool = True
    indexer_budget: int = 2048
    indexer_heads: int = 4
    indexer_head_dim: int = 128
    indexer_kv_heads: int = 1
    vocab_size: int = 248320

    @property
    def rotary_dim(self) -> int:
        # vLLM: rotary_dim = int(head_size * partial_rotary_factor) (rotary_embedding/__init__.py:68-71)
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def mtp_width(self) -> int:
        return self.hc_count * self.hidden_size

    @classmethod
    def from_config_json(cls, path, **overrides) -> "MTPConfig":
        raw = json.loads(Path(path).read_text())
        t = raw.get("text_config", raw)
        rope = t.get("rope_parameters", {})
        assert t.get("mtp_num_hidden_layers", 1) == 1, "single MTP layer expected"
        assert t.get("output_gate_type", "sigmoid") == "sigmoid"
        assert t.get("decoder_sparse_step", 1) == 1 and not t.get("mlp_only_layers")
        kw = dict(
            hidden_size=t["hidden_size"],
            hc_count=t["hc_count"],
            hc_lowrank=t["hc_lowrank"],
            num_heads=t["num_attention_heads"],
            num_kv_heads=t["num_key_value_heads"],
            head_dim=t["head_dim"],
            partial_rotary_factor=rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0)),
            rope_theta=float(rope.get("rope_theta", t.get("rope_theta", 1e4))),
            rms_norm_eps=t["rms_norm_eps"],
            num_experts=t["num_experts"],
            top_k=t["num_experts_per_tok"],
            moe_intermediate=t["moe_intermediate_size"],
            shared_intermediate=t["shared_expert_intermediate_size"],
            norm_topk_prob=t.get("norm_topk_prob", True),
            indexer_budget=t["indexer_budget"],
            indexer_heads=t["indexer_n_heads"],
            indexer_head_dim=t["indexer_head_dim"],
            indexer_kv_heads=t["indexer_kv_heads"],
            vocab_size=t["vocab_size"],
        )
        kw.update(overrides)
        return cls(**kw)


def expected_mtp_names(cfg: MTPConfig, layout: str = "fused") -> dict[str, tuple[int, ...]]:
    """Map every expected ``mtp.*`` name (with prefix) to its shape."""
    H, HC, L = cfg.hidden_size, cfg.hc_count, cfg.hc_lowrank
    D = HC * H
    nh, nkv, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    E, I, S = cfg.num_experts, cfg.moe_intermediate, cfg.shared_intermediate
    ih, ikv, idh = cfg.indexer_heads, cfg.indexer_kv_heads, cfg.indexer_head_dim
    shapes = {
        "fc_embedding.weight": (H, H),
        "fc_hidden.weight": (H, H),
        "pre_fc_norm_embedding.weight": (H,),
        "pre_fc_norm_hidden.weight": (D,),
        "hyper_connection_mixer.hc_norm.weight": (D,),
        "hyper_connection_mixer.input_mix_weight_down.weight": (L, D),
        "hyper_connection_mixer.input_mix_weight_up.weight": (D, L),
        "layers.0.mlp.gate.weight": (E, H),
        "layers.0.mlp.shared_expert.down_proj.weight": (H, S),
        "layers.0.mlp.shared_expert.gate_proj.weight": (S, H),
        "layers.0.mlp.shared_expert.up_proj.weight": (S, H),
        "layers.0.mlp.shared_expert_gate.weight": (1, H),
        "layers.0.self_attn.indexer.index_qk_proj.weight": ((ih + ikv) * idh, H),
        "layers.0.self_attn.indexer.k_layernorm.weight": (idh,),
        "layers.0.self_attn.indexer.q_layernorm.weight": (idh,),
        "layers.0.self_attn.k_norm.weight": (hd,),
        "layers.0.self_attn.k_proj.weight": (nkv * hd, H),
        "layers.0.self_attn.o_proj.weight": (H, nh * hd),
        "layers.0.self_attn.q_norm.weight": (hd,),
        "layers.0.self_attn.q_proj.weight": (2 * nh * hd, H),
        "layers.0.self_attn.v_proj.weight": (nkv * hd, H),
    }
    for m in ("attn", "mlp"):
        shapes[f"layers.0.{m}_hyper_connection.hc_norm.weight"] = (D,)
        shapes[f"layers.0.{m}_hyper_connection.input_mix_weight_down.weight"] = (L, D)
        shapes[f"layers.0.{m}_hyper_connection.input_mix_weight_up.weight"] = (D, L)
        shapes[f"layers.0.{m}_hyper_connection.block_inject_weight.weight"] = (HC, D)
    assert set(shapes) == set(NON_EXPERT_KEYS), set(shapes) ^ set(NON_EXPERT_KEYS)
    if layout == "fused":
        shapes["layers.0.mlp.experts.gate_up_proj"] = (E, 2 * I, H)
        shapes["layers.0.mlp.experts.down_proj"] = (E, H, I)
    elif layout == "per_expert":
        for e in range(E):
            shapes[f"layers.0.mlp.experts.{e}.gate_proj.weight"] = (I, H)
            shapes[f"layers.0.mlp.experts.{e}.up_proj.weight"] = (I, H)
            shapes[f"layers.0.mlp.experts.{e}.down_proj.weight"] = (H, I)
    else:
        raise ValueError(layout)
    return {MTP_PREFIX + k: v for k, v in shapes.items()}


def detect_expert_layout(names) -> str:
    names = set(names)
    if all(MTP_PREFIX + k in names for k in EXPERT_FUSED_KEYS):
        return "fused"
    if any(".mlp.experts.0.gate_proj.weight" in n for n in names):
        return "per_expert"
    raise ValueError("no recognised expert layout among mtp.* names")


# ---------------------------------------------------------------------------
# primitive ops (each mirrors a named vLLM function)
# ---------------------------------------------------------------------------
def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Mirrors layernorm.py:GemmaRMSNorm.forward_native -> ir/ops/layernorm.py:rms_norm.

    fp32 variance, x * rsqrt(var + eps), then * (1 + w) in fp32, cast back.
    """
    orig = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y = xf * torch.rsqrt(var + eps)
    return (y * (1.0 + weight.float())).to(orig)


def grouped_gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, groups: int) -> torch.Tensor:
    """Per-stream RMS norm over H = D/groups with a full D-length affine.

    Mirrors common/hyperconnection.py:GroupedGemmaRMSNorm.forward (group_size=H)
    and nvidia/ops/hc.py:_grouped_gemma_rmsnorm_kernel (y = x*rrms; y += y*w).
    """
    orig = x.dtype
    xf = x.float().unflatten(-1, (groups, -1))
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y = (xf * torch.rsqrt(var + eps)).flatten(-2)
    return (y * (1.0 + weight.float())).to(orig)


def apply_partial_neox_rope(x: torch.Tensor, positions: torch.Tensor, rotary_dim: int, base: float) -> torch.Tensor:
    """Partial NeoX RoPE on the first ``rotary_dim`` dims of each head.

    x: [B, T, heads, head_dim]; positions: [B, T] (text: all three M-RoPE axes equal,
    so the 1-D path is exact).  Mirrors:
      * rotary_embedding/base.py:_compute_inv_freq / _compute_cos_sin_cache (fp32 table)
      * rotary_embedding/mrope.py:MRotaryEmbedding.forward_native, 1-D branch (cos/sin cast to x.dtype)
      * rotary_embedding/common.py:ApplyRotaryEmb.forward_static (is_neox_style=True)
    """
    half = rotary_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=x.device) / rotary_dim))
    pos = positions.to(torch.float32)
    if ROPE_MODE == "shift+1":
        pos = pos + 1.0
    elif ROPE_MODE == "shift-1":
        pos = pos - 1.0
    freqs = pos[..., None] * inv_freq  # [B, T, half]
    if ROPE_MODE == "fp32cs":
        cos = freqs.cos()[:, :, None, :]
        sin = freqs.sin()[:, :, None, :]
        xf = x.float()
        xr, xp = xf[..., :rotary_dim], xf[..., rotary_dim:]
        x1, x2 = xr[..., :half], xr[..., half:]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat([o1, o2, xp], dim=-1).to(x.dtype)
    cos = freqs.cos().to(x.dtype)[:, :, None, :]
    sin = freqs.sin().to(x.dtype)[:, :, None, :]
    if ROPE_MODE == "fused":
        # ops: fused_qk_norm_rope.py:125-126 -- fp32 math on bf16 cos/sin, single rounding at store
        xf = x.float()
        cf, sf = cos.float(), sin.float()
        xr, xp = xf[..., :rotary_dim], xf[..., rotary_dim:]
        x1, x2 = xr[..., :half], xr[..., half:]
        o1 = x1 * cf - x2 * sf
        o2 = x2 * cf + x1 * sf
        return torch.cat([o1, o2, xp], dim=-1).to(x.dtype)
    xr, xp = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = xr[..., :half], xr[..., half:]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat([o1, o2, xp], dim=-1)


class GemmaNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gemma_rms_norm(x, self.weight, self.eps)


class GroupedGemmaNorm(nn.Module):
    def __init__(self, dim: int, groups: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.groups = groups
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grouped_gemma_rms_norm(x, self.weight, self.eps, self.groups)


# ---------------------------------------------------------------------------
# hyper-connection mixer (GatedResidual, the use_combine=True / False variants)
# ---------------------------------------------------------------------------
class HyperMixer(nn.Module):
    """Mirrors nvidia/hyperconnection.py:GatedResidual (+ ops/hc.py kernels).

    with_inject=True  -> layer HC (attn/mlp): has block_inject_weight, combine uses the
                         injection logits produced by the *previous* mix.
    with_inject=False -> final mixer (use_combine=False in mtp.py:228-236): it still
                         combines with the injection handed in by the decoder layer.
    """

    def __init__(self, cfg: MTPConfig, with_inject: bool):
        super().__init__()
        self.hc = cfg.hc_count
        self.h = cfg.hidden_size
        self.d = cfg.hc_count * cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        self.hc_norm = GroupedGemmaNorm(self.d, self.hc, cfg.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(self.d, cfg.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(cfg.hc_lowrank, self.d, bias=False)
        self.block_inject_weight = nn.Linear(self.d, self.hc, bias=False) if with_inject else None

    def _gate(self, xn: torch.Tensor):
        """Down/inject projections, hc_silu, up projection, gated mean over streams."""
        n = xn.shape[0]
        dn = self.input_mix_weight_down(xn)  # [N, L]
        # ops/hc.py:_hc_silu_kernel: x = y/HC; y = x*sigmoid(x)
        lora = (F.silu(dn.float() / self.hc)).to(dn.dtype)
        inj = self.block_inject_weight(xn) if self.block_inject_weight is not None else None
        gate = self.input_mix_weight_up(lora)  # [N, D]
        # ops/hc.py:_hc_gate_mix_kernel: acc = sum_s sigmoid(g_s) * x_s; acc /= HC
        g = torch.sigmoid(gate.float()).view(n, self.hc, self.h)
        block_input = (g * xn.float().view(n, self.hc, self.h)).sum(dim=1) / self.hc
        return block_input.to(xn.dtype), inj

    def combine(self, res: torch.Tensor, block: torch.Tensor, inj: Optional[torch.Tensor]) -> torch.Tensor:
        """Residual update.  ops/hc.py:_hc_combine_norm_kernel (rounded to residual dtype).

        Injection weight 2*sigmoid(logit/HC) per stream (ops/hc.py:_hc_combine_norm_kernel);
        inj=None -> unit weight (as in the kernel's inj_ptr-is-None branch).
        common/hyperconnection.py:GatedResidual.combine is the torch reference of the same math.
        """
        n = res.shape[0]
        r = res.float().view(n, self.hc, self.h)
        b = block.float()[:, None, :]
        if inj is not None:
            b = b * (2.0 * torch.sigmoid(inj.float() / self.hc))[:, :, None]
        return (r + b).reshape(n, self.d).to(res.dtype)

    def norm_mix(self, res: torch.Tensor):
        """Mix without combine: returns (block_input, normalized xn, injection logits)."""
        xn = self.hc_norm(res)
        block, inj = self._gate(xn)
        return block, xn, inj

    def combine_and_mix(self, res: torch.Tensor, block: torch.Tensor, inj: Optional[torch.Tensor]):
        """Mirrors nvidia/hyperconnection.py:GatedResidual.combine_and_mix (line 177)."""
        new_res = self.combine(res, block, inj)
        xn = self.hc_norm(new_res)
        blk, inj_new = self._gate(xn)
        return new_res, blk, inj_new


# ---------------------------------------------------------------------------
# attention (Qwen4ExpQSAAttention, eager reference path)
# ---------------------------------------------------------------------------
@dataclass
class KVState:
    k: torch.Tensor  # [B, S, nkv, hd] (post k_norm + RoPE, as stored by vLLM)
    v: torch.Tensor  # [B, S, nkv, hd]
    pos: torch.Tensor  # [B, S] absolute positions


class QSAAttention(nn.Module):
    """Mirrors qsa.py:Qwen4ExpQSAAttention.forward (qsa.py ~L470-520) and the eager
    projection path of qwen3_next.py:Qwen3NextAttention._project_qkv_gate (L387-436)."""

    def __init__(self, cfg: MTPConfig):
        super().__init__()
        H = cfg.hidden_size
        self.nh, self.nkv, self.hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.rotary_dim = cfg.rotary_dim
        self.rope_base = cfg.rope_theta
        self.eps = cfg.rms_norm_eps
        self.budget = cfg.indexer_budget
        self.scale = self.hd**-0.5  # qsa.py:Qwen4ExpQSAAttention.__init__ (self.scaling)
        self.q_proj = nn.Linear(H, 2 * self.nh * self.hd, bias=False)  # [q | gate] per head
        self.k_proj = nn.Linear(H, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(H, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, H, bias=False)
        self.q_norm = GemmaNorm(self.hd, cfg.rms_norm_eps)
        self.k_norm = GemmaNorm(self.hd, cfg.rms_norm_eps)
        self.indexer = nn.Module()  # weights only; selection not run in-budget
        self.indexer.index_qk_proj = nn.Linear(H, (cfg.indexer_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim, bias=False)
        self.indexer.q_layernorm = GemmaNorm(cfg.indexer_head_dim, cfg.rms_norm_eps)
        self.indexer.k_layernorm = GemmaNorm(cfg.indexer_head_dim, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, kv: Optional[KVState]):
        B, T, _ = x.shape
        q_gate = self.q_proj(x).view(B, T, self.nh, 2 * self.hd)
        # qwen3_next.py:401-405 -> per-head chunk: first hd = query, second hd = output gate
        q, gate = q_gate.chunk(2, dim=-1)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd)
        q = self.q_norm(q)  # qwen3_next.py:424-425
        k = self.k_norm(k)
        q = apply_partial_neox_rope(q, positions, self.rotary_dim, self.rope_base)  # qwen3_next.py:435
        k = apply_partial_neox_rope(k, positions, self.rotary_dim, self.rope_base)

        if kv is None:
            K, V, P = k, v, positions
        else:
            K = torch.cat([kv.k, k], dim=1)
            V = torch.cat([kv.v, v], dim=1)
            P = torch.cat([kv.pos, positions], dim=1)
        S = K.shape[1]
        if S > self.budget:
            # The indexer selects <= budget keys; with S <= budget it selects every causal key
            # (indexer_qsa.py output_width / expand_qsa_block_indices, qsa_indexer.py:442-473),
            # so dense causal attention is exact.  Beyond that we do not emulate selection.
            raise ValueError(f"sequence length {S} exceeds indexer_budget={self.budget}; sparse QSA not implemented")
        new_kv = KVState(K, V, P)

        rep = self.nh // self.nkv  # kv head of query head h is h // rep (qsa.py:89 first_head = kv_head*GROUP)
        Kr = K.repeat_interleave(rep, dim=2).float()
        Vr = V.repeat_interleave(rep, dim=2).float()
        scores = torch.einsum("bthd,bshd->bhts", q.float(), Kr) * self.scale
        # causal over *absolute positions* (a key is visible iff its position <= query position)
        mask = P[:, None, :] <= positions[:, :, None]  # [B, T, S]
        scores = scores.masked_fill(~mask[:, None, :, :], float("-inf"))
        if ATTN_MODE == "fp32":
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("bhts,bshd->bthd", probs, Vr)
        else:
            # mirror ops/qsa.py _qsa_sparse_paged_gqa_splitk_kernel: P rounded to bf16 for the PV dot,
            # normalizer summed from fp32 P (ops/qsa.py:173-186)
            m = scores.amax(dim=-1, keepdim=True)
            pe = torch.exp(scores - m)
            denom = pe.sum(dim=-1, keepdim=True)
            num = torch.einsum("bhts,bshd->bthd", pe.to(torch.bfloat16).float(), Vr)
            out = num / denom.transpose(1, 2)
            if ATTN_MODE == "vllm":
                # ops/qsa.py:202-205 normalized output rounded to bf16 before the gate multiply
                out = out.to(torch.bfloat16).float()
        out = out * torch.sigmoid(gate.float())  # qsa.py:205 normalized_output * sigmoid(output_gate)
        out = out.to(x.dtype).reshape(B, T, self.nh * self.hd)
        return self.o_proj(out), new_kv


# ---------------------------------------------------------------------------
# MoE (Qwen3NextSparseMoeBlock + Qwen2MoeMLP shared expert)
# ---------------------------------------------------------------------------
class SharedMLP(nn.Module):
    """Qwen2MoeMLP (model_executor/models/qwen2_moe.py:76-120): down(silu(gate)*up) * sigmoid(expert_gate(x))."""

    def __init__(self, H: int, S: int):
        super().__init__()
        self.gate_proj = nn.Linear(H, S, bias=False)
        self.up_proj = nn.Linear(H, S, bias=False)
        self.down_proj = nn.Linear(S, H, bias=False)


class _Experts(nn.Module):
    def __init__(self, E: int, H: int, I: int):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(E, 2 * I, H))  # [gate; up] along out dim
        self.down_proj = nn.Parameter(torch.zeros(E, H, I))


class MoEBlock(nn.Module):
    """Mirrors qwen3_next.py:Qwen3NextSparseMoeBlock (L131-270) with the router of
    fused_topk_router.py:fused_topk (softmax, top-k, renormalize = norm_topk_prob) and
    the fp32 router GEMM fallback (fp32_router_gemm_dispatch_impl: F.linear(x.float(), w)).
    routed_scaling_factor = 1.0 (FusedMoE default; no config key)."""

    def __init__(self, cfg: MTPConfig):
        super().__init__()
        self.top_k = cfg.top_k
        self.norm_topk_prob = cfg.norm_topk_prob
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.shared_expert = SharedMLP(cfg.hidden_size, cfg.shared_intermediate)
        self.shared_expert_gate = nn.Linear(cfg.hidden_size, 1, bias=False)
        self.experts = _Experts(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # vLLM computes the router GEMM in bf16 (moe_runner.py:910, GateLinear out_dtype unset); softmax in fp32
        if ROUTER_MODE == "fp32":
            logits = F.linear(x.float(), self.gate.weight.float())
        else:
            logits = F.linear(x.to(torch.bfloat16), self.gate.weight.to(torch.bfloat16)).float()
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


# ---------------------------------------------------------------------------
# decoder layer and draft model
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """Mirrors model.py:Qwen4ExpDecoderLayer.forward (L277-333) for the MTP layer (no PLE)."""

    def __init__(self, cfg: MTPConfig):
        super().__init__()
        self.attn_hyper_connection = HyperMixer(cfg, with_inject=True)
        self.mlp_hyper_connection = HyperMixer(cfg, with_inject=True)
        self.self_attn = QSAAttention(cfg)
        self.mlp = MoEBlock(cfg)


class MTPDraft(nn.Module):
    """Mirrors models/qwen4_exp/nvidia/mtp.py:Qwen4ExpMultiTokenPredictor.forward (L276-342)."""

    def __init__(self, cfg: MTPConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        self.embed_tokens = nn.Embedding(cfg.vocab_size, H)
        self.fc_embedding = nn.Linear(H, H, bias=False)
        self.fc_hidden = nn.Linear(H, H, bias=False)
        self.pre_fc_norm_embedding = GemmaNorm(H, cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaNorm(cfg.mtp_width, cfg.rms_norm_eps)
        self.hyper_connection_mixer = HyperMixer(cfg, with_inject=False)
        self.layers = nn.ModuleList([DecoderLayer(cfg)])
        self.embed_loaded = False

    # -- loading -----------------------------------------------------------
    @classmethod
    def from_safetensors(cls, path, config_json, *, dtype=torch.bfloat16, device="cpu",
                         config_overrides: Optional[dict] = None, strict_names: bool = True) -> "MTPDraft":
        from safetensors import safe_open

        cfg = MTPConfig.from_config_json(config_json, **(config_overrides or {}))
        model = cls(cfg)
        with safe_open(str(path), framework="pt", device="cpu") as f:
            all_names = set(f.keys())
            mtp_names = {n for n in all_names if n.startswith(MTP_PREFIX)}
            layout = detect_expert_layout(mtp_names)
            expected = expected_mtp_names(cfg, layout)
            missing = sorted(set(expected) - mtp_names)
            unexpected = sorted(mtp_names - set(expected))
            if strict_names and (missing or unexpected):
                raise ValueError(f"mtp.* name mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
            tensors = {n: f.get_tensor(n) for n in mtp_names}
            if EMBED_KEY in all_names:
                tensors["__embed__"] = f.get_tensor(EMBED_KEY)
        model.load_mtp_tensors(tensors)
        return model.to(device=device, dtype=dtype)

    def load_mtp_tensors(self, tensors: dict) -> list[str]:
        """Load ``mtp.*`` tensors (keys with or without the prefix) plus optional ``__embed__``."""
        state: dict[str, torch.Tensor] = {}
        experts: dict[str, dict[int, torch.Tensor]] = {}
        for name, t in tensors.items():
            if name == "__embed__":
                state["embed_tokens.weight"] = t
                continue
            key = name[len(MTP_PREFIX):] if name.startswith(MTP_PREFIX) else name
            if key.startswith("layers.0.mlp.experts.") and key.count(".") == 6:
                # per-expert HF layout: layers.0.mlp.experts.{e}.{proj}.weight
                parts = key.split(".")  # ['layers','0','mlp','experts','{e}','{proj}','weight']
                experts.setdefault(parts[5], {})[int(parts[4])] = t
                continue
            state[key] = t
        if experts:
            E = self.cfg.num_experts
            gate = torch.stack([experts["gate_proj"][e] for e in range(E)])
            up = torch.stack([experts["up_proj"][e] for e in range(E)])
            state["layers.0.mlp.experts.gate_up_proj"] = torch.cat([gate, up], dim=1)
            state["layers.0.mlp.experts.down_proj"] = torch.stack([experts["down_proj"][e] for e in range(E)])
        own = self.state_dict()
        for k, t in state.items():
            if k not in own:
                raise KeyError(f"unexpected key {k}")
            if own[k].shape != t.shape:
                raise ValueError(f"shape mismatch {k}: {tuple(t.shape)} vs {tuple(own[k].shape)}")
        res = self.load_state_dict(state, strict=False)
        allowed_missing = {"embed_tokens.weight"}
        bad_missing = [k for k in res.missing_keys if k not in allowed_missing]
        if bad_missing:
            raise KeyError(f"missing keys after load: {bad_missing[:8]}")
        self.embed_loaded = "embed_tokens.weight" in state
        return sorted(state)

    # -- forward -------------------------------------------------------------
    def forward(self, target_multi_hidden: torch.Tensor, next_token_ids: torch.Tensor,
                positions: torch.Tensor, kv: Optional[KVState] = None):
        """One draft-model call.

        target_multi_hidden: [B, T, HC*H] pre-final-mixer multi-stream state (for step 0 this is
            the target hidden; for later steps the previous call's multi_hidden_out).
        next_token_ids: [B, T] MTP input ids (already shifted, see unroll()).
        positions: [B, T] absolute positions.
        kv: previous KVState (all earlier positions of the same sequences) or None.
        Returns (sample_hidden [B,T,H], multi_hidden_out [B,T,HC*H], new_kv).
        """
        cfg = self.cfg
        B, T = next_token_ids.shape
        H, HC, D = cfg.hidden_size, cfg.hc_count, cfg.mtp_width

        # mtp.py:248-250  embedding branch: pre-norm -> fc_embedding
        emb = self.embed_tokens(next_token_ids)
        emb = self.pre_fc_norm_embedding(emb)
        prev_block = self.fc_embedding(emb).reshape(B * T, H)
        # mtp.py:252-262  hidden branch: flatten HC*H -> one GemmaRMSNorm over the full D (not grouped!)
        hid = self.pre_fc_norm_hidden(target_multi_hidden.reshape(B, T, D))
        hid = self.fc_hidden(hid.reshape(B, T, HC, H))  # shared fc_hidden applied to every stream
        hidden = hid.reshape(B * T, D)

        layer = self.layers[0]
        # mtp.py:/model.py:293-298 -> attn_hc.combine_and_mix(hidden, prev_block_output, None)
        hidden, block_in, inj = layer.attn_hyper_connection.combine_and_mix(hidden, prev_block, None)
        attn_out, new_kv = layer.self_attn(block_in.reshape(B, T, H), positions, kv)  # model.py:313-316
        # model.py:325-327 -> mlp_hc.combine_and_mix(hidden, attn_out, injection_attn)
        hidden, block_in2, inj2 = layer.mlp_hyper_connection.combine_and_mix(hidden, attn_out.reshape(B * T, H), inj)
        mlp_out = layer.mlp(block_in2)  # model.py:328-329
        # mtp.py:327-330 -> final mixer: combine with mlp injection, collapse to single stream
        multi, sample, _ = self.hyper_connection_mixer.combine_and_mix(hidden, mlp_out, inj2)
        return sample.reshape(B, T, H), multi.reshape(B, T, D), new_kv

    @torch.no_grad()
    def unroll(self, target_multi_hidden: torch.Tensor, token_ids: torch.Tensor, k: int = 4,
               lm_head_rows: Optional[torch.Tensor] = None, id_table: Optional[torch.Tensor] = None,
               next_token_ids: Optional[torch.Tensor] = None, positions: Optional[torch.Tensor] = None,
               kv: Optional[KVState] = None):
        """Reproduce the k-step draft chain of vLLM's SpecDecodeBaseProposer.propose.

        Inputs (one sequence per batch row, all rows the same length T):
          target_multi_hidden [B, T, HC*H]  target multi-stream hidden of every verified position
          token_ids           [B, T]        target token id at each position (target_token_ids)
          next_token_ids      [B]           target-sampled token that follows position T-1
          positions           [B, T]        default arange(T)
          kv                  KVState|None  draft KV of earlier rounds (default: none)
          lm_head_rows        [V_sub, H]    rows of the (possibly trimmed) lm_head
          id_table            [V_sub]       subset index -> target token id (None: raw index)

        Chain (vLLM llm_base_proposer.py):
          step 0  input_ids[i] = token_ids[i+1], input_ids[T-1] = next_token_ids   (L839-847)
                  one forward over all T rows, positions unchanged                  (L600, L732-754)
                  draft_0 = argmax(sample_hidden[T-1] @ lm_head_rows^T)             (L615 + sampler)
                  carry multi_hidden[T-1] forward                                   (L643)
          step j  (1..k-1) input id = draft_{j-1}, hidden = carried multi_hidden,
                  position = pos[T-1] + j, one row per sequence                      (L689-693, L701)
                  KV: the row's own K/V is written before attention (kv appended); attention
                  reads every earlier row of the sequence, i.e. step-0 rows 0..T-1 and draft rows
                  of steps 1..j-1 (see KV_BEHAVIOUR below).                          (L754)

        Returns dict: draft_ids [B,k] (mapped through id_table), logits list of k [B,V_sub],
        sample_hidden [B,k,H], and kv (final KVState).

        KV_BEHAVIOUR: steps 1..k-1 do NOT re-run the target and do not re-attend the
        target's hidden states; they attend to the draft model's own KV cache, which
        step 0 populated for all T verified rows (L600, L754 via the per-step slot mapping
        L701 -> eagle_step_update_slot_mapping_and_metadata, v1/spec_decode/utils.py:141)
        plus one new row per earlier draft step.  No sliding window, no re-compaction.
        """
        B, T = token_ids.shape
        if lm_head_rows is None:
            raise ValueError("lm_head_rows is required to produce draft ids")
        if positions is None:
            positions = torch.arange(T, device=token_ids.device).expand(B, T)
        if next_token_ids is None:
            raise ValueError("next_token_ids is required")
        ids0 = torch.empty_like(token_ids)
        ids0[:, : T - 1] = token_ids[:, 1:]
        ids0[:, T - 1] = next_token_ids
        # kv: draft KV of earlier verify rounds (positions < first row of this batch); the step-0 rows
        # are appended after it, exactly as the paged MTP cache holds them.
        sample, multi, kv = self.forward(target_multi_hidden, ids0, positions, kv=kv)

        sample_list = [sample[:, T - 1]]
        draft_raw = []
        logits = []

        def _head(h):
            lg = F.linear(h.to(lm_head_rows.dtype), lm_head_rows)  # [B, V_sub] (hook: F.linear in model dtype)
            return lg, lg.argmax(dim=-1)  # first max wins (ids ascending -> lowest token id)

        lg, idx = _head(sample[:, T - 1])
        logits.append(lg)
        draft_raw.append(idx)
        carried = multi[:, T - 1 : T]  # llm_base_proposer.py:643 hidden_states[token_indices_to_sample]
        last_pos = positions[:, T - 1 : T]
        for j in range(1, k):
            tok_target = id_table[draft_raw[-1]] if id_table is not None else draft_raw[-1]
            s_j, m_j, kv = self.forward(carried, tok_target[:, None], last_pos + j, kv=kv)
            sample_list.append(s_j[:, 0])
            lg, idx = _head(s_j[:, 0])
            logits.append(lg)
            draft_raw.append(idx)
            carried = m_j
        raw = torch.stack(draft_raw, dim=1)
        draft_ids = id_table[raw] if id_table is not None else raw
        return {
            "draft_ids": draft_ids,
            "logits": logits,
            "sample_hidden": torch.stack(sample_list, dim=1),
            "kv": kv,
        }
