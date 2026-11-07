"""CPU unit tests for mtp_torch.py.  Run:  python -I test_mtp_torch.py   (or pytest)

(a) name / shape loading (synthetic file with the exact expected names; plus the
    real BF16 non-expert tensors from the nvidia checkpoint when present)
(b) op parity against vLLM's own torch code where a CPU reference exists
(c) MoE block vs a naive per-token expert loop
(d) full forward on T=8 (finite, right shapes); KV-incremental consistency; unroll smoke;
    KV budget guard
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mtp_torch as M  # noqa: E402

NVIDIA_DIR = "/u701/j692wu/models/nvidia-Qwen3.8-Flash-Next-NVFP4"
NVIDIA_CONFIG = os.path.join(NVIDIA_DIR, "config.json")
RESULTS: list[tuple[str, str, str]] = []  # (test, status, detail)
REFERENCE_NOTES: list[tuple[str, str]] = []  # (op, reference status)

torch.manual_seed(0)


class _vllm_config:
    """CustomOp constructors (GemmaRMSNorm, get_rope) need a current vLLM config."""

    def __enter__(self):
        from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

        self._cm = set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu")))
        self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def _vllm_ref():
    """Import vLLM reference modules lazily; returns dict or None per symbol."""
    refs = {}
    try:
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm

        refs["GemmaRMSNorm"] = GemmaRMSNorm
    except Exception as exc:  # pragma: no cover
        refs["GemmaRMSNorm_err"] = repr(exc)
    try:
        from vllm.models.qwen4_exp.common.hyperconnection import (
            GatedResidual,
            GroupedGemmaRMSNorm,
            HyperConnectionConfig,
        )

        refs["GatedResidual"] = GatedResidual
        refs["GroupedGemmaRMSNorm"] = GroupedGemmaRMSNorm
        refs["HyperConnectionConfig"] = HyperConnectionConfig
    except Exception as exc:  # pragma: no cover
        refs["HC_err"] = repr(exc)
    try:
        from vllm.model_executor.layers.rotary_embedding import get_rope

        refs["get_rope"] = get_rope
    except Exception as exc:  # pragma: no cover
        refs["get_rope_err"] = repr(exc)
    return refs


def check(name):
    def deco(fn):
        def run():
            try:
                detail = fn()
                RESULTS.append((name, "PASS", detail or ""))
            except Exception as exc:  # noqa: BLE001
                RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))

        run.__name__ = fn.__name__
        return run

    return deco


def small_cfg(**kw):
    """Real dims except the expert count and vocab (full 512 experts ~ 5 GB fp32 on CPU)."""
    base = dict(num_experts=16, vocab_size=1000)
    base.update(kw)
    return M.MTPConfig(**base)


def write_config_json(path, cfg: M.MTPConfig):
    t = dict(
        hidden_size=cfg.hidden_size, hc_count=cfg.hc_count, hc_lowrank=cfg.hc_lowrank,
        num_attention_heads=cfg.num_heads, num_key_value_heads=cfg.num_kv_heads, head_dim=cfg.head_dim,
        rope_parameters={"partial_rotary_factor": cfg.partial_rotary_factor, "rope_theta": cfg.rope_theta,
                         "rope_type": "default", "mrope_section": [11, 11, 10], "mrope_interleaved": True},
        rms_norm_eps=cfg.rms_norm_eps, num_experts=cfg.num_experts, num_experts_per_tok=cfg.top_k,
        moe_intermediate_size=cfg.moe_intermediate, shared_expert_intermediate_size=cfg.shared_intermediate,
        norm_topk_prob=cfg.norm_topk_prob, indexer_budget=cfg.indexer_budget, indexer_n_heads=cfg.indexer_heads,
        indexer_head_dim=cfg.indexer_head_dim, indexer_kv_heads=cfg.indexer_kv_heads,
        vocab_size=cfg.vocab_size, mtp_num_hidden_layers=1, output_gate_type="sigmoid",
    )
    with open(path, "w") as f:
        json.dump({"text_config": t}, f)


def random_state(cfg, layout):
    shapes = M.expected_mtp_names(cfg, layout)
    return {k: torch.randn(v) * 0.02 for k, v in shapes.items()}


def init_random(model, std=0.02):
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, std)


# ---------------------------------------------------------------------------
# (a) loading
# ---------------------------------------------------------------------------
@check("a1_load_fused_31_tensors_no_missing_or_unexpected")
def test_a1():
    from safetensors.torch import save_file

    cfg = small_cfg()
    with tempfile.TemporaryDirectory() as td:
        cj = os.path.join(td, "config.json")
        write_config_json(cj, cfg)
        state = random_state(cfg, "fused")
        assert len(state) == 31, len(state)  # 29 non-expert + 2 fused expert tensors
        # embed_tokens rides in the same file in the real checkpoint (model-tail shard)
        state["model.language_model.embed_tokens.weight"] = torch.randn(cfg.vocab_size, cfg.hidden_size) * 0.02
        wf = os.path.join(td, "model-tail.safetensors")
        save_file(state, wf)
        model = M.MTPDraft.from_safetensors(wf, cj, dtype=torch.float32)
        assert model.embed_loaded
        # every expected name present with exact value
        sd = model.state_dict()
        assert torch.equal(sd["layers.0.mlp.experts.down_proj"], state["mtp.layers.0.mlp.experts.down_proj"])
        assert torch.equal(sd["fc_hidden.weight"], state["mtp.fc_hidden.weight"])
        # a missing tensor must be refused
        bad = dict(state)
        del bad["mtp.layers.0.self_attn.o_proj.weight"]
        save_file(bad, wf)
        try:
            M.MTPDraft.from_safetensors(wf, cj)
        except ValueError as exc:
            assert "missing" in str(exc)
        else:
            raise AssertionError("missing tensor not detected")
    return "31 tensors (29 non-expert + 2 fused experts), exact values; missing name rejected"


@check("a2_load_per_expert_layout")
def test_a2():
    from safetensors.torch import save_file

    cfg = small_cfg()
    with tempfile.TemporaryDirectory() as td:
        cj = os.path.join(td, "config.json")
        write_config_json(cj, cfg)
        state = random_state(cfg, "per_expert")
        wf = os.path.join(td, "m.safetensors")
        save_file(state, wf)
        model = M.MTPDraft.from_safetensors(wf, cj, dtype=torch.float32)
        E = cfg.num_experts
        ref_gate = state[f"mtp.layers.0.mlp.experts.{E - 1}.gate_proj.weight"]
        got = model.layers[0].mlp.experts.gate_up_proj[E - 1, : cfg.moe_intermediate]
        assert torch.equal(got, ref_gate)
    return f"per-expert names stacked into fused [E={E},2I,H]; {len(state)} tensors"


@check("a3_real_checkpoint_nonexpert_names_and_shapes")
def test_a3():
    idx_path = os.path.join(NVIDIA_DIR, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        return "SKIPPED (nvidia checkpoint not present)"
    cfg = M.MTPConfig.from_config_json(NVIDIA_CONFIG)
    weight_map = json.load(open(idx_path))["weight_map"]
    shapes_expected = M.expected_mtp_names(cfg, "per_expert")
    n = 0
    for name in M.NON_EXPERT_KEYS:
        full = M.MTP_PREFIX + name
        shard = weight_map[full]
        with open(os.path.join(NVIDIA_DIR, shard), "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
        assert hdr[full]["dtype"] == "BF16", (full, hdr[full]["dtype"])
        assert tuple(hdr[full]["shape"]) == tuple(shapes_expected[full]), (full, hdr[full]["shape"])
        n += 1
    return f"{n}/29 non-expert tensors present in the nvidia checkpoint, BF16, shapes match config"


# ---------------------------------------------------------------------------
# (b) op parity
# ---------------------------------------------------------------------------
REFS = _vllm_ref()


@check("b1_gemma_rmsnorm_vs_vllm")
def test_b1():
    if "GemmaRMSNorm" not in REFS:
        REFERENCE_NOTES.append(("GemmaRMSNorm / gemma_rms_norm", "NO REFERENCE (vLLM import failed)"))
        return "no reference"
    out = []
    for dtype, tol in ((torch.float32, 1e-6), (torch.bfloat16, 1e-3)):
        D = 2560
        with _vllm_config():
            ref = REFS["GemmaRMSNorm"](D, eps=1e-6)
        with torch.no_grad():
            ref.weight.copy_(torch.randn(D) * 0.1)
        x = torch.randn(16, D, dtype=dtype) * 3
        ours = M.gemma_rms_norm(x, ref.weight.detach(), 1e-6)
        theirs = ref.forward_native(x)
        d = (ours.float() - theirs.float()).abs().max().item()
        out.append(f"{dtype}: max|d|={d:.2e}")
        assert d <= tol, (dtype, d)
    REFERENCE_NOTES.append(("GemmaRMSNorm / gemma_rms_norm", "vLLM GemmaRMSNorm.forward_native (CPU)"))
    return "; ".join(out)


@check("b2_grouped_gemma_rmsnorm_vs_vllm")
def test_b2():
    if "GroupedGemmaRMSNorm" not in REFS:
        REFERENCE_NOTES.append(("grouped_gemma_rms_norm", "NO REFERENCE"))
        return "no reference"
    out = []
    H, HC = 2560, 4
    for dtype, tol in ((torch.float32, 1e-6), (torch.bfloat16, 1e-3)):
        ref = REFS["GroupedGemmaRMSNorm"](HC * H, eps=1e-6, group_size=H, dtype=dtype)
        with torch.no_grad():
            ref.weight.copy_(torch.randn(HC * H) * 0.1)
        x = torch.randn(9, HC * H, dtype=dtype) * 2
        ours = M.grouped_gemma_rms_norm(x, ref.weight, 1e-6, HC)
        theirs = ref.forward(x)
        d = (ours.float() - theirs.float()).abs().max().item()
        out.append(f"{dtype}: max|d|={d:.2e}")
        assert d <= tol, (dtype, d)
    REFERENCE_NOTES.append(("grouped_gemma_rms_norm", "vLLM common GroupedGemmaRMSNorm.forward (CPU)"))
    return "; ".join(out)


def _hc_pair(cfg, dtype, use_real):
    """Our HyperMixer and vLLM common GatedResidual with identical weights."""
    H, HC, L = cfg.hidden_size, cfg.hc_count, cfg.hc_lowrank
    D = HC * H
    ours = M.HyperMixer(cfg, with_inject=True).to(dtype)
    ref_cfg = REFS["HyperConnectionConfig"](
        hc_count=HC, hidden_size=H, params_dtype=dtype, hc_lowrank=L, rms_norm_eps=cfg.rms_norm_eps,
        hc_per_branch_norm=True,
    )
    ref = REFS["GatedResidual"](ref_cfg, use_mix=True, use_combine=True).to(dtype)
    real = None
    if use_real and os.path.exists(os.path.join(NVIDIA_DIR, "model-00009-of-00010.safetensors")):
        from safetensors import safe_open

        real = {}
        with safe_open(os.path.join(NVIDIA_DIR, "model-00009-of-00010.safetensors"), "pt") as f:
            for k in ("layers.0.attn_hyper_connection.hc_norm.weight", "layers.0.attn_hyper_connection.input_mix_weight_down.weight",
                      "layers.0.attn_hyper_connection.input_mix_weight_up.weight", "layers.0.attn_hyper_connection.block_inject_weight.weight"):
                real[k] = f.get_tensor(M.MTP_PREFIX + k)
            real["__src__"] = "real"
    with torch.no_grad():
        sd = {
            "hc_norm.weight": torch.randn(D) * 0.1,
            "input_mix_weight_down.weight": torch.randn(L, D) / D**0.5,
            "input_mix_weight_up.weight": torch.randn(D, L) / L**0.5,
            "block_inject_weight.weight": torch.randn(HC, D) / D**0.5,
        }
        if real is not None:
            sd = {
                "hc_norm.weight": real["layers.0.attn_hyper_connection.hc_norm.weight"],
                "input_mix_weight_down.weight": real["layers.0.attn_hyper_connection.input_mix_weight_down.weight"],
                "input_mix_weight_up.weight": real["layers.0.attn_hyper_connection.input_mix_weight_up.weight"],
                "block_inject_weight.weight": real["layers.0.attn_hyper_connection.block_inject_weight.weight"],
            }
        for k, v in sd.items():
            tgt = ours if k.startswith(("hc_norm",)) else ours
            if k == "hc_norm.weight":
                ours.hc_norm.weight.copy_(v.to(dtype))
                ref.hc_norm.weight.copy_(v.to(dtype))
            else:
                mod_name, _ = k.split(".")
                getattr(ours, mod_name).weight.copy_(v.to(dtype))
                getattr(ref, mod_name).weight.copy_(v.to(dtype))
    return ours, ref, (real is not None)


@check("b3_hc_mixer_vs_vllm_gatedresidual")
def test_b3():
    if "GatedResidual" not in REFS:
        REFERENCE_NOTES.append(("HyperMixer.combine/norm_mix/_gate", "NO REFERENCE"))
        return "no reference"
    cfg = small_cfg()
    H, HC = cfg.hidden_size, cfg.hc_count
    D = HC * H
    out = []
    for dtype, tol, use_real in ((torch.float32, 1e-5, True), (torch.bfloat16, 1e-2, True)):
        ours, ref, real = _hc_pair(cfg, dtype, use_real)
        res = torch.randn(7, D, dtype=dtype) * 2
        blk = torch.randn(7, H, dtype=dtype)
        # mix(): ours norm_mix vs vLLM mix()
        b_o, xn_o, inj_o = ours.norm_mix(res)
        b_r, (res_r, xn_r) = ref.mix(res)
        d_mix = (b_o.float() - b_r.float()).abs().max().item()
        d_xn = (xn_o.float() - xn_r.float()).abs().max().item()
        # combine(): ours combine with the injection from our own mix vs vLLM combine with its mix residuals
        new_o = ours.combine(res, blk, inj_o)
        new_r = ref.combine(blk, (res_r, xn_r))
        d_comb = (new_o.float() - new_r.float()).abs().max().item()
        # full combine_and_mix (layer boundary): compare residual + block output of the next mix
        new2, blk2, inj2 = ours.combine_and_mix(res, blk, inj_o)
        b2_r, _ = ref.mix(new_r)
        d_cam = (blk2.float() - b2_r.float()).abs().max().item()
        d_cam_res = (new2.float() - new_r.float()).abs().max().item()
        scale = max(new_r.float().abs().max().item(), 1e-6)
        # relative to the activation scale: the torch reference rounds intermediates to bf16
        # (residual + block*inj in bf16), so bf16 gets ~1 ulp of slack
        rel = max(d_mix, d_xn, d_comb, d_cam, d_cam_res) / scale
        out.append(f"{dtype} real={real}: mix={d_mix:.1e} xn={d_xn:.1e} combine={d_comb:.1e} "
                   f"combine_and_mix(block)={d_cam:.1e} combine_and_mix(res)={d_cam_res:.1e} rel={rel:.1e}")
        assert rel <= tol, (dtype, rel)
    REFERENCE_NOTES.append(("HyperMixer.norm_mix/_gate (hc_silu+hc_gate_mix)", "vLLM common GatedResidual.mix (torch)"))
    REFERENCE_NOTES.append(("HyperMixer.combine (hc_combine / hc_combine_norm injection)", "vLLM common GatedResidual.combine (torch)"))
    REFERENCE_NOTES.append(("HyperMixer.combine_and_mix", "composition of the two above; no separate vLLM torch ref"))
    return "; ".join(out)


@check("b4_partial_neox_rope_vs_vllm")
def test_b4():
    if "get_rope" not in REFS:
        REFERENCE_NOTES.append(("apply_partial_neox_rope", "NO REFERENCE"))
        return "no reference"
    cfg = small_cfg()
    with _vllm_config():
        rope = REFS["get_rope"](
            head_size=cfg.head_dim, max_position=4096, is_neox_style=True, dtype=torch.float32,
            rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta,
                             "partial_rotary_factor": cfg.partial_rotary_factor,
                             "mrope_section": [11, 11, 10], "mrope_interleaved": True},
        )
    T = 9
    pos = torch.arange(3, 3 + T)
    q = torch.randn(T, cfg.num_heads, cfg.head_dim)
    k = torch.randn(T, cfg.num_kv_heads, cfg.head_dim)
    qr, kr = rope.forward_native(pos, q.reshape(T, -1).clone(), k.reshape(T, -1).clone())
    ours_q = M.apply_partial_neox_rope(q[None], pos[None], cfg.rotary_dim, cfg.rope_theta)[0]
    ours_k = M.apply_partial_neox_rope(k[None], pos[None], cfg.rotary_dim, cfg.rope_theta)[0]
    d = max((ours_q.reshape(T, -1) - qr).abs().max().item(), (ours_k.reshape(T, -1) - kr).abs().max().item())
    assert d < 1e-5, d
    REFERENCE_NOTES.append(("apply_partial_neox_rope", "vLLM get_rope(MRotaryEmbedding).forward_native, 1-D positions"))
    return f"max|d|={d:.2e} (rotary_dim={cfg.rotary_dim})"


@check("b5_triton_hc_kernels_no_cpu_reference")
def test_b5():
    REFERENCE_NOTES.append(("hc_down_silu fused path (cute_dsl, sm90)", "NO CPU REFERENCE; not mirrored (we use the unfused path)"))
    return "documented only"


# ---------------------------------------------------------------------------
# (c) MoE
# ---------------------------------------------------------------------------
@check("c1_moe_vs_naive_per_token_loop")
def test_c1():
    cfg = small_cfg()
    moe = M.MoEBlock(cfg)
    init_random(moe, 0.05)
    N = 12
    x = torch.randn(N, cfg.hidden_size) * 0.5
    got = moe(x)
    out = torch.zeros_like(x)
    for t in range(N):
        xt = x[t]
        logits = (moe.gate.weight.to(torch.bfloat16) @ xt.to(torch.bfloat16)).float()  # bf16 router GEMM, as vLLM
        probs = torch.softmax(logits, dim=-1)
        top = torch.argsort(probs, descending=True)[: cfg.top_k]
        w = probs[top] / probs[top].sum()
        acc = torch.zeros(cfg.hidden_size)
        for wi, e in zip(w, top):
            g = moe.experts.gate_up_proj[e, : cfg.moe_intermediate] @ xt
            u = moe.experts.gate_up_proj[e, cfg.moe_intermediate :] @ xt
            acc += wi * (moe.experts.down_proj[e] @ (torch.nn.functional.silu(g) * u))
        s = moe.shared_expert
        sh = s.down_proj(torch.nn.functional.silu(s.gate_proj(xt)) * s.up_proj(xt))
        sh = torch.sigmoid(moe.shared_expert_gate.weight.float() @ xt.float()) * sh
        out[t] = acc + sh
    d = (got - out).abs().max().item()
    assert d < 1e-5, d
    return f"fp32 max|d|={d:.2e} over {N} tokens, top-{cfg.top_k} of {cfg.num_experts}"


# ---------------------------------------------------------------------------
# (d) forward / unroll
# ---------------------------------------------------------------------------
@check("d1_full_forward_T8_shapes_finite")
def test_d1():
    cfg = small_cfg()
    model = M.MTPDraft(cfg).float()
    init_random(model, 0.02)
    B, T = 2, 8
    h = torch.randn(B, T, cfg.mtp_width) * 0.5
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    pos = torch.arange(T).expand(B, T)
    sample, multi, kv = model(h, ids, pos)
    assert sample.shape == (B, T, cfg.hidden_size), sample.shape
    assert multi.shape == (B, T, cfg.mtp_width), multi.shape
    assert kv.k.shape == (B, T, cfg.num_kv_heads, cfg.head_dim)
    assert torch.isfinite(sample).all() and torch.isfinite(multi).all()
    return f"sample {tuple(sample.shape)}, multi {tuple(multi.shape)}, finite"


@check("d2_kv_incremental_equals_full")
def test_d2():
    cfg = small_cfg()
    model = M.MTPDraft(cfg).float()
    init_random(model, 0.02)
    B, T = 1, 8
    h = torch.randn(B, T, cfg.mtp_width) * 0.5
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    pos = torch.arange(T).expand(B, T)
    s_full, m_full, _ = model(h, ids, pos)
    s1, m1, kv = model(h[:, :5], ids[:, :5], pos[:, :5])
    s2, m2, _ = model(h[:, 5:], ids[:, 5:], pos[:, 5:], kv=kv)
    d = max((torch.cat([s1, s2], 1) - s_full).abs().max().item(), (torch.cat([m1, m2], 1) - m_full).abs().max().item())
    assert d < 1e-5, d
    return f"max|d|={d:.2e} (prefix 5 + continuation 3 == full 8)"


@check("d3_unroll_k4_smoke_and_chain_positions")
def test_d3():
    cfg = small_cfg()
    model = M.MTPDraft(cfg).float()
    init_random(model, 0.02)
    B, T, k, V_sub = 2, 6, 4, 50
    h = torch.randn(B, T, cfg.mtp_width) * 0.5
    tok = torch.randint(0, cfg.vocab_size, (B, T))
    nxt = torch.randint(0, cfg.vocab_size, (B,))
    rows = torch.randn(V_sub, cfg.hidden_size) * 0.05
    id_table = torch.sort(torch.randperm(cfg.vocab_size)[:V_sub]).values
    res = model.unroll(h, tok, k=k, lm_head_rows=rows, id_table=id_table, next_token_ids=nxt)
    assert res["draft_ids"].shape == (B, k)
    assert len(res["logits"]) == k and res["logits"][0].shape == (B, V_sub)
    assert res["sample_hidden"].shape == (B, k, cfg.hidden_size)
    assert torch.isin(res["draft_ids"], id_table).all()
    assert res["kv"].k.shape[1] == T + k - 1, res["kv"].k.shape
    assert torch.equal(res["kv"].pos[0], torch.arange(T + k - 1))
    return f"draft_ids {tuple(res['draft_ids'].shape)}, KV length {T + k - 1} (T + k-1 rows)"


@check("d4_kv_budget_guard")
def test_d4():
    cfg = small_cfg(indexer_budget=16)
    model = M.MTPDraft(cfg).float()
    B = 1
    kv = M.KVState(torch.zeros(B, 16, cfg.num_kv_heads, cfg.head_dim), torch.zeros(B, 16, cfg.num_kv_heads, cfg.head_dim),
                   torch.arange(16)[None])
    try:
        model(torch.zeros(B, 1, cfg.mtp_width), torch.zeros(B, 1, dtype=torch.long), torch.tensor([[16]]), kv=kv)
    except ValueError as exc:
        assert "indexer_budget" in str(exc)
        return "S=17 > budget=16 raises"
    raise AssertionError("budget not enforced")


# ---------------------------------------------------------------------------
# (e) capture hook and parity script (CPU, synthetic)
# ---------------------------------------------------------------------------
@check("e1_capture_hook_wraps_and_caps")
def test_e1():
    import swift_mtp_capture as C

    os.environ.pop(C.ENV_DIR, None)
    assert C.install() is False, "hook must be inert without SWIFT_MTP_CAPTURE_DIR"
    H, HC, k = 8, 2, 3

    class FakeMTP:
        def forward(self, input_ids, positions, hidden_states=None, intermediate_tensors=None,
                    inputs_embeds=None, spec_step_idx=0):
            return hidden_states[..., :H] * 2, hidden_states + 1

    class FakeProposer:
        def __init__(self):
            self.model = FakeMTP()

        def propose(self, num_speculative_tokens, target_token_ids, target_positions, target_hidden_states,
                    next_token_ids, token_indices_to_sample=None, common_attn_metadata=None):
            h, m = self.model.forward(target_token_ids[None], target_positions[None], target_hidden_states[None])
            for _ in range(num_speculative_tokens - 1):
                h, m = self.model.forward(target_token_ids[:1, None], target_positions[:1, None], m[:, -1:])
            return torch.zeros(1, num_speculative_tokens, dtype=torch.long)

    with tempfile.TemporaryDirectory() as td:
        C._STATE.update(dir=td, max_bytes=1 << 30, bytes=0, rounds=0, skipped=0, sample_rounds=1,
                        guard=False, stack=[], installed=True)
        FakeMTP.forward = C._wrap_forward(FakeMTP.forward)
        FakeProposer.propose = C._wrap_propose(FakeProposer.propose)
        p = FakeProposer()
        args = dict(num_speculative_tokens=k, target_token_ids=torch.arange(5), target_positions=torch.arange(5),
                    target_hidden_states=torch.randn(5, HC * H), next_token_ids=torch.tensor([9]),
                    token_indices_to_sample=torch.tensor([4]),
                    common_attn_metadata=type("CAM", (), {"query_start_loc": torch.tensor([0, 5]),
                                                          "seq_lens": torch.tensor([5]), "num_actual_tokens": 5,
                                                          "max_query_len": 5})())
        p.propose(**args)
        files = sorted(os.listdir(td))
        assert files == ["mtp_round_%d_000000.pt" % os.getpid()], files
        rec = torch.load(os.path.join(td, files[0]), weights_only=False)
        assert len(rec["calls"]) == 1 + (k - 1), len(rec["calls"])
        assert "sample_hidden" in rec["calls"][0] and "multi_hidden" in rec["calls"][0]
        assert torch.equal(rec["propose_args"]["target_token_ids"], torch.arange(5))
        assert rec["propose_args"]["common_attn_metadata"]["query_start_loc"].tolist() == [0, 5]
        assert rec["draft_token_ids"].shape == (1, k)
        # size cap: a second round is skipped once the cap is exceeded
        C._STATE["max_bytes"] = 1
        p.propose(**args)
        assert len(os.listdir(td)) == 1 and C._STATE["skipped"] == 1
        C._STATE.update(dir=None, installed=False, rounds=0, bytes=0, skipped=0, stack=[])
    return "calls=1+(k-1) recorded with sample outputs; cap skips later rounds; inert without env"


@check("e2_parity_check_smoke_single_round")
def test_e2():
    import numpy as np
    from safetensors.torch import save_file

    import parity_check as PC

    cfg = small_cfg(vocab_size=200)
    with tempfile.TemporaryDirectory() as td:
        cj = os.path.join(td, "config.json")
        write_config_json(cj, cfg)
        state = random_state(cfg, "fused")
        state["model.language_model.embed_tokens.weight"] = torch.randn(cfg.vocab_size, cfg.hidden_size) * 0.02
        state["lm_head.weight"] = torch.randn(cfg.vocab_size, cfg.hidden_size) * 0.02
        wf = os.path.join(td, "model-tail.safetensors")
        save_file({k_: v.contiguous() for k_, v in state.items()}, wf)
        ids = np.sort(np.random.RandomState(0).choice(cfg.vocab_size, 40, replace=False)).astype(np.int32)
        vf = os.path.join(td, "vocab.npy")
        np.save(vf, ids)
        # reference run (same weights, bf16, same chain as the capture would record)
        ref = M.MTPDraft.from_safetensors(wf, cj, dtype=torch.bfloat16)
        W_sub = state["lm_head.weight"].to(torch.bfloat16)[torch.from_numpy(ids.astype(np.int64))]
        T, k = 5, 3
        tok = torch.randint(0, cfg.vocab_size, (T,))
        nxt = torch.randint(0, cfg.vocab_size, (1,))
        pos = torch.arange(T)
        hid = (torch.randn(T, cfg.mtp_width) * 0.5).to(torch.bfloat16)
        out = ref.unroll(hid[None], tok[None], k=k, lm_head_rows=W_sub, id_table=torch.from_numpy(ids.astype(np.int64)),
                         next_token_ids=nxt, positions=pos[None])
        # build the capture the hook would write (call 0 = all rows, calls 1.. = one row per draft step)
        ids0 = torch.empty_like(tok)
        ids0[:-1] = tok[1:]
        ids0[-1] = nxt[0]
        s0, m0, _ = ref(hid[None], ids0[None], pos[None])
        calls = [{"input_ids": ids0, "positions": pos, "hidden_in": hid, "sample_hidden": s0[0], "multi_hidden": m0[0]}]
        draft = out["draft_ids"][0]
        # steps 1..k-1: parity only reads their input ids and positions (chain consistency)
        for j in range(1, k):
            calls.append({"input_ids": torch.tensor([int(draft[j - 1])]), "positions": torch.tensor([T - 1 + j])})
        rec = {
            "meta": {}, "propose_args": {
                "target_token_ids": tok, "next_token_ids": nxt, "target_positions": pos,
                "target_hidden_states": hid, "token_indices_to_sample": torch.tensor([T - 1]),
                "common_attn_metadata": {"query_start_loc": torch.tensor([0, T]), "seq_lens": torch.tensor([T])},
                "num_speculative_tokens": k},
            "calls": calls, "draft_token_ids": draft[None].clone(),
        }
        cap_dir = os.path.join(td, "caps")
        os.makedirs(cap_dir)
        torch.save(rec, os.path.join(cap_dir, "mtp_round_1_000000.pt"))
        out_json = os.path.join(td, "rep.json")
        argv = sys.argv
        sys.argv = ["parity_check.py", "--captures", cap_dir, "--config", cj, "--weights", wf, "--vocab", vf,
                    "--device", "cpu", "--out", out_json]
        try:
            import io
            import contextlib

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                PC.main()
        finally:
            sys.argv = argv
        rep = json.load(open(out_json))["report"]
        assert rep["full_chain_agreement"] == 1.0, rep
        assert rep["step0_input_mismatch"] == "0/%d" % T, rep
        assert rep["chain_input_position_mismatch"].startswith("0/"), rep
    return f"per-step agreement {rep['top1_agreement_per_step']}, chain/inputs consistent"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    width = max(len(n) for n, _, _ in RESULTS)
    n_fail = 0
    for name, status, detail in RESULTS:
        first = detail.splitlines()[0] if detail else ""
        print(f"{status:4s}  {name.ljust(width)}  {first}")
        if status == "FAIL":
            n_fail += 1
            print(detail)
    print("\nReference coverage:")
    for op, note in REFERENCE_NOTES:
        print(f"  {op}: {note}")
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
