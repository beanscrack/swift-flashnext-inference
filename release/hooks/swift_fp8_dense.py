"""swift_fp8_dense.py - weight-only FP8 (e4m3) for Swift/Qwen4Exp BF16 dense layers on vLLM 0.31, without editing vLLM.

W8A16: weights stored as e4m3 with fp32 scales (per output row, or per row per GROUP K-columns), activations stay
BF16, fp32 accumulation.  Halves the bytes the bandwidth-bound decode GEMMs read.  NOT lossless: greedy outputs will
differ from BF16 (validate quality before deploying).

How it hooks in (import hook on vllm.models.qwen4_exp.nvidia.low_latency_gemm, like swift_skinny_patch):
  * enable_qwen4_exp_low_latency_gemm(module, dtype) is wrapped.  It runs in the model constructors (target and MTP),
    before weights load.  The wrapper gives the selected Linear modules a W8LinearMethod.
  * W8LinearMethod.process_weights_after_loading quantizes the loaded BF16 weight and REPLACES layer.weight with the
    e4m3 tensor (scale in layer.w8_scale).  vLLM runs this before its memory profile, so the freed memory goes to the
    KV pool.  Code that reads layer.weight later (the L2-prefetch hook) sees the FP8 bytes.
  * apply(): M rows with a plan entry (or M <= 32 with a default config) -> Triton FP8 GEMM (fp8_gemv.w8_gemm);
    larger M (prefill) -> dequantize into a temporary BF16 matrix and call F.linear.

Env:
  SWIFT_FP8=1                       enable (otherwise a no-op)
  SWIFT_FP8_TARGETS=qkvz,gdn_out,qsa_qkv,qsa_o   categories (also: shared, mtp_*  e.g. mtp_qkvz; "mtp" = all for MTP)
  SWIFT_FP8_GROUP=0                 0 = per-row scales, 128 = per-row per-128-K-group scales
  SWIFT_FP8_PLAN=plan.json          {"NxK": {"M": {"kind":..,"block_n":..,...}}} from fp8_bench.py (optional)
  SWIFT_FP8_WARM_MS=1,2,3,4,5,10,15,20   token counts to compile + self-check at conversion time
  SWIFT_FP8_MAX_M=32                above this M use dequantize + F.linear
  SWIFT_FP8_STRICT=1                abort start-up if nothing was selected or a self-check fails
"""
import functools
import importlib.abc
import json
import os
import sys
import traceback

TARGET = "vllm.models.qwen4_exp.nvidia.low_latency_gemm"
_installed = False
_STATE = {"selected": 0, "converted": 0, "bytes_before": 0, "bytes_after": 0, "checked": set(), "plan": None}

CATEGORIES = {   # category -> module-name suffix
    "qkvz": "linear_attn.in_proj_qkvz",
    "gdn_out": "linear_attn.out_proj",
    "qsa_qkv": "self_attn.qkv_proj",
    "qsa_o": "self_attn.o_proj",
    "shared_gu": "shared_expert.gate_up_proj",
    "shared_down": "shared_expert.down_proj",
}


def _log(msg):
    sys.stderr.write("[swift_fp8] %s\n" % msg)
    sys.stderr.flush()


def _strict():
    return os.environ.get("SWIFT_FP8_STRICT", "1").strip().lower() not in ("0", "false", "no", "off")


def _group():
    return int(os.environ.get("SWIFT_FP8_GROUP", "0"))


def _q():
    import fp8_gemv   # same directory
    return fp8_gemv


def _plan():
    if _STATE["plan"] is None:
        Q = _q()
        plan = {}
        path = os.environ.get("SWIFT_FP8_PLAN")
        if path:
            for key, per_m in json.load(open(path)).items():
                n, k = (int(v) for v in key.split("x"))
                plan[(n, k)] = {int(m): Q.W8Conf(**c) for m, c in per_m.items()}
        _STATE["plan"] = plan
    return _STATE["plan"]


def _default_conf(n, k, m):
    Q = _q()
    g = _group()
    bk = 128 if g else 256
    if m <= 8:
        sk = 1 if n >= 4096 else (2 if n >= 1024 else 4)
        return Q.W8Conf("fma", 8, bk, sk, 4)
    return Q.W8Conf("dot", 64, 128, 1 if n >= 4096 else 2, 4, 4, 16 if m <= 16 else 32)


def conf_for(n, k, m):
    if m > int(os.environ.get("SWIFT_FP8_MAX_M", "32")):
        return None
    per = _plan().get((n, k), {})
    c = per.get(m)
    if c is None:
        # nearest planned M whose kernel can take this M (fma compiles per M, so any planned fma conf works for M<=8)
        cands = sorted((pm for pm, pc in per.items()
                        if (pc.kind == "fma" and m <= 8) or (pc.kind == "dot" and m <= pc.block_m)),
                       key=lambda pm: abs(pm - m))
        c = per[cands[0]] if cands else _default_conf(n, k, m)
    return c


_DEQ_CHUNK_ROWS = int(os.environ.get("SWIFT_FP8_DEQ_CHUNK_ROWS", "8192"))


def _dequant_linear(x2, w8, scale, group):
    """Large-M fallback (prefill, prompt logprobs): dequantize at most _DEQ_CHUNK_ROWS weight rows at a time so the
    temporaries stay bounded (~84 MB fp32 for K=2560) even for the 248,320-row lm_head."""
    import torch
    Q = _q()
    n = w8.shape[0]
    if n <= _DEQ_CHUNK_ROWS:
        return torch.nn.functional.linear(x2, Q.dequantize_fp8(w8, scale, group, dtype=x2.dtype))
    out = torch.empty((x2.shape[0], n), dtype=x2.dtype, device=x2.device)
    for n0 in range(0, n, _DEQ_CHUNK_ROWS):
        n1 = min(n, n0 + _DEQ_CHUNK_ROWS)
        w = Q.dequantize_fp8(w8[n0:n1], scale[n0:n1], group, dtype=x2.dtype)
        out[:, n0:n1] = torch.nn.functional.linear(x2, w)
        del w
    return out


def _make_method_class():
    import torch
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    class _W8Mixin:
        """Weight-only FP8 for one Linear / LM head; created by the selection wrapper, armed after weights load."""

        def __init__(self, name):
            super().__init__()
            self.name = name
            self.group = _group()

        def process_weights_after_loading(self, layer):
            Q = _q()
            w = layer.weight.data
            if w.dtype == Q.FP8:
                return
            n, k = w.shape
            if self.group and k % self.group:
                _log("%s: K=%d not divisible by group %d; left BF16" % (self.name, k, self.group))
                layer.quant_method = self._fallback()
                return
            w8, s = Q.quantize_fp8(w, self.group)
            _STATE["bytes_before"] += w.numel() * w.element_size()
            _STATE["bytes_after"] += w8.numel() + s.numel() * 4
            layer.weight = torch.nn.Parameter(w8, requires_grad=False)
            layer.w8_scale = s
            del w
            _STATE["converted"] += 1
            self._warm(layer, n, k)
            if _STATE["converted"] == _STATE["selected"]:
                torch.cuda.empty_cache()
                _log("converted %d modules, saved %.2f GiB (%.2f -> %.2f GiB), group=%d, plan=%s" % (
                    _STATE["converted"], (_STATE["bytes_before"] - _STATE["bytes_after"]) / 2**30,
                    _STATE["bytes_before"] / 2**30, _STATE["bytes_after"] / 2**30, self.group,
                    os.environ.get("SWIFT_FP8_PLAN") or "defaults"))

        def _warm(self, layer, n, k):
            """Compile every warm M once per (N, K) (outside CUDA-graph capture) and check it against
            F.linear on the dequantized weight (same quantized values, so only accumulation order differs)."""
            Q = _q()
            ms = [int(v) for v in os.environ.get("SWIFT_FP8_WARM_MS", "1,2,3,4,5,10,15,20").split(",") if v]
            for m in ms:
                key = (n, k, m)
                if key in _STATE["checked"]:
                    continue
                _STATE["checked"].add(key)
                cfg = conf_for(n, k, m)
                if cfg is None:
                    continue
                x = torch.randn((m, k), dtype=torch.bfloat16, device=layer.weight.device)
                try:
                    out = Q.w8_gemm(x, layer.weight, layer.w8_scale, self.group, cfg)
                    ref = _dequant_linear(x, layer.weight, layer.w8_scale, self.group).float()
                    torch.cuda.synchronize()
                    err = ((out.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
                    bad = not (err <= 2e-2)
                except Exception:
                    _log("%dx%d M=%d [%s] failed:\n%s" % (n, k, m, cfg.label(), traceback.format_exc()))
                    bad, err = True, float("nan")
                _log("check %dx%d M=%d [%s]: max|err|/max|ref| vs dequant+cuBLAS = %.2e%s" % (
                    n, k, m, cfg.label(), err, "  <-- FAIL" if bad else ""))
                if bad and _strict():
                    raise RuntimeError("[swift_fp8] kernel check failed for %dx%d M=%d" % (n, k, m))

        def apply(self, layer, x, bias=None):
            w8 = layer.weight
            if w8.dtype != _q().FP8:      # not converted (e.g. weights not loaded yet): stock path
                return super().apply(layer, x, bias)
            n, k = w8.shape
            x2 = x.reshape(-1, k)
            m = x2.shape[0]
            cfg = conf_for(n, k, m) if m > 0 else None
            if cfg is not None and x2.stride(1) == 1 and x2.stride(0) % 8 == 0:
                out = _q().w8_gemm(x2, w8, layer.w8_scale, self.group, cfg)
            else:
                out = _dequant_linear(x2, w8, layer.w8_scale, self.group)
            if bias is not None:
                out = out + bias
            return out.reshape(*x.shape[:-1], n)

    class W8LinearMethod(_W8Mixin, UnquantizedLinearMethod):
        _fallback = UnquantizedLinearMethod

    class W8HeadMethod(_W8Mixin, UnquantizedEmbeddingMethod):
        """LM head (logits only: ParallelLMHead never calls embedding()).  The weight row order is unchanged, so
        consumers that index rows (the draft-vocab hook) must read layer.weight + layer.w8_scale."""
        _fallback = UnquantizedEmbeddingMethod

    return W8LinearMethod, W8HeadMethod


def _targets():
    raw = [t.strip() for t in os.environ.get("SWIFT_FP8_TARGETS", "qkvz,gdn_out,qsa_qkv,qsa_o").split(",") if t.strip()]
    main, mtp = set(), set()
    for t in raw:
        if t == "mtp":
            mtp.update(CATEGORIES)
        elif t.startswith("mtp_"):
            mtp.add(t[4:])
        else:
            main.add(t)
    unknown = (main | mtp) - set(CATEGORIES) - {"lm_head"}
    if unknown:
        raise ValueError("[swift_fp8] unknown SWIFT_FP8_TARGETS categories: %s" % sorted(unknown))
    return main, mtp


def _select(module):
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod
    if _STATE.get("cls") is None:
        _STATE["cls"] = _make_method_class()
    W8, W8Head = _STATE["cls"]
    is_mtp = type(module).__name__.endswith("MTP")
    main, mtp = _targets()
    cats = mtp if is_mtp else main
    counts = {}
    if "lm_head" in cats:
        head = getattr(module, "lm_head", None)
        if isinstance(head, ParallelLMHead) and isinstance(head.quant_method, UnquantizedEmbeddingMethod)                 and getattr(head, "tp_size", 1) == 1 and getattr(head, "bias", None) is None:
            head.quant_method = W8Head("lm_head")
            counts["lm_head"] = 1
            _STATE["selected"] += 1
    for name, child in module.named_modules():
        if not isinstance(child, LinearBase) or getattr(child, "bias", None) is not None:
            continue
        for cat in cats:
            if cat in CATEGORIES and name.endswith(CATEGORIES[cat]):
                child.quant_method = W8(name)
                counts[cat] = counts.get(cat, 0) + 1
                _STATE["selected"] += 1
                break
    _log("%s: selected for FP8 %s (categories %s)" % (type(module).__name__, counts, sorted(cats)))
    if (cats - {"lm_head"} if is_mtp else cats) and not counts and _strict():
        raise RuntimeError("[swift_fp8] SWIFT_FP8=1 but no module of %s matched %s" % (type(module).__name__, sorted(cats)))


def _patch(mod):
    orig_enable = mod.enable_qwen4_exp_low_latency_gemm

    @functools.wraps(orig_enable)
    def enable(module, dtype, _orig=orig_enable):
        _orig(module, dtype)
        _select(module)

    mod.enable_qwen4_exp_low_latency_gemm = enable
    _plan()   # a bad plan file fails at import time
    _log("hooked %s (targets=%s, group=%d, plan=%s, strict=%s)" % (
        TARGET, os.environ.get("SWIFT_FP8_TARGETS", "qkvz,gdn_out,qsa_qkv,qsa_o"), _group(),
        os.environ.get("SWIFT_FP8_PLAN") or "defaults", _strict()))


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, target):
        self.target = target

    def find_spec(self, name, path, target=None):
        # re-entrancy guard: swift_skinny_patch's finder also walks sys.meta_path and would call us back
        if name != self.target or getattr(self, "_busy", False):
            return None
        self._busy = True
        try:
            return self._find(name, path, target)
        finally:
            self._busy = False

    def _find(self, name, path, target):
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                orig_exec = spec.loader.exec_module

                def exec_module(module, _orig=orig_exec):
                    _orig(module)
                    try:
                        _patch(module)
                    except Exception:
                        _log("patch failed, vLLM left unmodified:\n" + traceback.format_exc())
                        if _strict():
                            raise

                spec.loader.exec_module = exec_module
                return spec
        return None


def install(target=TARGET, force=False):
    """Idempotent.  No-op unless SWIFT_FP8=1 (or force=True).  Install AFTER swift_skinny_patch so this wrapper runs
    after the skinny one (the skinny wrapper must assign its methods first; we then override the selected modules)."""
    global _installed
    if _installed or not (force or os.environ.get("SWIFT_FP8") == "1"):
        return
    _installed = True
    if target in sys.modules:
        _patch(sys.modules[target])
    else:
        sys.meta_path.insert(0, _Finder(target))
