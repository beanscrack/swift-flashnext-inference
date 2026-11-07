"""swift_skinny_patch.py - enable low-latency skinny BF16 GEMMs for Qwen4Exp on SM120, without editing vLLM.

vLLM 0.31 already routes every UnquantizedLinearMethod / UnquantizedEmbeddingMethod layer whose weight shape
matches a plan through torch.ops.vllm.qwen4_exp_low_latency_gemm (vllm/models/qwen4_exp/nvidia/low_latency_gemm.py),
but _gemm_plans() returns {} on SM120 and the shipped plans are TP=4 shapes.  This module patches, right after
that module is first imported (import hook, so it works in the spawned EngineCore process):

  * low_latency_gemm._gemm_plans              -> our plan {(N, K): {M: config}} for the TP=1 shapes
  * low_latency_gemm.shape_dynamic_skinny_gemm -> dispatcher: Triton configs (bench_skinny.TConf) run our Triton
                                                 kernels, vLLM CuTe SkinnyGemmConfig configs run vLLM's kernel
  * optional (SWIFT_SKINNY_HC_TB=<64..512>)   -> thread-block size of vLLM's fused hc_down_silu FMA kernel

Everything is CUDA-graph safe (no host sync; outputs / split-K workspaces come from torch.empty inside the custom
op), uses the original weight tensors (no copies) and only fires for the exact M values present in the plan.

Enable:   export SWIFT_SKINNY=1 [SWIFT_SKINNY_PLAN=plan.json] [SWIFT_SKINNY_SHAPES=96x2560,16384x2560]
and make a sitecustomize.py on PYTHONPATH do:   import swift_skinny_patch; swift_skinny_patch.install()
(this directory must be on PYTHONPATH too, together with bench_skinny.py).  Without SWIFT_SKINNY=1 it is a no-op.
NOTE: /u701/j692wu/swift-profile/patch2 is the merged hook dir; PYTHONPATH=patch:gemm does NOT arm this patch
(patch/sitecustomize.py only knows the draft-vocab hook and gemm/ has no sitecustomize.py).
SWIFT_SKINNY_STRICT=1 (default): any failure to apply the patch (bad plan file, bench_skinny not importable, no layer of
the target model matching a plan key) aborts start-up instead of silently running stock vLLM; =0 = log and continue.
Every enable_qwen4_exp_low_latency_gemm call logs which layers (by weight shape) were routed to the skinny op.
SWIFT_SKINNY_CHECK=<n> (default 2): the first n EAGER calls (never inside CUDA-graph capture) of every planned
(N,K,M) are compared on the GPU against an fp32 reference and cuBLAS (real weights, warm-up activations); a result
worse than max(8e-3, 4 x cuBLAS) of max|ref| aborts (strict) - this is the first on-hardware check of the kernels.
Triton plan entries with pdl=True are downgraded to pdl=False unless SWIFT_SKINNY_ALLOW_PDL=1 (bench_skinny cannot
detect a PDL read-before-producer race: its preceding kernel never writes the GEMM input).
UNTESTED on GPU at the time of writing (the GPU was reserved); see INTEGRATION.md for the validation protocol.
"""
import dataclasses
import functools
import importlib.abc
import json
import os
import sys
import traceback

TARGET = "vllm.models.qwen4_exp.nvidia.low_latency_gemm"
_installed = False


def _log(msg):
    sys.stderr.write("[swift_skinny] %s\n" % msg)
    sys.stderr.flush()


def _strict():
    return os.environ.get("SWIFT_SKINNY_STRICT", "1").strip().lower() not in ("0", "false", "no", "off")


def _bench():
    import bench_skinny   # same directory; defines TConf / triton_skinny_gemm / config generators
    return bench_skinny


@dataclasses.dataclass(frozen=True)
class _Planned:
    """Marker wrapper so one frozen, hashable object carries the shape (for pre-compilation) and the config."""
    cfg: object
    n: int
    k: int
    m: int


def _config_from_entry(entry):
    fam, cfg = entry["family"], entry.get("cfg")
    B = _bench()
    if fam in ("triton_fma", "triton_dot"):
        c = B.TConf(**cfg)
        if c.pdl and os.environ.get("SWIFT_SKINNY_ALLOW_PDL") != "1":
            _log("plan entry %s uses PDL; downgraded to pdl=False (set SWIFT_SKINNY_ALLOW_PDL=1 to keep it)" % c.label())
            c = dataclasses.replace(c, pdl=False)
        return c
    if fam == "cute":
        from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import SkinnyGemmConfig
        return SkinnyGemmConfig(**cfg)
    return None     # flashinfer / torch variants are not wired into the plan (benchmark-only)


def _starter_plan():
    """Untuned fallback used when SWIFT_SKINNY_PLAN is not given: top-prior Triton config per (shape, M)."""
    import torch
    B = _bench()
    sm = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    plan = {}
    for name in ("gdn_ba", "qkvz", "o_proj", "qsa_qkv", "hc_up", "indexer"):
        sh = B.SHAPE_BY_NAME[name]
        for m in (1, 2, 4, 8):
            gen = B.gen_fma_configs if m <= 2 else B.gen_dot_configs
            cfgs = gen(sh.n, sh.k, m, sm, 1, [False])
            if cfgs:
                plan.setdefault((sh.n, sh.k), {})[m] = cfgs[0]
    return plan


def _load_plan():
    path = os.environ.get("SWIFT_SKINNY_PLAN")
    only = os.environ.get("SWIFT_SKINNY_SHAPES")
    only = set(only.split(",")) if only else None
    plan = {}
    if path:
        data = json.load(open(path))
        for key, per_m in data.items():
            if only and key not in only:
                continue
            n, k = (int(v) for v in key.split("x"))
            for m, entry in per_m.items():
                cfg = _config_from_entry(entry)
                if cfg is not None:
                    plan.setdefault((n, k), {})[int(m)] = cfg
    else:
        for (n, k), per_m in _starter_plan().items():
            if only and ("%dx%d" % (n, k)) not in only:
                continue
            plan[(n, k)] = per_m
    # wrap Triton configs so the dispatcher can pre-compile with the right shapes
    wrapped = {}
    B = _bench()
    for (n, k), per_m in plan.items():
        wrapped[(n, k)] = {m: (_Planned(c, n, k, m) if isinstance(c, B.TConf) else c) for m, c in per_m.items()}
    return wrapped


class _Dispatcher:
    """Drop-in for ShapeDynamicSkinnyGemm used by low_latency_gemm."""

    def __init__(self, real):
        self._real = real
        self._checked = {}
        try:
            self._check_n = int(os.environ.get("SWIFT_SKINNY_CHECK", "2"))
        except ValueError:
            self._check_n = 2

    def is_available(self):
        return True

    def request_warmup_configs(self, dtype, configs, *, has_residual=False):
        B = _bench()
        cute = [c for c in configs if not isinstance(c, _Planned)]
        if cute and self._real.is_available():
            self._real.request_warmup_configs(dtype, cute, has_residual=has_residual)
        import torch
        for c in configs:
            if isinstance(c, _Planned) and isinstance(c.cfg, B.TConf):
                try:   # compile now (outside CUDA-graph capture); keeps N % 16 / N % block_n class of the real shape
                    n = c.n if c.n <= 16384 else 16384 + (c.n % 128)
                    x = torch.zeros(c.m, c.k, dtype=dtype, device="cuda")
                    w = torch.zeros(n, c.k, dtype=dtype, device="cuda")
                    B.triton_skinny_gemm(x, w, c.cfg)
                    torch.cuda.synchronize()
                except Exception:
                    _log("warm-up compile failed for %s (will compile lazily):\n%s" % (c, traceback.format_exc()))

    def __call__(self, a, b, config=None, residual=None):
        if isinstance(config, _Planned):
            out = _bench().triton_skinny_gemm(a, b, config.cfg)
            if self._checked.get(config, 0) < self._check_n:
                self._self_check(a, b, out, config)
            return out
        return self._real(a, b, config, residual)

    def _self_check(self, a, b, out, config):
        import torch
        if a.is_cuda and torch.cuda.is_current_stream_capturing():
            return   # never sync inside a graph capture; the eager warm-up pass before every capture gets checked
        if not bool(torch.isfinite(a).all().item()) or not bool((a != 0).any().item()):
            # dummy/warm-up activations can be all-zero or non-finite (uninitialised KV in a dummy batch): uninformative,
            # so do not count this call; give up silently after 16 such calls for this config
            skips = self._checked.get((config, "skip"), 0) + 1
            self._checked[(config, "skip")] = skips
            if skips == 1:
                _log("self-check %dx%d M=%d deferred: uninformative input (all-zero or non-finite)" % (config.n, config.k, config.m))
            if skips >= 16:
                self._checked[config] = self._check_n
            return
        self._checked[config] = self._checked.get(config, 0) + 1
        # fp32-output GEMM as the reference: no fp32 copy of the weight (would add up to 168 MB to the profiled peak)
        try:   # aten::mm.dtype has a CUDA kernel only (verified: NotImplementedError on CPU in torch 2.13)
            ref = torch.mm(a, b.t(), out_dtype=torch.float32)
        except (RuntimeError, NotImplementedError, TypeError):
            ref = a.float() @ b.float().t()
        scale = ref.abs().max().clamp_min(1e-6)
        err = ((out.float() - ref).abs().max() / scale).item()
        err_blas = ((torch.nn.functional.linear(a, b).float() - ref).abs().max() / scale).item()
        finite = bool(torch.isfinite(out).all().item())
        bad = (not finite) or not (err <= max(8e-3, 4.0 * err_blas))
        _log("self-check %dx%d M=%d [%s]: max|err|/max|ref| = %.2e (cuBLAS bf16 %.2e, max|ref| %.3g)%s" % (
            config.n, config.k, config.m, config.cfg.label(), err, err_blas, scale.item(), "  <-- FAIL" if bad else ""))
        if bad and _strict():
            raise RuntimeError("[swift_skinny] kernel self-check failed for %dx%d M=%d %s" % (
                config.n, config.k, config.m, config.cfg.label()))

    def __getattr__(self, name):
        return getattr(self._real, name)


def _patch_hc_tb(tb):
    from vllm.models.qwen4_exp.nvidia.ops.cute_dsl import hc_down_silu as hd
    orig = hd.HcDownSiluGemm.dispatch

    def dispatch(self, m):
        if m <= 4 or self.k < 2048:
            return ("fma", m, self.k, tb)
        return orig(self, m)

    hd.HcDownSiluGemm.dispatch = dispatch
    _log("hc_down_silu FMA thread-block size set to %d" % tb)


def _report(mod, module, plan):
    """Log which layers enable_qwen4_exp_low_latency_gemm routed to the skinny op (grouped by weight shape)."""
    classes = tuple(c for c in (getattr(mod, "Qwen4ExpLowLatencyLinearMethod", None),
                                getattr(mod, "Qwen4ExpLowLatencyEmbeddingMethod", None)) if c is not None)
    counts = {}
    for child in module.modules():
        if classes and isinstance(getattr(child, "quant_method", None), classes):
            w = child.weight
            key = "%dx%d" % (w.shape[0], w.shape[1])
            counts[key] = counts.get(key, 0) + 1
    unmatched = sorted("%dx%d" % nk for nk in plan if "%dx%d" % nk not in counts)
    _log("enable_qwen4_exp_low_latency_gemm(%s): layers routed to the skinny op by weight shape: %s ; "
         "plan keys without a matching layer in this model: %s" % (type(module).__name__, counts, unmatched))
    if plan and not counts and not type(module).__name__.endswith("MTP") and _strict():
        raise RuntimeError("[swift_skinny] SWIFT_SKINNY=1 with a non-empty plan but no layer of %s matched a plan key "
                           "(set SWIFT_SKINNY_STRICT=0 to run anyway)" % type(module).__name__)


def _patch(mod):
    state = {}

    def plans():
        plan = state.get("plan")
        if plan is None:
            try:
                plan = _load_plan()
            except Exception:
                _log("plan load failed:" + chr(10) + traceback.format_exc())
                if _strict():
                    raise
                plan = {}
            state["plan"] = plan
            summary = {"%dx%d" % nk: sorted(v) for nk, v in plan.items()}
            if plan:   # run_config2.sh gates on this exact phrase: never print it for an empty plan
                _log("patched %s: plan keys (N x K -> M list): %s" % (TARGET, summary))
            else:
                _log("plan is EMPTY: %s left on stock F.linear for every shape (NOT patched)" % TARGET)
        return plan

    if os.environ.get("SWIFT_SKINNY_PLAN"):
        plans()   # a plan file needs no GPU: load it now so that a bad file fails at import time
    # The untuned starter plan queries the GPU (SM count), so it is built lazily on the first _gemm_plans() call, i.e.
    # inside enable_qwen4_exp_low_latency_gemm in the EngineCore, never in a process that merely imports the model module
    # (e.g. vLLM's model-registry inspection subprocess).
    mod.shape_dynamic_skinny_gemm = _Dispatcher(mod.shape_dynamic_skinny_gemm)
    mod._gemm_plans = plans
    if os.environ.get("SWIFT_SKINNY_STRIDED", "1") == "1":
        # HC input_mix_weight_up gets lora = output[:, :rank] of the fused hc_down_silu kernel (row stride 336, not
        # 320), which stock _runtime_ok rejects -> cuBLAS for 96 of 97 calls.  Our Triton kernels take a row stride,
        # so accept strided activations when the planned config for this (N, K, M) is a Triton one.
        orig_ok = mod._runtime_ok

        def _runtime_ok(x, weight, _orig=orig_ok):
            if _orig(x, weight):
                return True
            if not (x.dim() == 2 and x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.dtype == weight.dtype
                    and weight.dim() == 2 and weight.stride() == (weight.shape[1], 1) and x.shape[1] == weight.shape[1]):
                return False
            cfg = plans().get((weight.shape[0], weight.shape[1]), {}).get(x.shape[0])
            return isinstance(cfg, _Planned)

        mod._runtime_ok = _runtime_ok
        _log("strided-activation dispatch enabled for Triton plan entries (SWIFT_SKINNY_STRIDED=1)")
    orig_enable = getattr(mod, "enable_qwen4_exp_low_latency_gemm", None)
    if orig_enable is not None:   # model.py / mtp.py bind this name via "from .low_latency_gemm import", after this hook
        @functools.wraps(orig_enable)
        def enable(module, dtype, _orig=orig_enable):
            _orig(module, dtype)
            _report(mod, module, plans())

        mod.enable_qwen4_exp_low_latency_gemm = enable
    _log("hooked %s (plan: %s, strict=%s)" % (TARGET, os.environ.get("SWIFT_SKINNY_PLAN") or "untuned starter plan, built lazily",
                                              _strict()))
    tb = os.environ.get("SWIFT_SKINNY_HC_TB")
    if tb:
        try:
            _patch_hc_tb(int(tb))
        except Exception:
            _log("hc_tb patch failed:" + chr(10) + traceback.format_exc())
            if _strict():
                raise


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, target):
        self.target = target

    def find_spec(self, name, path, target=None):
        if name != self.target:
            return None
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
                        if _strict():   # fail fast: never let a SWIFT_SKINNY=1 run silently measure stock vLLM
                            raise

                spec.loader.exec_module = exec_module
                return spec
        return None


def install(target=TARGET, force=False):
    """Idempotent.  No-op unless SWIFT_SKINNY=1 (or force=True)."""
    global _installed
    if _installed or not (force or os.environ.get("SWIFT_SKINNY") == "1"):
        return
    _installed = True
    if target in sys.modules:
        _patch(sys.modules[target])
    else:
        sys.meta_path.insert(0, _Finder(target))
