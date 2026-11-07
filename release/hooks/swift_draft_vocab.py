# SPDX-License-Identifier: Apache-2.0
"""Trimmed-vocabulary MTP draft head for vLLM 0.31.0 (Swift / Qwen4Exp, V2 model runner).

What it does
------------
The MTP drafter of vLLM's V2 runner (``vllm.v1.worker.gpu.spec_decode``) computes full-vocabulary
logits (a BF16 [248320, 2560] GEMV, ~840 us on an RTX PRO 6000) three times per verify cycle and takes
a plain ``argmax``.  With ``draft_sample_method == "greedy"`` (the default) those logits are used for
nothing else: the draft token is just a *proposal* that the target model verifies.  This patch replaces
that head by a GEMV over a contiguous copy of a kept subset of rows of the (shared) lm_head, an argmax
over the subset, and a gather through an int64 id table that maps the subset index back to the real
token id.  The target model's own lm_head / sampler / rejection sampler are untouched, so every emitted
token is still the target model's own token (lossless); only draft quality (acceptance) can change.

Activation
----------
Strictly opt-in.  Nothing is imported or patched unless ``SWIFT_DRAFT_VOCAB_FILE`` is set:

  SWIFT_DRAFT_VOCAB_FILE   path to the kept token ids (.npy int array, .pt/.pth torch tensor or
                           {"ids": tensor}, .json list, .txt whitespace/comma separated, anything
                           else = raw little-endian int32).  Sorted/unique is enforced here (np.unique)
                           so that ties resolve to the lowest token id exactly like the full-vocab argmax.
  SWIFT_DRAFT_HEAD_DTYPE   bf16 (default).  fp8 is NOT implemented (see TEST_PLAN.md) and is refused.
  SWIFT_DRAFT_VOCAB_PAD    pad the table to a multiple of this many rows (default 64) by repeating the
                           last kept id (harmless: duplicates map to the same token id).  1 = no padding.
  SWIFT_DRAFT_VOCAB_STRICT 1 (default): any reason the patch cannot be activated is a hard error at
                           start-up (so a benchmark can never silently run un-patched).  0: log a WARNING
                           and run the stock full-vocabulary path.

How it is hooked (import-order robust)
--------------------------------------
``sitecustomize.py`` (on PYTHONPATH, therefore executed by every interpreter that inherits the env,
including the spawned EngineCore) calls :func:`install`.  ``install`` imports nothing heavy (no torch,
no vllm).  It puts a ``sys.meta_path`` finder in front of the normal finders; when - and only when -
``vllm.v1.worker.gpu.spec_decode.mtp.speculator`` is imported (vLLM imports it lazily inside
``init_speculator``) the finder lets the normal loader execute the module and then runs :func:`_apply`
on the fully initialised module.  If that module is already in ``sys.modules`` when ``install`` runs
(late plugin-style use, fork after import) ``_apply`` runs immediately.  ``_apply`` is idempotent.

Patched call sites (vllm 0.31.0, file:line in the installed tree)
-----------------------------------------------------------------
* ``MTPSpeculator.__init__`` (inherited from ``AutoRegressiveSpeculator.__init__``,
  autoregressive/speculator.py:30; called from ``init_speculator``, model_runner.py:274, i.e. BEFORE any
  weight is loaded): after the stock ``__init__`` run the config-level eligibility checks and read /
  validate the vocab file.  Any problem is a hard error within seconds of start-up (strict mode).
* ``MTPSpeculator.load_model`` (inherited from ``AutoRegressiveSpeculator.load_model``,
  autoregressive/speculator.py:50; called at model_runner.py:398-400 after the target model is loaded):
  run the stock ``load_model`` (which shares the target's lm_head into the draft via
  ``load_eagle_model``, eagle/utils.py:100-118), then build ``W_sub = lm_head.weight[ids]`` and the id table
  ONCE.  This is before the graph capture at model_runner.py:1066, so the captured graphs hold the
  addresses of these static buffers.
* ``MTPSpeculator.sample_draft`` (inherited from ``DraftModelSpeculator.sample_draft``,
  speculator.py:402-436), the single funnel through which draft step 0 (autoregressive/speculator.py:485,
  inside ``_prefill``) and steps 1..k-1 (autoregressive/speculator.py:664, inside ``_generate_draft``)
  obtain their tokens - i.e. inside CUDA graphs 20 and 29 of the nsys profile.  Replaces
  ``compute_logits`` (-> LogitsProcessor -> lm_head.quant_method.apply -> F.linear) + ``argmax`` of the
  ``draft_logits is None`` greedy branch (speculator.py:432-434).
"""
from __future__ import annotations

import functools
import hashlib
import importlib.abc
import os
import re
import sys

__all__ = ["install", "status"]

ENV_FILE = "SWIFT_DRAFT_VOCAB_FILE"
ENV_DTYPE = "SWIFT_DRAFT_HEAD_DTYPE"
ENV_PAD = "SWIFT_DRAFT_VOCAB_PAD"
ENV_STRICT = "SWIFT_DRAFT_VOCAB_STRICT"

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.mtp.speculator"
SUPPORTED_DRAFT_MODELS = ("Qwen4ExpMTP",)
TAG = "[swift-draft-vocab]"

# Test hook: the CPU unit tests flip this so a CPU tensor is accepted as the lm_head weight.
_ALLOW_CPU_FOR_TESTS = False

_STATE: dict = {"installed": False, "applied": False, "heads": 0}


# --------------------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------------------
def _log():
    try:
        from vllm.logger import init_logger

        return init_logger("vllm.swift_draft_vocab")
    except Exception:  # pragma: no cover - only without vllm (unit tests)
        import logging

        return logging.getLogger("swift_draft_vocab")


# --------------------------------------------------------------------------------------
# configuration / vocab file
# --------------------------------------------------------------------------------------
class _Cfg:
    def __init__(self) -> None:
        self.path = os.environ[ENV_FILE]
        self.dtype = os.environ.get(ENV_DTYPE, "bf16").strip().lower()
        self.pad = int(os.environ.get(ENV_PAD, "64"))
        self.strict = os.environ.get(ENV_STRICT, "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )


def _load_ids(path: str):
    """Read the kept-id file into a 1-D int64 numpy array (validation of the values is separate)."""
    import numpy as np

    if not os.path.isfile(path):
        raise FileNotFoundError(f"{ENV_FILE}={path!r} does not exist")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif ext in (".pt", ".pth"):
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(obj, dict):
            for key in ("ids", "token_ids", "keep_ids", "kept_ids", "vocab_ids"):
                if key in obj:
                    obj = obj[key]
                    break
            else:
                raise ValueError(f"{path}: dict without one of ids/token_ids/keep_ids/kept_ids/vocab_ids")
        arr = obj.detach().cpu().numpy() if hasattr(obj, "detach") else np.asarray(obj)
    elif ext == ".json":
        import json

        with open(path) as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            obj = obj.get("ids", obj.get("token_ids"))
        arr = np.asarray(obj)
    elif ext in (".txt", ".csv", ".ids"):
        with open(path) as f:
            text = f.read().strip()
        arr = np.array([int(t) for t in re.split(r"[\s,;]+", text) if t], dtype=np.int64)
    else:  # .bin / .i32 / anything else: raw little-endian int32
        arr = np.fromfile(path, dtype="<i4")
    arr = np.asarray(arr)
    if arr.ndim != 1 or arr.size == 0 or arr.dtype.kind not in "iu":
        raise ValueError(f"{path}: expected a non-empty 1-D integer array, got shape={arr.shape} dtype={arr.dtype}")
    return arr.astype(np.int64)


def _normalize_ids(arr, vocab_size: int, pad: int):
    """sorted-unique ids, padded to a multiple of ``pad`` with duplicates of the last id.

    Returns (ids int64 ndarray, n_real, notes).
    """
    import numpy as np

    notes = []
    if int(arr.min()) < 0 or int(arr.max()) >= vocab_size:
        raise ValueError(
            f"kept ids out of range: min={int(arr.min())} max={int(arr.max())} but vocab_size={vocab_size}"
        )
    uniq = np.unique(arr)
    if uniq.size != arr.size:
        notes.append(f"{arr.size - uniq.size} duplicate id(s) removed")
    elif not bool(np.all(arr[:-1] < arr[1:])):
        notes.append("ids were not sorted ascending; sorted (needed for lowest-id tie-breaking)")
    n_real = int(uniq.size)
    if n_real >= vocab_size:
        raise ValueError(f"kept ids cover the whole vocabulary ({n_real}/{vocab_size}); nothing to trim")
    if pad > 1 and n_real % pad:
        extra = pad - n_real % pad
        uniq = np.concatenate([uniq, np.full(extra, uniq[-1], dtype=uniq.dtype)])
        notes.append(f"padded {extra} duplicate row(s) of id {int(uniq[-1])} to reach a multiple of {pad}")
    return uniq.astype(np.int64), n_real, notes


# --------------------------------------------------------------------------------------
# the device-side head (static buffers; CUDA-graph safe)
# --------------------------------------------------------------------------------------
class _DraftHead:
    """Contiguous [V_sub, H] copy of lm_head rows + int64 id table.  Allocated once, never resized."""

    def __init__(self, w_sub, ids, v_full: int, n_real: int, path: str, digest: str):
        self.w = w_sub  # [V_sub, H] bf16, row-major contiguous (same layout F.linear sees in the stock path)
        self.ids = ids  # [V_sub] int64, ascending, last rows may repeat the last real id
        self.v_full = v_full
        self.v_sub = int(w_sub.shape[0])
        self.n_real = n_real
        self.path = path
        self.digest = digest
        self.calls = {False: 0, True: 0}  # python-level call counts: eager/warm-up vs CUDA-graph capture

    def sample(self, hidden_states):
        """[M, H] hidden -> [M] int64 FULL-vocabulary token ids.  Pure ATen, no host sync, no data-dependent shapes."""
        import torch

        h = hidden_states
        if h.dtype != torch.bfloat16:
            h = h.to(torch.bfloat16)
        if getattr(self, "w8", None) is not None and h.shape[0] <= 32 and h.stride(1) == 1 and h.stride(0) % 8 == 0:
            # SWIFT_DRAFT_HEAD_FP8=1: weight-only FP8 copy of W_sub (draft proposals only; target unaffected)
            import fp8_gemv
            import swift_fp8_dense
            n, k = self.w8.shape
            logits = fp8_gemv.w8_gemm(h, self.w8, self.w8_scale, 0, swift_fp8_dense.conf_for(n, k, h.shape[0]))
        else:
            # same op the stock path ends in (default_unquantized_gemm -> F.linear), just with N = V_sub
            w = self.w
            if w is None:   # FP8 head, large M: dequantize (never on the CUDA-graph path, M <= 32 there)
                import fp8_gemv
                w = fp8_gemv.dequantize_fp8(self.w8, self.w8_scale, 0)
            logits = torch.nn.functional.linear(h, w)  # [M, V_sub] bf16
        local = logits.argmax(dim=-1)  # [M] int64, first max wins -> lowest id on ties (ids ascending)
        out = self.ids.index_select(0, local)  # [M] int64, contiguous (update_draft_inputs needs stride 1)
        capturing = bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
        self.calls[capturing] += 1
        if self.calls[capturing] <= 2:  # python side effects run at warm-up/capture time only, never at graph replay
            _log().info(
                "%s sample_draft call #%d (cuda-graph capture=%s): hidden %s %s -> logits [%d, %d] -> argmax -> "
                "id table -> int64 %s",
                TAG,
                self.calls[capturing],
                capturing,
                tuple(hidden_states.shape),
                str(hidden_states.dtype).replace("torch.", ""),
                h.shape[0],
                self.v_sub,
                tuple(out.shape),
            )
        return out


# --------------------------------------------------------------------------------------
# eligibility checks
# --------------------------------------------------------------------------------------
def _precheck(spec, cfg: _Cfg):
    """Config-level reasons the trimmed head must not be used.  Cheap; runs before the weight load."""
    if cfg.dtype not in ("bf16", "bfloat16"):
        if cfg.dtype.startswith("fp8"):
            return (
                f"{ENV_DTYPE}={cfg.dtype}: fp8 draft head is not implemented (unverifiable without a GPU, "
                "gain <= ~0.15 ms/cycle); use bf16"
            )
        return f"{ENV_DTYPE}={cfg.dtype!r} is not a recognised value (bf16)"
    sc = spec.speculative_config
    if getattr(sc, "draft_sample_method", "greedy") != "greedy" or spec.draft_logits is not None:
        return "draft_sample_method is not 'greedy': the rejection sampler needs full-vocabulary draft logits"
    if spec.use_local_argmax_reduction:
        return "use_local_argmax_reduction=True uses model.get_top_tokens (separate path, not patched)"
    if spec.enable_adaptive_verification or spec.use_acceptance_estimator:
        return "adaptive verification reads the full draft logits"
    if spec.draft_watermarker is not None:
        return "draft watermarking reads the full draft logits"
    tp = spec.vllm_config.parallel_config.tensor_parallel_size
    if tp != 1:
        return f"tensor_parallel_size={tp}: lm_head is vocab-sharded; only TP=1 is supported"
    import vllm.envs as envs

    if envs.VLLM_BATCH_INVARIANT:
        return "VLLM_BATCH_INVARIANT=1 routes the head through linear_batch_invariant"
    if spec.num_speculative_steps < 1:
        return "num_speculative_tokens < 1"
    return None


def _build_head(spec, cfg: _Cfg, ids_np, n_real: int, target_model=None):
    """Post-load: validate the actual modules, then allocate W_sub + id table.  Raises RuntimeError on any mismatch."""
    import torch
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    model = spec.model
    mname = type(model).__name__
    if mname not in SUPPORTED_DRAFT_MODELS:
        raise RuntimeError(f"draft model {mname} is not in the verified set {SUPPORTED_DRAFT_MODELS}")
    head = getattr(model, "lm_head", None)
    if head is None or not hasattr(head, "weight") or not hasattr(head, "quant_method"):
        raise RuntimeError(f"draft model has no usable lm_head (got {type(head).__name__})")
    if getattr(head, "tp_size", 1) != 1:
        raise RuntimeError(f"lm_head.tp_size={head.tp_size}")
    if not isinstance(head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)):
        raise RuntimeError(f"lm_head quant_method {type(head.quant_method).__name__} is not an unquantized BF16 head")
    if getattr(head, "bias", None) is not None:
        raise RuntimeError("lm_head has a bias")
    w = head.weight
    w8_scale = getattr(head, "w8_scale", None)
    if w.dtype == torch.float8_e4m3fn and w8_scale is not None:
        # swift_fp8_dense converted the shared lm_head to FP8: build the trimmed head from the dequantized rows
        import fp8_gemv
        _g = getattr(head.quant_method, "group", 0)
        _w8, _s = w, w8_scale

        class _Deq:   # minimal view so the checks and index_select below work unchanged
            pass
        w = _Deq()
        w.shape, w.device, w.is_cuda, w.dtype = _w8.shape, _w8.device, _w8.is_cuda, torch.bfloat16
        w.dim = lambda: 2
        w.element_size = lambda: 2
        w.index_select = lambda d, idx: fp8_gemv.dequantize_fp8(_w8.index_select(0, idx), _s.index_select(0, idx), _g)
    if w.dim() != 2 or w.dtype != torch.bfloat16:
        raise RuntimeError(f"lm_head.weight must be 2-D bf16, got {tuple(w.shape)} {w.dtype}")
    if not w.is_cuda and not _ALLOW_CPU_FOR_TESTS:
        raise RuntimeError("lm_head.weight is not on a CUDA device")
    start = getattr(getattr(head, "shard_indices", None), "org_vocab_start_index", 0)
    if start != 0:
        raise RuntimeError(f"lm_head shard starts at vocab row {start}, expected 0")
    org = int(getattr(head, "org_vocab_size", spec.vocab_size))
    if int(ids_np.max()) >= org or int(ids_np.max()) >= w.shape[0]:
        raise RuntimeError(f"kept ids reach {int(ids_np.max())} but lm_head has org_vocab_size={org} rows={w.shape[0]}")
    hidden = getattr(getattr(model, "config", None), "hidden_size", w.shape[1])
    if hidden != w.shape[1]:
        raise RuntimeError(f"model.config.hidden_size={hidden} != lm_head.weight.shape[1]={w.shape[1]}")
    lp = getattr(model, "logits_processor", None)
    if lp is not None:
        if getattr(lp, "scale", 1.0) != 1.0 or getattr(lp, "soft_cap", None) is not None or getattr(
            lp, "logits_as_input", False
        ):
            raise RuntimeError("LogitsProcessor applies scale/soft_cap/logits_as_input; not reproduced by the trimmed head")
        hd = getattr(lp, "head_dtype", None)
        if hd is not None and hd != torch.bfloat16:
            raise RuntimeError(f"LogitsProcessor.head_dtype={hd} (only bf16 supported)")

    n_rows = int(ids_np.shape[0])
    need = n_rows * w.shape[1] * w.element_size()
    if w.is_cuda:
        free, _ = torch.cuda.mem_get_info(w.device)
        if free < need * 1.5 + (256 << 20):
            raise RuntimeError(f"not enough free GPU memory for the draft head copy: need {need >> 20} MiB, free {free >> 20} MiB")

    with torch.no_grad():
        ids_dev = torch.from_numpy(ids_np).to(device=w.device, dtype=torch.int64)
        w_sub = w.index_select(0, ids_dev)  # fresh, contiguous [V_sub, H] allocation
        if not w_sub.is_contiguous():  # pragma: no cover - index_select always returns contiguous
            w_sub = w_sub.contiguous()
        # exact consistency check of the id -> row mapping on 64 random rows (also warms index_select)
        pick = torch.randint(0, n_rows, (min(64, n_rows),), device=w.device)
        if not torch.equal(w.index_select(0, ids_dev.index_select(0, pick)), w_sub.index_select(0, pick)):
            raise RuntimeError("self-test failed: W_sub rows do not match lm_head rows at the mapped ids")
    if w.is_cuda:
        torch.cuda.synchronize(w.device)
    digest = hashlib.sha1(ids_np[:n_real].tobytes()).hexdigest()[:12]
    shared = None
    try:  # informational: load_eagle_model shares the target's lm_head object into the draft
        tl = target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
        thead = getattr(tl, "lm_head", None) or getattr(target_model, "lm_head", None)
        shared = head is thead
    except Exception:
        pass
    dh = _DraftHead(w_sub, ids_dev, int(w.shape[0]), n_real, cfg.path, digest)
    dh.w8 = None
    if os.environ.get("SWIFT_DRAFT_HEAD_FP8") == "1":
        import fp8_gemv
        dh.w8, dh.w8_scale = fp8_gemv.quantize_fp8(w_sub, 0)
        dh.w = None   # free the BF16 copy; the fallback path for M > 32 dequantizes
        for _m in (1, 2, 3, 4, 5, 8):   # compile outside CUDA-graph capture
            dh.sample(torch.zeros((_m, w_sub.shape[1]), dtype=torch.bfloat16, device=dh.w8.device))
        torch.cuda.synchronize()
        _log().info("%s draft head stored as FP8 e4m3 (%d x %d, %.1f MiB)", TAG, dh.w8.shape[0], dh.w8.shape[1],
                    dh.w8.numel() / 2**20)
    _log().info(
        "%s ACTIVE pid=%d: draft lm_head trimmed %d -> %d rows (%d kept + %d pad; %.2f%% of vocab), dtype=bf16, "
        "extra GPU memory %.1f MiB, ids sha1=%s, file=%s, draft model=%s, head shared with target=%s "
        "(built before CUDA-graph capture)",
        TAG,
        os.getpid(),
        w.shape[0],
        n_rows,
        n_real,
        n_rows - n_real,
        100.0 * n_rows / w.shape[0],
        (w_sub.numel() * w_sub.element_size() + ids_dev.numel() * 8) / (1 << 20),
        digest,
        cfg.path,
        mname,
        shared,
    )
    _STATE["heads"] += 1
    return dh


# --------------------------------------------------------------------------------------
# class patch
# --------------------------------------------------------------------------------------
def _apply(module) -> None:
    """Patch ``module.MTPSpeculator`` (idempotent)."""
    cls = getattr(module, "MTPSpeculator", None)
    if cls is None:
        raise RuntimeError(f"{module.__name__} has no MTPSpeculator (vLLM layout changed?)")
    if getattr(cls, "_swift_draft_vocab_patched", False):
        return
    orig_init = cls.__init__
    orig_load_model = cls.load_model
    orig_sample_draft = cls.sample_draft

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # runs from GPUModelRunner.__init__ -> init_speculator, i.e. before ANY weight is loaded
        self._swift_dv = None
        self._swift_cfg = None
        self._swift_ids = None
        cfg = _Cfg()
        log = _log()
        reason = None
        try:
            reason = _precheck(self, cfg)
            if reason is None:
                ids_np, n_real, notes = _normalize_ids(_load_ids(cfg.path), int(self.vocab_size), cfg.pad)
                self._swift_ids = (ids_np, n_real)
                for note in notes:
                    log.info("%s vocab file note: %s", TAG, note)
                log.info(
                    "%s pre-load checks passed pid=%d: %d kept ids (+%d pad) of vocab %d from %s",
                    TAG,
                    os.getpid(),
                    n_real,
                    len(ids_np) - n_real,
                    int(self.vocab_size),
                    cfg.path,
                )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        if reason is not None:
            msg = f"{TAG} cannot activate: {reason}"
            if cfg.strict:
                raise RuntimeError(msg)
            log.warning("%s -> DISABLED, running the stock full-vocabulary draft head", msg)
            self._swift_ids = None
        self._swift_cfg = cfg

    @functools.wraps(orig_load_model)
    def load_model(self, target_model):
        orig_load_model(self, target_model)  # loads the draft and shares the target lm_head into it
        ids = getattr(self, "_swift_ids", None)
        if ids is None:  # disabled (non-strict) in __init__
            return
        cfg = self._swift_cfg
        try:
            self._swift_dv = _build_head(self, cfg, ids[0], ids[1], target_model)
        except Exception as exc:
            msg = f"{TAG} cannot activate after weight load: {type(exc).__name__}: {exc}"
            if cfg.strict:
                raise RuntimeError(msg) from exc
            _log().warning("%s -> DISABLED, running the stock full-vocabulary draft head", msg)
            self._swift_dv = None

    @functools.wraps(orig_sample_draft)
    def sample_draft(
        self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds, draft_step, draft_logits
    ):
        dh = getattr(self, "_swift_dv", None)
        if dh is None or draft_logits is not None:
            return orig_sample_draft(
                self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds, draft_step, draft_logits
            )
        return dh.sample(hidden_states)

    cls.__init__ = __init__
    cls.load_model = load_model
    cls.sample_draft = sample_draft
    cls._swift_draft_vocab_patched = True
    _STATE["applied"] = True
    _log().info(
        "%s armed in pid=%d: patched %s.__init__/.load_model/.sample_draft (file=%s, dtype=%s, strict=%s)",
        TAG,
        os.getpid(),
        cls.__qualname__,
        os.environ.get(ENV_FILE),
        os.environ.get(ENV_DTYPE, "bf16"),
        os.environ.get(ENV_STRICT, "1"),
    )


# --------------------------------------------------------------------------------------
# post-import hook machinery (stdlib only)
# --------------------------------------------------------------------------------------
class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Runs ``hooks[name](module)`` right after the *normal* loader finished executing module ``name``."""

    def __init__(self, hooks):
        self._hooks = dict(hooks)
        self._busy: set = set()

    def find_spec(self, fullname, path=None, target=None):
        hook = self._hooks.get(fullname)
        if hook is None or fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
            spec = None
            for finder in sys.meta_path:  # delegate to every other finder, in order
                if finder is self:
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                spec = find(fullname, path, target)
                if spec is not None:
                    break
        finally:
            self._busy.discard(fullname)
        loader = getattr(spec, "loader", None)
        if spec is None or loader is None or not hasattr(loader, "exec_module"):
            return spec
        orig_exec = loader.exec_module
        hooks = self._hooks

        def exec_module(module, _orig=orig_exec, _name=fullname):
            _orig(module)
            cb = hooks.pop(_name, None)  # one-shot
            if cb is not None:
                cb(module)

        loader.exec_module = exec_module  # per-instance override; the loader object belongs to this spec only
        return spec


def install(hooks=None) -> bool:
    """Arm the hook(s).  Safe to call many times / from any process; a no-op when the env var is unset.

    ``hooks`` ({module name: callback}) is for tests; production use is ``install()``.
    """
    if hooks is None:
        if not os.environ.get(ENV_FILE):
            return False
        if _STATE["installed"]:
            return True
        hooks = {TARGET_MODULE: _apply}
    pending = {}
    for name, cb in hooks.items():
        mod = sys.modules.get(name)
        spec = getattr(mod, "__spec__", None)
        if mod is not None and not getattr(spec, "_initializing", False):
            cb(mod)  # already imported (late install / forked child): patch right now
        else:
            pending[name] = cb
    if pending:
        sys.meta_path.insert(0, _PostImportFinder(pending))
    _STATE["installed"] = True
    return True


def status() -> dict:
    return dict(_STATE)
