"""swift_hidden_capture.py - capture the TARGET model's pre-final-mixer multi-stream hidden state.

Hooks Qwen4ExpModel.forward (vllm/models/qwen4_exp/nvidia/model.py).  After the original forward returns, if
$SWIFT_HIDDEN_CAPTURE_DIR/ARM exists and the forward has T > 1 tokens (a prefill), it saves
model._mtp_hidden_buffer[:T]  ([T, 4*2560] bf16, exactly what the MTP drafter consumes) plus input_ids/positions.

Inert unless SWIFT_HIDDEN_CAPTURE_DIR is set.  Stdlib only at import time.  Never raises into inference.

ARM protocol (written by the client before each request, removed after it):
  line 1: sample id (required, [A-Za-z0-9._-]+, used as file stem)
  line 2: output directory (optional, default = SWIFT_HIDDEN_CAPTURE_DIR)
  line 3: "bf16" (optional, also write <id>.bf16.pt with the unquantized buffer)
The first qualifying forward consumes ARM (renamed to ARM.done), so one request yields at most one save.

Output <outdir>/<id>.pt (torch dict):
  format="fp8_e4m3_rowscale_v1", sample_id, fp8 [T,10240] float8_e4m3fn, scale [T] fp32 (row amax/448),
  input_ids [T] int32 (None if inputs_embeds was used), positions int32, shape [T,10240], nonfinite int.
Every save appends one JSON line to $SWIFT_HIDDEN_CAPTURE_DIR/saves.log.
"""

from __future__ import annotations

import functools
import importlib.abc
import json
import os
import re
import sys
import time

ENV_DIR = "SWIFT_HIDDEN_CAPTURE_DIR"
TARGET_MODULE = "vllm.models.qwen4_exp.nvidia.model"
TARGET_CLASS = "Qwen4ExpModel"
TAG = "[swift-hidden-capture]"
FP8_MAX = 448.0
SID_RE = re.compile(r"[A-Za-z0-9._-]+")

_STATE = {"installed": False, "saves": 0, "errors": 0, "patched": False, "missing_buffer_logged": False}


def _log(msg: str) -> None:
    sys.stderr.write(f"{TAG} pid={os.getpid()} {msg}\n")
    sys.stderr.flush()


def _append_saves_log(base: str, obj: dict) -> None:
    try:
        with open(os.path.join(base, "saves.log"), "a") as f:
            f.write(json.dumps(obj) + "\n")
    except Exception:
        pass


def _read_arm(base: str):
    arm = os.path.join(base, "ARM")
    try:
        with open(arm) as f:
            lines = [ln.strip() for ln in f.read().splitlines()]
    except FileNotFoundError:
        return None
    sid = lines[0] if lines else ""
    if not SID_RE.fullmatch(sid or ""):
        raise ValueError(f"ARM has invalid sample id {sid!r}")
    outdir = lines[1] if len(lines) > 1 and lines[1] else base
    bf16 = len(lines) > 2 and lines[2].lower() == "bf16"
    return {"sid": sid, "outdir": outdir, "bf16": bf16, "arm": arm}


def _atomic_save(obj, path: str) -> None:
    import torch

    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _save(buf, T: int, input_ids, positions, arm: dict, base: str) -> None:
    import torch

    sid = arm["sid"]
    outdir = arm["outdir"]
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    x = buf[:T].detach()
    xf = x.float()
    nonfinite = int((~torch.isfinite(xf)).sum().item())
    amax = xf.abs().amax(dim=1)
    scale = (amax / FP8_MAX).clamp(min=1e-30)
    q = (xf / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    rec = {
        "format": "fp8_e4m3_rowscale_v1",
        "sample_id": sid,
        "fp8": q.to("cpu", copy=True),
        "scale": scale.to("cpu", copy=True),
        "input_ids": None if input_ids is None else input_ids.reshape(-1).to(torch.int32).to("cpu", copy=True),
        "positions": positions.to(torch.int32).to("cpu", copy=True),
        "shape": [int(s) for s in x.shape],
        "nonfinite": nonfinite,
        "buffer_shape": [int(s) for s in buf.shape],
    }
    path = os.path.join(outdir, sid + ".pt")
    _atomic_save(rec, path)
    if arm["bf16"]:
        _atomic_save(
            {"sample_id": sid, "bf16": x.to("cpu", copy=True), "shape": [int(s) for s in x.shape]},
            os.path.join(outdir, sid + ".bf16.pt"),
        )
    os.replace(arm["arm"], arm["arm"] + ".done")
    _STATE["saves"] += 1
    _append_saves_log(
        base,
        {
            "event": "save",
            "sample_id": sid,
            "outdir": outdir,
            "T": T,
            "bf16": arm["bf16"],
            "nonfinite": nonfinite,
            "seconds": round(time.time() - t0, 4),
            "pid": os.getpid(),
            "time": time.time(),
        },
    )


def _maybe_capture(model, input_ids, positions, base: str) -> None:
    buf = getattr(model, "_mtp_hidden_buffer", None)
    if buf is None:
        if not _STATE["missing_buffer_logged"]:
            _STATE["missing_buffer_logged"] = True
            _log("model has no _mtp_hidden_buffer (MTP speculation not enabled?); nothing captured")
        return
    T = int(positions.shape[-1])
    if T <= 1:
        return  # decode / single-token step: not a prefill
    arm = _read_arm(base)
    if arm is None:
        return
    if T > buf.shape[0]:
        raise RuntimeError(f"T={T} exceeds buffer rows {buf.shape[0]}")
    _save(buf, T, input_ids, positions, arm, base)


def _wrap_forward(orig):
    @functools.wraps(orig)
    def forward(self, input_ids, positions, *args, **kwargs):
        out = orig(self, input_ids, positions, *args, **kwargs)
        base = os.environ.get(ENV_DIR)
        if base and os.path.exists(os.path.join(base, "ARM")):
            try:
                _maybe_capture(self, input_ids, positions, base)
            except Exception as exc:  # never break inference; the client detects missing/invalid files
                _STATE["errors"] += 1
                _log(f"capture failed: {type(exc).__name__}: {exc}")
                _append_saves_log(base, {"event": "error", "error": f"{type(exc).__name__}: {exc}", "pid": os.getpid()})
        return out

    forward._swift_hidden_capture = True
    return forward


def _apply(module) -> None:
    cls = getattr(module, TARGET_CLASS, None)
    if cls is None:
        _log(f"{TARGET_MODULE} has no {TARGET_CLASS}; hook not applied")
        return
    if not getattr(cls.forward, "_swift_hidden_capture", False):
        cls.forward = _wrap_forward(cls.forward)
    _STATE["patched"] = True
    _log(f"patched {TARGET_CLASS}.forward in {TARGET_MODULE}")


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Runs hooks[name](module) right after the normal loader has executed module ``name``."""

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
            for finder in sys.meta_path:
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

        def exec_module(module, _orig=orig_exec, _hook=hook, _name=fullname, _self=self):
            _orig(module)
            try:
                _hook(module)
            finally:
                _self._hooks.pop(_name, None)
                if not _self._hooks and _self in sys.meta_path:
                    sys.meta_path.remove(_self)

        loader.exec_module = exec_module
        return spec


def install() -> bool:
    """Arm the hook.  No-op unless SWIFT_HIDDEN_CAPTURE_DIR is set."""
    if not os.environ.get(ENV_DIR):
        return False
    if _STATE["installed"]:
        return True
    mod = sys.modules.get(TARGET_MODULE)
    spec = getattr(mod, "__spec__", None)
    if mod is not None and not getattr(spec, "_initializing", False):
        _apply(mod)
    else:
        sys.meta_path.insert(0, _PostImportFinder({TARGET_MODULE: _apply}))
    _STATE["installed"] = True
    _log(f"armed dir={os.environ[ENV_DIR]}")
    return True
