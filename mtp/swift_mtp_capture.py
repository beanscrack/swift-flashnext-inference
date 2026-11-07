"""swift_mtp_capture.py - record MTP draft proposals from vLLM 0.31 for offline parity checks.

Completely inert unless SWIFT_MTP_CAPTURE_DIR is set.  Stdlib-only at import time (torch is
imported lazily inside the hooks).  Intended for EAGER runs (--enforce-eager): inside a
CUDA graph the python hooks run only at capture time and would record nothing useful.

What is hooked (vLLM 0.31 install under .../site-packages/vllm):
  * vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer.propose   (llm_base_proposer.py:512)
      opens a "round" per proposal call, records the propose() arguments BEFORE the call
      (tensors copied to CPU; common_attn_metadata reduced to query_start_loc / seq_lens),
      and at return writes the vLLM draft tokens [B, k].
  * vllm.models.qwen4_exp.nvidia.mtp.Qwen4ExpMTP.forward                  (mtp.py:416)
      inside an open round, records every draft-model call in order: input_ids, positions,
      hidden_states (the multi-stream input), and - for the first SWIFT_MTP_CAPTURE_SAMPLE_ROUNDS
      rounds only - the outputs (sample_hidden [rows,H], multi_hidden [rows,HC*H]).
Call 0 is step 0 (all verified rows); calls 1..k-1 are the draft steps.  (llm_base_proposer.py:600,
754.)  The V2 runner (vllm/v1/worker/gpu/spec_decode, MTPSpeculator) is NOT hooked: run the parity
server with VLLM_USE_V2_MODEL_RUNNER=0 so this V1 path is the one that executes.

Env:
  SWIFT_MTP_CAPTURE_DIR            output directory (created if missing); arms the hook
  SWIFT_MTP_CAPTURE_MAX_MB         total payload cap in MB (default 2000); rounds beyond it are skipped
  SWIFT_MTP_CAPTURE_SAMPLE_ROUNDS  number of first rounds that also store per-call outputs (default 4)

Output: DIR/mtp_round_<pid>_<seq:06d>.pt, a dict with keys
  meta, propose_args, calls (list of dicts), draft_token_ids (CPU int tensor [B, k]).
"""

from __future__ import annotations

import functools
import importlib.abc
import inspect
import os
import sys
import time

ENV_DIR = "SWIFT_MTP_CAPTURE_DIR"
ENV_MAX_MB = "SWIFT_MTP_CAPTURE_MAX_MB"
ENV_SAMPLE_ROUNDS = "SWIFT_MTP_CAPTURE_SAMPLE_ROUNDS"
TAG = "[swift-mtp-capture]"
PROPOSER_MODULE = "vllm.v1.spec_decode.llm_base_proposer"
MTP_MODULE = "vllm.models.qwen4_exp.nvidia.mtp"

_STATE = {
    "installed": False,
    "dir": None,
    "max_bytes": 0,
    "bytes": 0,
    "rounds": 0,
    "skipped": 0,
    "sample_rounds": 4,
    "guard": False,  # re-entrancy guard: never record while we are writing or inside a recording
    "stack": [],  # open proposal rounds
}


def _log(msg: str) -> None:
    sys.stderr.write(f"{TAG} pid={os.getpid()} {msg}\n")


def _cpu(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", copy=True)
    return x


def _nbytes(obj) -> int:
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(v) for v in obj)
    return 0


def _record_args(bound) -> dict:
    import torch

    out = {}
    for name, val in bound.arguments.items():
        if name == "self":
            continue
        if val is None or isinstance(val, (int, float, bool, str)):
            out[name] = val
        elif isinstance(val, torch.Tensor):
            out[name] = _cpu(val)
        elif hasattr(val, "query_start_loc") and hasattr(val, "seq_lens"):  # CommonAttentionMetadata
            out[name] = {
                "query_start_loc": _cpu(val.query_start_loc),
                "seq_lens": _cpu(val.seq_lens),
                "num_actual_tokens": getattr(val, "num_actual_tokens", None),
                "max_query_len": getattr(val, "max_query_len", None),
            }
        # other objects (sampling metadata, slot mappings, ...) are not needed for parity
    return out


def _save_round(rec: dict) -> None:
    import torch

    st = _STATE
    size = _nbytes(rec)
    if st["bytes"] + size > st["max_bytes"]:
        st["skipped"] += 1
        if st["skipped"] == 1:
            _log(f"payload cap {st['max_bytes'] >> 20} MB reached after {st['rounds']} rounds; skipping further rounds")
        return
    st["guard"] = True
    try:
        path = os.path.join(st["dir"], f"mtp_round_{os.getpid()}_{st['rounds']:06d}.pt")
        tmp = path + ".tmp"
        torch.save(rec, tmp)
        os.replace(tmp, path)
        st["bytes"] += size
        st["rounds"] += 1
    finally:
        st["guard"] = False


def _wrap_propose(orig):
    """Wrap SpecDecodeBaseProposer.propose (llm_base_proposer.py:512)."""
    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def propose(self, *args, **kwargs):
        st = _STATE
        if st["guard"] or st["dir"] is None:
            return orig(self, *args, **kwargs)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        rec = {
            "meta": {"pid": os.getpid(), "time": time.time(), "round_index": st["rounds"]},
            "propose_args": _record_args(bound),  # copied BEFORE propose mutates seq_lens in place
            "calls": [],
        }
        st["stack"].append(rec)
        try:
            out = orig(self, *args, **kwargs)
        finally:
            st["stack"].pop()
        rec["draft_token_ids"] = _cpu(out)
        rec["meta"]["n_calls"] = len(rec["calls"])
        _save_round(rec)
        return out

    propose._swift_mtp_capture = True
    return propose


def _wrap_forward(orig):
    """Wrap Qwen4ExpMTP.forward (mtp.py:416)."""

    @functools.wraps(orig)
    def forward(self, input_ids, positions, hidden_states=None, intermediate_tensors=None,
                inputs_embeds=None, spec_step_idx=0):
        st = _STATE
        recording = bool(st["stack"]) and not st["guard"]
        # V2 runner (production): no propose() wrapper; record every MTP forward while $DIR/ARM exists
        standalone = os.environ.get("SWIFT_MTP_CAPTURE_STANDALONE") == "1" and st.get("dir")             and os.path.exists(os.path.join(st["dir"], "ARM"))
        if standalone:
            n = st.setdefault("sa_calls", 0)
            limit = int(os.environ.get("SWIFT_MTP_CAPTURE_MAX_CALLS", "400"))
            if n < limit:
                rec = {"input_ids": _cpu(input_ids), "positions": _cpu(positions), "hidden_in": _cpu(hidden_states),
                       "spec_step_idx": spec_step_idx}
                out = orig(self, input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds,
                           spec_step_idx=spec_step_idx)
                if isinstance(out, tuple) and len(out) == 2:
                    rec["sample_hidden"] = _cpu(out[0])
                    rec["multi_hidden"] = _cpu(out[1])
                import torch as _t
                _t.save(rec, os.path.join(st["dir"], "call_%05d.pt" % n))
                st["sa_calls"] = n + 1
                return out
        if recording:
            # copy inputs before the call: hidden_states may be a view of a persistent buffer
            call = {
                "input_ids": _cpu(input_ids),
                "positions": _cpu(positions),
                "hidden_in": _cpu(hidden_states),
                "inputs_embeds_given": inputs_embeds is not None,
                "spec_step_idx": spec_step_idx,
            }
        out = orig(self, input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds,
                   spec_step_idx=spec_step_idx)
        if recording:
            if isinstance(out, tuple) and len(out) == 2 and st["rounds"] < st["sample_rounds"]:
                call["sample_hidden"] = _cpu(out[0])
                call["multi_hidden"] = _cpu(out[1])
            st["stack"][-1]["calls"].append(call)
        return out

    forward._swift_mtp_capture = True
    return forward


def _apply_proposer(module) -> None:
    cls = getattr(module, "SpecDecodeBaseProposer", None)
    if cls is None:
        _log(f"{PROPOSER_MODULE} has no SpecDecodeBaseProposer; hook not applied")
        return
    if not getattr(cls.propose, "_swift_mtp_capture", False):
        cls.propose = _wrap_propose(cls.propose)
    # V1 proposer load_model reads target_model.config.image_token_index (llm_base_proposer.py:1389), which
    # Qwen4ExpConfig names image_token_id: alias it so the V1 runner can load (parity capture only).
    if not getattr(cls.load_model, "_swift_mtp_alias", False):
        orig_load = cls.load_model

        def load_model(self, target_model, *a, _orig=orig_load, **kw):
            for obj in (getattr(target_model, "config", None),):
                if obj is not None and not hasattr(obj, "image_token_index") and hasattr(obj, "image_token_id"):
                    obj.image_token_index = obj.image_token_id
                    _log("aliased target config image_token_index = image_token_id (%r)" % obj.image_token_id)
            return _orig(self, target_model, *a, **kw)

        load_model._swift_mtp_alias = True
        cls.load_model = load_model


def _apply_mtp(module) -> None:
    cls = getattr(module, "Qwen4ExpMTP", None)
    if cls is None:
        _log(f"{MTP_MODULE} has no Qwen4ExpMTP; hook not applied")
        return
    if not getattr(cls.forward, "_swift_mtp_capture", False):
        cls.forward = _wrap_forward(cls.forward)


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
        hooks = self._hooks

        def exec_module(module, _orig=orig_exec, _name=fullname):
            _orig(module)
            cb = hooks.pop(_name, None)  # one-shot
            if cb is not None:
                cb(module)

        loader.exec_module = exec_module
        return spec


def install(hooks=None) -> bool:
    """Arm the hooks.  No-op (returns False) unless SWIFT_MTP_CAPTURE_DIR is set.

    ``hooks`` ({module name: callback}) is for tests only.
    """
    if hooks is None:
        out_dir = os.environ.get(ENV_DIR)
        if not out_dir:
            return False
        if _STATE["installed"]:
            return True
        os.makedirs(out_dir, exist_ok=True)
        _STATE["dir"] = out_dir
        _STATE["max_bytes"] = int(float(os.environ.get(ENV_MAX_MB, "2000")) * (1 << 20))
        _STATE["sample_rounds"] = int(os.environ.get(ENV_SAMPLE_ROUNDS, "4"))
        hooks = {PROPOSER_MODULE: _apply_proposer, MTP_MODULE: _apply_mtp}
        _log(f"armed: dir={out_dir} cap={_STATE['max_bytes'] >> 20}MB sample_rounds={_STATE['sample_rounds']} "
             f"VLLM_USE_V2_MODEL_RUNNER={os.environ.get('VLLM_USE_V2_MODEL_RUNNER', 'unset')}")
    pending = {}
    for name, cb in hooks.items():
        mod = sys.modules.get(name)
        spec = getattr(mod, "__spec__", None)
        if mod is not None and not getattr(spec, "_initializing", False):
            cb(mod)
        else:
            pending[name] = cb
    if pending:
        sys.meta_path.insert(0, _PostImportFinder(pending))
    _STATE["installed"] = True
    return True


def status() -> dict:
    return {k: _STATE[k] for k in ("installed", "dir", "max_bytes", "bytes", "rounds", "skipped", "sample_rounds")}
