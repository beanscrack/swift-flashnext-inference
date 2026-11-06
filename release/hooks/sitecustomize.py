# SPDX-License-Identifier: Apache-2.0
"""Interpreter start-up hook for the Swift trimmed-vocabulary draft head.

Put this directory on PYTHONPATH (the spawned vLLM EngineCore inherits the environment, so it runs this
too).  Completely inert unless SWIFT_DRAFT_VOCAB_FILE is set.  Imports only the stdlib (never torch/vllm),
so it is cheap and harmless in every other Python process that inherits PYTHONPATH (bench clients, nsys
helpers, pip, ...).  Never raises: a failure here must not stop an interpreter from starting.
"""
import os
import sys

if os.environ.get("SWIFT_DRAFT_VOCAB_FILE"):
    try:
        import swift_draft_vocab

        swift_draft_vocab.install()
    except Exception as _exc:  # pragma: no cover
        sys.stderr.write(f"[swift-draft-vocab] sitecustomize failed to arm the hook: {type(_exc).__name__}: {_exc}\n")


# --- merged hook (patch2): skinny BF16 GEMM patch, inert unless SWIFT_SKINNY=1 ---
if os.environ.get("SWIFT_SKINNY") == "1":
    try:
        import swift_skinny_patch

        swift_skinny_patch.install()
    except Exception as _exc2:  # pragma: no cover
        sys.stderr.write(f"[swift_skinny] sitecustomize failed to arm the hook: {type(_exc2).__name__}: {_exc2}" + chr(10))

# --- patch3: fused MoE router (SWIFT_ROUTER_FUSED=1) and side-stream L2 weight prefetch (SWIFT_L2PF=1) ---
if os.environ.get("SWIFT_ROUTER_FUSED") == "1":
    try:
        import swift_router_patch

        swift_router_patch.install()
    except Exception as _exc3:  # pragma: no cover
        sys.stderr.write(f"[swift_router] sitecustomize failed to arm the hook: {type(_exc3).__name__}: {_exc3}" + chr(10))
if os.environ.get("SWIFT_L2PF") == "1":
    try:
        import swift_l2pf

        swift_l2pf.install()
    except Exception as _exc4:  # pragma: no cover
        sys.stderr.write(f"[swift_l2pf] sitecustomize failed to arm the hook: {type(_exc4).__name__}: {_exc4}" + chr(10))

def _chain_next_sitecustomize():
    """We shadow any other sitecustomize on sys.path (e.g. the conda env's); run the next one, if any."""
    if getattr(sys, "_swift_sitecustomize_chaining", False):  # recursion guard (same dir reached via a symlink)
        return
    sys._swift_sitecustomize_chaining = True
    try:
        import importlib.machinery
        import importlib.util

        here = os.path.dirname(os.path.abspath(__file__))
        for entry in sys.path:
            if not entry or os.path.abspath(entry) == here:
                continue
            spec = importlib.machinery.PathFinder.find_spec("sitecustomize", [entry])
            if spec is not None and spec.loader is not None:
                if spec.origin and os.path.samefile(spec.origin, __file__):
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return
    except Exception:
        pass


_chain_next_sitecustomize()
