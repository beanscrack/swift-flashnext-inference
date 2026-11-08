"""Interpreter start-up hook for the P2b hidden-state capture (swift_hidden_capture.py).

Put /u701/j692wu/swift-opt/mtp/p2b on PYTHONPATH.  Inert unless SWIFT_HIDDEN_CAPTURE_DIR is set.
Stdlib only.  After arming, runs the next sitecustomize on sys.path (release/hooks), so the
release hooks still run.  Never raises.
"""
import os
import sys

if os.environ.get("SWIFT_HIDDEN_CAPTURE_DIR"):
    try:
        import swift_hidden_capture

        swift_hidden_capture.install()
    except Exception as _exc:  # pragma: no cover
        sys.stderr.write(f"[swift-hidden-capture] sitecustomize failed to arm the hook: {type(_exc).__name__}: {_exc}\n")


def _chain_next_sitecustomize():
    if getattr(sys, "_swift_sitecustomize_chaining", False):
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
