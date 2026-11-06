# SPDX-License-Identifier: Apache-2.0
"""swift_l2pf.py - side-stream L2 weight prefetch for the Qwen4Exp verify CUDA graph (vLLM 0.31), out-of-tree.

What it does (per decoder layer of the MAIN model only; the MTP drafter is a different class and is untouched):
  fork A  forward post-hook on the layer's big input projection (GDN linear_attn.in_proj_qkvz, QSA self_attn.qkv_proj):
          right after that GEMM is enqueued, the prefetch stream waits on the current (capture) stream and a small
          prefetch kernel pulls this layer's out_proj/o_proj weight (31.5 MB), then the MLP hyper-connection weights
          (13.4 MB) and the router weight (2.6 MB) into L2.  It overlaps the ~33-35 us of latency-bound kernels that
          precede out_proj (GDN: in_proj_ba, fill, conv1d, gdn_core; QSA: rope/indexer/topk/sparse-attn kernels).
  fork B  (SWIFT_L2PF_PHASE2=1, experimental) forward pre-hook on layer.mlp: prefetch the NEXT layer's attention HC
          weights and the first SWIFT_L2PF_NEXT_INPROJ_MB of its in-proj weight with an evict_last policy so they
          survive the Marlin traffic; they are demoted back to evict_normal (applypriority) at the next fork A.
  join    the wrapped Qwen4ExpModel.forward makes the current stream wait for the prefetch stream once, at the very
          end of the forward (the last prefetch finished ~100 us earlier, so the join never stalls).  The main stream
          never waits on a prefetch anywhere else.  Under CUDA-graph capture this is a fork/join of the capture stream,
          i.e. the prefetch kernels become a side branch of the graph.

Safety: the prefetch kernels only READ weights (output tokens are bit-identical by construction - still validate);
no allocation or host sync happens inside capture; by default prefetch runs only while a CUDA graph is being
captured for <= SWIFT_L2PF_MAX_TOKENS tokens (decode / verify), never in eager prefill.  Triton kernels are compiled
in the eager warm-up forward that vLLM runs before each capture; if the first opportunity is already inside a capture
the graph is captured WITHOUT prefetch (logged).  Requires compilation mode 0 (no torch.compile), as used here.

Environment (all optional except SWIFT_L2PF=1):
  SWIFT_L2PF=1                      enable; otherwise install() is a strict no-op (nothing imported or patched)
  SWIFT_L2PF_TARGETS=o_proj,mlp_hc,router   fork-A targets in priority order (also: shared, idx)
  SWIFT_L2PF_BUDGET_MB=48           max MB per fork-A launch (truncates the target list in order)
  SWIFT_L2PF_MODE=touch             touch | touch_el | pf | pf_el | bulk | bulk_el   (see l2pf_kernels.py)
  SWIFT_L2PF_CTAS=48                prefetch CTAs (touch: in-flight bytes = CTAS x BLOCK x 16)
  SWIFT_L2PF_BLOCK=0                kernel BLOCK override (0 = mode default)
  SWIFT_L2PF_RATE_TBPS=0            pace the issue rate (TB/s); default 0 for touch, 1.0 for pf/bulk modes
  SWIFT_L2PF_DELAY_US=0             start delay after the fork point
  SWIFT_L2PF_FORK=inproj            inproj (after in_proj_qkvz / qkv_proj) | late (after in_proj_ba / index_qk_proj)
  SWIFT_L2PF_LAYERS=all             e.g. "4-46" or "0-47:2"
  SWIFT_L2PF_TYPES=GDN,QSA          layer types to arm
  SWIFT_L2PF_MAX_TOKENS=8           only for forwards with <= this many tokens
  SWIFT_L2PF_EAGER=0                1 = also prefetch in eager (non-captured) small-batch forwards
  SWIFT_L2PF_PHASE2=0               enable fork B (see above);  SWIFT_L2PF_NEXT_INPROJ_MB=16
  SWIFT_L2PF_VERBOSE=0              1 = log the per-layer schedule at arm time
"""
import importlib.abc
import os
import sys
import traceback

TARGET = "vllm.models.qwen4_exp.nvidia.model"
_installed = False
_MB = 1 << 20


def _log(msg):
    sys.stderr.write("[swift_l2pf] %s\n" % msg)
    sys.stderr.flush()


def _parse_layers(spec, n):
    if not spec or spec == "all":
        return set(range(n))
    out = set()
    for part in spec.split(","):
        part, _, step = part.partition(":")
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b) if b else int(a)
        out.update(range(lo, hi + 1, int(step) if step else 1))
    return {i for i in out if 0 <= i < n}


class Config:
    def __init__(self, env=None):
        e = os.environ if env is None else env
        self.targets = [t.strip() for t in e.get("SWIFT_L2PF_TARGETS", "o_proj,mlp_hc,router").split(",") if t.strip()]
        self.budget = int(float(e.get("SWIFT_L2PF_BUDGET_MB", "48")) * 1e6)
        self.mode = e.get("SWIFT_L2PF_MODE", "touch")
        self.ctas = int(e.get("SWIFT_L2PF_CTAS", "48"))
        self.block = int(e.get("SWIFT_L2PF_BLOCK", "0")) or None
        default_rate = "0" if self.mode.startswith("touch") else "1.0"
        self.rate = float(e.get("SWIFT_L2PF_RATE_TBPS", default_rate))
        self.delay_ns = int(float(e.get("SWIFT_L2PF_DELAY_US", "0")) * 1000)
        self.fork = e.get("SWIFT_L2PF_FORK", "inproj")
        self.layers_spec = e.get("SWIFT_L2PF_LAYERS", "all")
        self.types = {t.strip().upper() for t in e.get("SWIFT_L2PF_TYPES", "GDN,QSA").split(",") if t.strip()}
        self.max_tokens = int(e.get("SWIFT_L2PF_MAX_TOKENS", "8"))
        self.eager = e.get("SWIFT_L2PF_EAGER", "0") == "1"
        self.phase2 = e.get("SWIFT_L2PF_PHASE2", "0") == "1"
        self.next_inproj = int(float(e.get("SWIFT_L2PF_NEXT_INPROJ_MB", "16")) * 1e6)
        self.verbose = e.get("SWIFT_L2PF_VERBOSE", "0") == "1"

    def el_mode(self):
        return {"touch": "touch_el", "pf": "pf_el", "bulk": "bulk_el"}.get(self.mode, self.mode)

    def __repr__(self):
        return ("targets=%s budget=%.1fMB mode=%s ctas=%d block=%s rate=%.2fTB/s delay=%dns fork=%s layers=%s types=%s "
                "max_tokens=%d eager=%s phase2=%s next_inproj=%.1fMB" % (
                    self.targets, self.budget / 1e6, self.mode, self.ctas, self.block, self.rate, self.delay_ns,
                    self.fork, self.layers_spec, sorted(self.types), self.max_tokens, self.eager, self.phase2,
                    self.next_inproj / 1e6))


def _w(mod, *path):
    """Follow an attribute path and return .weight (or None)."""
    for p in path:
        mod = getattr(mod, p, None)
        if mod is None:
            return None
    w = getattr(mod, "weight", None)
    return w if w is not None and hasattr(w, "data_ptr") else None


def _hc_weights(hc):
    if hc is None:
        return []
    down = _w(hc, "input_mix_weight_down_block_inject")
    if down is None:
        down = _w(hc, "input_mix_weight_down")
    return [down, _w(hc, "input_mix_weight_up")]


def layer_type(layer):
    if getattr(layer, "linear_attn", None) is not None:
        return "GDN"
    if getattr(layer, "self_attn", None) is not None:
        return "QSA"
    return None


def inproj_module(layer):
    t = layer_type(layer)
    if t == "GDN":
        return layer.linear_attn.in_proj_qkvz
    if t == "QSA":
        return layer.self_attn.qkv_proj
    return None


def fork_module(layer, fork):
    t = layer_type(layer)
    if fork == "late":
        if t == "GDN":
            return getattr(layer.linear_attn, "in_proj_ba", None)
        if t == "QSA":
            idx = getattr(layer.self_attn, "indexer", None)
            return getattr(idx, "index_qk_proj", None) if idx is not None else None
    return inproj_module(layer)


def fork_a_tensors(layer, targets):
    """Weights consumed after fork A in this layer, in the requested priority order."""
    t = layer_type(layer)
    out = []
    for name in targets:
        if name == "o_proj":
            out.append(_w(layer, "linear_attn", "out_proj") if t == "GDN" else _w(layer, "self_attn", "o_proj"))
        elif name == "mlp_hc":
            out += _hc_weights(getattr(layer, "mlp_hyper_connection", None))
        elif name == "router":
            out.append(_w(layer, "mlp", "gate"))
        elif name == "shared":
            out += [_w(layer, "mlp", "shared_expert", "gate_up_proj"), _w(layer, "mlp", "shared_expert", "down_proj")]
        elif name == "idx" and t == "QSA":
            out.append(_w(layer, "self_attn", "indexer", "index_qk_proj"))
    return [x for x in out if x is not None]


def fork_b_tensors(next_layer):
    out = list(_hc_weights(getattr(next_layer, "attn_hyper_connection", None)))
    ip = inproj_module(next_layer)
    out.append(getattr(ip, "weight", None) if ip is not None else None)
    return [x for x in out if x is not None]


class _State:
    """Per-Qwen4ExpModel prefetch state.  Built lazily on the first forward (weights are loaded by then)."""

    def __init__(self, model, cfg):
        import torch
        import l2pf_kernels as K
        self.torch, self.K, self.cfg, self.model = torch, K, cfg, model
        self.pf = None
        self.sink = None
        self.warm = False
        self.active = False
        self.launched = False
        self._warned_capture = False
        layers = list(model.layers)
        start = getattr(model, "start_layer", 0)
        end = getattr(model, "end_layer", len(layers))
        chosen = _parse_layers(cfg.layers_spec, len(layers))
        self.plans = {}
        self.handles = []
        for i in range(start, end):
            layer = layers[i]
            t = layer_type(layer)
            if t is None or t not in cfg.types or i not in chosen:
                continue
            fm = fork_module(layer, cfg.fork)
            if fm is None:
                continue
            plan = {"idx": i, "type": t, "layer": layer, "segsA": [], "segsB": [], "demote": []}
            self.plans[i] = plan
            self.handles.append(fm.register_forward_hook(self._make_hook_a(plan)))
            if cfg.phase2 and i + 1 < end and getattr(layer, "mlp", None) is not None:
                plan["next"] = layers[i + 1]
                self.handles.append(layer.mlp.register_forward_pre_hook(self._make_hook_b(plan)))
        self.refresh_segments()
        nb = sum(n for p in self.plans.values() for _, n in p["segsA"])
        ng = sum(1 for p in self.plans.values() if p["type"] == "GDN")
        _log("armed %d layers (GDN %d, QSA %d); fork-A bytes per forward %.1f MB; %r" % (
            len(self.plans), ng, len(self.plans) - ng, nb / 1e6, cfg))
        if cfg.verbose:
            for i, p in sorted(self.plans.items()):
                _log("  L%-2d %s A=%s B=%s" % (i, p["type"], ["%.2fMB" % (n / 1e6) for _, n in p["segsA"]],
                                                ["%.2fMB" % (n / 1e6) for _, n in p["segsB"]]))

    # -- schedule ------------------------------------------------------------------------------------------------
    def refresh_segments(self):
        K, cfg = self.K, self.cfg
        for i, p in self.plans.items():
            p["segsA"] = K.segments_from_tensors(fork_a_tensors(p["layer"], cfg.targets), cfg.budget)
            p["segsB"] = []
            p["demote"] = []
        if cfg.phase2:
            for i, p in self.plans.items():
                nxt = p.get("next")
                if nxt is None:
                    continue
                ts = fork_b_tensors(nxt)
                hc, ip = ts[:-1], ts[-1:]
                segs = K.segments_from_tensors(hc) + K.segments_from_tensors(ip, cfg.next_inproj)
                p["segsB"] = segs
                if i + 1 in self.plans:
                    self.plans[i + 1]["demote"] = segs

    # -- per forward ---------------------------------------------------------------------------------------------
    def begin(self, num_tokens):
        torch = self.torch
        capturing = torch.cuda.is_current_stream_capturing()
        small = bool(self.plans) and num_tokens is not None and num_tokens <= self.cfg.max_tokens
        if small and not capturing and not self.warm:
            self._warmup()           # vLLM's eager warm-up forward before each capture lands here
        ok = small and (capturing or self.cfg.eager)
        if ok and not self.warm:
            if not self._warned_capture:
                _log("first prefetch opportunity is inside a CUDA-graph capture and the kernels are not compiled "
                     "yet: this graph is captured WITHOUT prefetch")
                self._warned_capture = True
            ok = False
        if ok:
            self.refresh_segments()  # cheap; makes the captured pointers match the current weights
        self.active = ok
        self.launched = False

    def end(self):
        if self.launched:
            self.torch.cuda.current_stream().wait_stream(self.pf)
        self.launched = False
        self.active = False

    def _ensure_stream(self):
        if self.pf is None:
            torch = self.torch
            self.pf = torch.cuda.Stream()
            self.sink = torch.zeros(max(self.cfg.ctas, 1) * 256, dtype=torch.int32, device="cuda")

    def _launch(self, segs, mode):
        cfg = self.cfg
        rate = cfg.rate if cfg.rate > 0 else None
        self.K.launch(segs, self.sink, mode=mode, ctas=cfg.ctas, delay_ns=cfg.delay_ns, rate_tbps=rate,
                      block=cfg.block)

    def _warmup(self):
        """Eager, outside capture: issue every launch the captured schedule will issue, with identical arguments
        (all plans, fork A / demote / fork B), so Triton compiles and loads every variant now.  Costs one pass over
        the prefetch set (~2.3 GB of reads, a few ms) once per process."""
        torch = self.torch
        try:
            self._ensure_stream()
            self.refresh_segments()
            main = torch.cuda.current_stream()
            self.pf.wait_stream(main)
            n = 0
            with torch.cuda.stream(self.pf):
                for p in self.plans.values():
                    if self.cfg.phase2 and p["demote"]:
                        self.K.launch(p["demote"], self.sink, mode="demote", ctas=self.cfg.ctas, block=None)
                        n += 1
                    if p["segsA"]:
                        self._launch(p["segsA"], self.cfg.mode)
                        n += 1
                    if self.cfg.phase2 and p["segsB"]:
                        self._launch(p["segsB"], self.cfg.el_mode())
                        n += 1
            main.wait_stream(self.pf)
            self.warm = True
            _log("warm-up done: %d launches compiled/issued eagerly" % n)
        except Exception:
            _log("warm-up failed; prefetch disabled:\n" + traceback.format_exc())
            self.plans = {}

    def _fork(self, segs, mode, demote=None):
        torch = self.torch
        main = torch.cuda.current_stream()
        self.pf.wait_stream(main)          # fork point = the work just enqueued on the main/capture stream
        with torch.cuda.stream(self.pf):
            if demote:
                self.K.launch(demote, self.sink, mode="demote", ctas=self.cfg.ctas, block=None)
            self._launch(segs, mode)
        self.launched = True

    def _make_hook_a(self, plan):
        def hook(module, args, output):
            if self.active and plan["segsA"]:
                self._fork(plan["segsA"], self.cfg.mode, plan["demote"] if self.cfg.phase2 else None)
            return None
        return hook

    def _make_hook_b(self, plan):
        def hook(module, args):
            if self.active and plan["segsB"]:
                self._fork(plan["segsB"], self.cfg.el_mode())
            return None
        return hook


def _num_tokens(args, kwargs):
    pos = kwargs.get("positions")
    if pos is None and len(args) >= 2:
        pos = args[1]
    if pos is not None and hasattr(pos, "shape"):
        return int(pos.shape[-1])
    ids = kwargs.get("input_ids", args[0] if args else None)
    if ids is not None and hasattr(ids, "shape"):
        return int(ids.shape[0])
    emb = kwargs.get("inputs_embeds")
    if emb is not None and hasattr(emb, "shape"):
        return int(emb.shape[0])
    return None


def _patch(mod):
    cls = getattr(mod, "Qwen4ExpModel")
    orig = cls.forward
    if getattr(orig, "_swift_l2pf", False):
        return
    cfg = Config()

    def forward(self, *args, **kwargs):
        st = self.__dict__.get("_swift_l2pf_state")
        if st is None:
            try:
                st = _State(self, cfg)
            except Exception:
                _log("arming failed; model left unmodified:\n" + traceback.format_exc())
                st = False
            self.__dict__["_swift_l2pf_state"] = st
        if not st:
            return orig(self, *args, **kwargs)
        st.begin(_num_tokens(args, kwargs))
        try:
            return orig(self, *args, **kwargs)
        finally:
            st.end()

    forward._swift_l2pf = True
    forward.__wrapped__ = orig
    cls.forward = forward
    _log("patched %s.Qwen4ExpModel.forward (%r)" % (TARGET, cfg))


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

                spec.loader.exec_module = exec_module
                return spec
        return None


def install(target=TARGET, force=False):
    """Idempotent.  Strict no-op unless SWIFT_L2PF=1 (or force=True)."""
    global _installed
    if _installed or not (force or os.environ.get("SWIFT_L2PF") == "1"):
        return
    _installed = True
    if target in sys.modules:
        _patch(sys.modules[target])
    else:
        sys.meta_path.insert(0, _Finder(target))
