#!/usr/bin/env python3
"""bench_skinny.py - skinny (M <= 8) BF16 GEMM benchmark for the Swift/Qwen4Exp dense layers.

Purpose (lever L2): compare, for every distinct dense weight shape [N, K] of the model at
M in {1,2,4,8} tokens (M=4 is the 3-token-MTP verify batch, M=1 the draft steps):

  torch        torch.nn.functional.linear (cuBLAS) - exactly what vLLM's UnquantizedLinearMethod calls
  torch_lt     same, with torch.backends.cuda.preferred_blas_library("cublaslt")   (--impls torch_lt)
  triton_fma   Triton GEMV-style kernel, CUDA-core FMA, fp32 accumulate, vectorised (16 B) weight loads,
               intra-CTA K parallelism, optional split-K over CTAs with a 2-stage (workspace + reduce) scheme.
               No atomics -> bitwise deterministic.
  triton_dot   Triton tensor-core (mma.sync) kernel, M padded to 16, cp.async pipelined weight stream,
               same optional split-K/2-stage reduction.
  cute         vLLM's own CuTe-DSL CuteSkinnyGemm (vllm.model_executor.kernels.linear.cute_dsl), swept over
               SkinnyGemmConfig (vLLM only enables it on SM90/SM103 with TP=4 shapes, not on SM120).
  flashinfer   flashinfer.mm_bf16 backends valid on SM120: tinygemm, cublaslt, cudnn (+cutile with --cutile)
  hcvllm       (shape hc_down only) vLLM's fused hc_down_silu FMA kernel, GEMM part only, tb-size sweep

Methodology
  * weights are cycled through R distinct copies (R*W >= flush-factor * L2 size) and the timed unit is a
    CUDA graph containing G*R back-to-back GEMMs (one per copy), replayed and timed with CUDA events.
    So every call streams its weights from HBM like layer-by-layer decode does, launch overhead is
    excluded exactly as in vLLM's FULL_DECODE_ONLY cudagraph.  --hot re-times with R=1 (L2-resident).
  * achieved GB/s = weight bytes / time (activation traffic ignored: it is <=0.1% of the bytes);
    %peak is against --peak-gbps (default 1790 = nominal 1.79 TB/s).  A Triton streaming-read kernel is run
    first (--no-calib to skip) to show the empirically reachable ceiling on this card/power cap.
  * every variant is checked against an fp32 reference (x.float() @ w.float().T): max abs error, max error
    relative to max|ref|, bit-exact repeatability, and the fraction of outputs that differ from torch's bf16
    result.  Variants above --tol are excluded from "best".

Usage (GPU, inference env):
  python bench_skinny.py --budget quick --json quick.json            # ~10-15 min incl. JIT compiles
  python bench_skinny.py --shapes qkvz,gdn_ba --ms 4 --budget std
  python bench_skinny.py --selftest-cpu                              # no GPU: Triton interpreter correctness
  python bench_skinny.py --list
Set TRITON_CACHE_DIR to a node-local directory (e.g. /tmp/triton_$USER) to avoid slow NFS JIT caches.
"""
import os
import sys

if "--selftest-cpu" in sys.argv:          # must happen before torch/triton are imported
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import dataclasses
import json
import math
import statistics
import time
import traceback

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

PEAK_GBPS_DEFAULT = 1790.0


# =====================================================================================
# Triton kernels
# =====================================================================================
@triton.jit
def _fma_acc(acc, w, x_ptr, k_off, row_off, k_mask, EVEN_K: tl.constexpr):
    if EVEN_K:
        xr = tl.load(x_ptr + row_off + k_off)
    else:
        xr = tl.load(x_ptr + row_off + k_off, mask=k_mask, other=0.0)
    return acc + w * xr.to(tl.float32)[None, :]


@triton.jit
def _fma_store(acc, row: tl.constexpr, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om,
               DIRECT: tl.constexpr):
    o = tl.sum(acc, axis=1)
    if DIRECT:
        tl.store(out_ptr + row * stride_om + offs_n, o.to(out_ptr.dtype.element_ty), mask=n_mask)
    else:
        tl.store(ws_ptr + (pid_k * M + row) * N + offs_n, o, mask=n_mask)


@triton.jit
def _fma_kernel(x_ptr, w_ptr, out_ptr, ws_ptr, M, N, K, k_per_split,
                stride_xm, stride_wn, stride_om,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                EVEN_K: tl.constexpr, EVEN_N: tl.constexpr, DIRECT: tl.constexpr,
                PDL: tl.constexpr):
    """C[m, n] = sum_k X[m, k] * W[n, k].  One CTA = BLOCK_N weight rows x one K slice.

    Every thread owns a [BLOCK_N, BLOCK_K]/threads slab of fp32 accumulators for each of the BLOCK_M (==M)
    activation rows, so each weight element is loaded exactly once (16 B vectors along K) and used for all
    M rows; the cross-thread reduction happens once at the end (like vLLM's CuteSkinnyGemm).
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_begin = pid_k * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, K)
    w_row = w_ptr + offs_n[:, None] * stride_wn

    acc0 = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    if BLOCK_M > 1:
        acc1 = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    if BLOCK_M > 2:
        acc2 = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    if BLOCK_M > 3:
        acc3 = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    if BLOCK_M > 4:
        acc4 = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    if BLOCK_M > 5:
        acc5 = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    if BLOCK_M > 6:
        acc6 = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    if BLOCK_M > 7:
        acc7 = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)

    for k0 in range(0, k_per_split, BLOCK_K):
        k_off = k_begin + k0 + offs_k
        k_mask = k_off < k_end
        if EVEN_K:
            if EVEN_N:
                w = tl.load(w_row + k_off[None, :], eviction_policy="evict_first")
            else:
                w = tl.load(w_row + k_off[None, :], mask=n_mask[:, None], other=0.0,
                            eviction_policy="evict_first")
        else:
            w = tl.load(w_row + k_off[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                        eviction_policy="evict_first")
        w = w.to(tl.float32)
        if PDL:
            # weights above do not depend on the previous kernel; only X does
            tl.extra.cuda.gdc_wait()
        acc0 = _fma_acc(acc0, w, x_ptr, k_off, 0, k_mask, EVEN_K)
        if BLOCK_M > 1:
            acc1 = _fma_acc(acc1, w, x_ptr, k_off, stride_xm, k_mask, EVEN_K)
        if BLOCK_M > 2:
            acc2 = _fma_acc(acc2, w, x_ptr, k_off, 2 * stride_xm, k_mask, EVEN_K)
        if BLOCK_M > 3:
            acc3 = _fma_acc(acc3, w, x_ptr, k_off, 3 * stride_xm, k_mask, EVEN_K)
        if BLOCK_M > 4:
            acc4 = _fma_acc(acc4, w, x_ptr, k_off, 4 * stride_xm, k_mask, EVEN_K)
        if BLOCK_M > 5:
            acc5 = _fma_acc(acc5, w, x_ptr, k_off, 5 * stride_xm, k_mask, EVEN_K)
        if BLOCK_M > 6:
            acc6 = _fma_acc(acc6, w, x_ptr, k_off, 6 * stride_xm, k_mask, EVEN_K)
        if BLOCK_M > 7:
            acc7 = _fma_acc(acc7, w, x_ptr, k_off, 7 * stride_xm, k_mask, EVEN_K)

    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    _fma_store(acc0, 0, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT)
    if BLOCK_M > 1:
        _fma_store(acc1, 1, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT)
    if BLOCK_M > 2:
        _fma_store(acc2, 2, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT)
    if BLOCK_M > 3:
        _fma_store(acc3, 3, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT)
    if BLOCK_M > 4:
        _fma_store(acc4, 4, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT)
    if BLOCK_M > 5:
        _fma_store(acc5, 5, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT)
    if BLOCK_M > 6:
        _fma_store(acc6, 6, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT)
    if BLOCK_M > 7:
        _fma_store(acc7, 7, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT)


@triton.jit
def _dot_kernel(x_ptr, w_ptr, out_ptr, ws_ptr, M, N, K, k_per_split,
                stride_xm, stride_wn, stride_om,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                EVEN_K: tl.constexpr, EVEN_N: tl.constexpr, DIRECT: tl.constexpr,
                PDL: tl.constexpr):
    """Tensor-core variant: acc[16, BLOCK_N] += X[16(pad), BK] @ W[BN, BK]^T, weights streamed through the
    Triton software pipeline (num_stages cp.async stages).  M (<=16) rows are masked, padded rows are zero."""
    BLOCK_M: tl.constexpr = 16
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    k_begin = pid_k * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k_begin + offs_k)[None, :]
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + (k_begin + offs_k)[:, None]
    if PDL:
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k0 in range(0, k_per_split, BLOCK_K):
        if EVEN_K:
            x = tl.load(x_ptrs + k0, mask=m_mask[:, None], other=0.0)
            if EVEN_N:
                w = tl.load(w_ptrs + k0, eviction_policy="evict_first")
            else:
                w = tl.load(w_ptrs + k0, mask=n_mask[None, :], other=0.0, eviction_policy="evict_first")
        else:
            k_mask = (k_begin + k0 + offs_k) < k_end
            x = tl.load(x_ptrs + k0, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs + k0, mask=n_mask[None, :] & k_mask[:, None], other=0.0,
                        eviction_policy="evict_first")
        acc = tl.dot(x, w, acc)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    mn_mask = m_mask[:, None] & n_mask[None, :]
    if DIRECT:
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
                 acc.to(out_ptr.dtype.element_ty), mask=mn_mask)
    else:
        tl.store(ws_ptr + (pid_k * M + offs_m[:, None]) * N + offs_n[None, :], acc, mask=mn_mask)


@triton.jit
def _reduce_kernel(ws_ptr, out_ptr, M, N, stride_om, SPLIT: tl.constexpr, BLOCK: tl.constexpr,
                   PDL: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = M * N
    mask = offs < total
    if PDL:
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(ws_ptr + s * total + offs, mask=mask, other=0.0)
    m = offs // N
    n = offs - m * N
    tl.store(out_ptr + m * stride_om + n, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _stream_read_kernel(p_ptr, o_ptr, per_prog, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * per_prog
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, per_prog, BLOCK):
        v = tl.load(p_ptr + base + i + offs, eviction_policy="evict_first")
        acc += v.to(tl.float32)
    tl.store(o_ptr + pid, tl.sum(acc, axis=0))


# =====================================================================================
# Triton wrapper
# =====================================================================================
@dataclasses.dataclass(frozen=True)
class TConf:
    kind: str                 # "fma" | "dot"
    block_n: int
    block_k: int
    split_k: int = 1          # requested split; the effective split is derived from K
    num_warps: int = 4
    num_stages: int = 3       # "dot" only
    pdl: bool = False

    def label(self):
        s = "%s bn=%d bk=%d sk=%d w=%d" % (self.kind, self.block_n, self.block_k, self.split_k, self.num_warps)
        if self.kind == "dot":
            s += " st=%d" % self.num_stages
        if self.pdl:
            s += " pdl"
        return s


def _cdiv(a, b):
    return -(-a // b)


def split_geometry(K, block_k, split_k):
    """-> (k_per_split, effective_split, even_k)"""
    kb = _cdiv(K, block_k)
    sk = max(1, min(split_k, kb))
    kps = _cdiv(kb, sk) * block_k
    seff = _cdiv(K, kps)
    return kps, seff, (K % kps == 0)


def triton_skinny_gemm(x, w, cfg, out=None):
    """out[M,N] (bf16/fp16 as x) = x[M,K] @ w[N,K]^T with fp32 accumulation.  CUDA-graph safe: no host sync;
    the split-K workspace comes from torch.empty (graph-pool allocation under capture)."""
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K and x.stride(1) == 1 and w.stride(1) == 1
    assert K % 8 == 0 and x.stride(0) % 8 == 0 and w.stride(0) % 8 == 0, "need 16 B aligned rows"
    if cfg.kind == "fma":
        assert 1 <= M <= 8
    else:
        assert 1 <= M <= 16
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    kps, seff, even_k = split_geometry(K, cfg.block_k, cfg.split_k)
    even_n = (N % cfg.block_n == 0)
    direct = (seff == 1)
    ws = out if direct else torch.empty((seff, M, N), dtype=torch.float32, device=x.device)
    grid = (_cdiv(N, cfg.block_n), seff)
    extra = {"launch_pdl": True} if cfg.pdl else {}
    if cfg.kind == "fma":
        _fma_kernel[grid](x, w, out, ws, M, N, K, kps, x.stride(0), w.stride(0), out.stride(0),
                          BLOCK_M=M, BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k,
                          EVEN_K=even_k, EVEN_N=even_n, DIRECT=direct, PDL=cfg.pdl,
                          num_warps=cfg.num_warps, **extra)
    else:
        _dot_kernel[grid](x, w, out, ws, M, N, K, kps, x.stride(0), w.stride(0), out.stride(0),
                          BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k,
                          EVEN_K=even_k, EVEN_N=even_n, DIRECT=direct, PDL=cfg.pdl,
                          num_warps=cfg.num_warps, num_stages=cfg.num_stages, **extra)
    if not direct:
        blk = 1024 if M * N >= 4096 else 256
        _reduce_kernel[(_cdiv(M * N, blk),)](ws, out, M, N, out.stride(0), SPLIT=seff, BLOCK=blk,
                                             PDL=cfg.pdl, num_warps=4, **extra)
    return out


# =====================================================================================
# Shape table (from the Nsight mapping, see INTEGRATION.md for the evidence)
# =====================================================================================
@dataclasses.dataclass(frozen=True)
class Shape:
    name: str
    n: int
    k: int
    calls: float        # calls per verify cycle (M=4 graph)
    crit_lo: float      # fraction of its kernel time that is NOT hidden behind a parallel branch (nsys)
    layers: str


SHAPES = [
    Shape("qkvz",       16384, 2560, 36, 1.00, "GDN in_proj_qkvz (qkv 10240 + z 6144 merged)"),
    Shape("o_proj",      2560, 6144, 48, 1.00, "GDN out_proj (36) + QSA o_proj (12)"),
    Shape("qsa_qkv",    13312, 2560, 12, 1.00, "QSA qkv_proj (q+gate 12288, k 512, v 512)"),
    Shape("ple_kv",     12800, 2560,  1, 1.00, "PLE kv_proj (key 10240 + value 2560) - guess"),
    Shape("hc_up",      10240,  320, 97, 1.00, "HC input_mix_weight_up (2/layer + final)"),
    Shape("hc_down",      336, 10240, 96, 0.91, "HC merged down+inject (vLLM uses fused CuTe hc_down_silu)"),
    Shape("hc_final",     320, 10240,  1, 1.00, "final HC down"),
    Shape("sh_gate_up",  1280, 2560, 48, 0.07, "shared-expert gate_up (overlaps routed branch)"),
    Shape("sh_down",     2560,  640, 48, 0.04, "shared-expert down_proj (overlaps routed branch)"),
    Shape("router",       512, 2560, 48, 0.46, "MoE router GateLinear (out_dtype=None -> tier-5 F.linear, bf16 out)"),
    Shape("indexer",      640, 2560, 12, 1.00, "QSA indexer qk proj"),
    Shape("gdn_ba",        96, 2560, 36, 1.00, "GDN in_proj_ba (b,a merged) - N=96 is inferred"),
    Shape("sh_gate1",       1, 2560, 48, 0.11, "shared_expert_gate (N=1)"),
    Shape("lm_head",   248320, 2560,  1, 1.00, "lm_head (main M=4; draft M=1 x3 per cycle)"),
]
SHAPE_BY_NAME = {s.name: s for s in SHAPES}


# =====================================================================================
# helpers
# =====================================================================================
def device_info(dev):
    p = torch.cuda.get_device_properties(dev)
    l2 = getattr(p, "L2_cache_size", 0) or (96 << 20)
    return p.name, p.multi_processor_count, l2


def alloc_weights(n, k, l2_bytes, flush_factor, max_gb, dev, hot=False):
    wb = n * k * 2
    if hot:
        r = 1
    else:
        r = max(2 if wb < 4 * l2_bytes else 1, _cdiv(int(flush_factor * l2_bytes), wb))
        r = min(r, max(1, int(max_gb * 1e9 // wb)), 1024)
    big = torch.empty((r, n, k), dtype=torch.bfloat16, device=dev)
    big.normal_(0.0, k ** -0.5)
    return [big[i] for i in range(r)], big


def make_x(m, k, dev):
    return torch.randn((m, k), dtype=torch.bfloat16, device=dev)


def time_calls(fn, x, ws, min_calls, replays, use_graph=True):
    """median/min us per call; one graph = G passes over all R weight copies."""
    R = len(ws)
    G = max(1, _cdiv(min_calls, R))

    def body():
        for _ in range(G):
            for w in ws:
                fn(x, w)

    body()
    body()
    torch.cuda.synchronize()
    eager = False
    runner = body
    if use_graph:
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
            runner = g.replay
        except Exception as e:  # pragma: no cover
            print("      [graph capture failed: %s: %s -> eager timing, CPU-launch bound]" %
                  (type(e).__name__, str(e)[:120]))
            eager = True
            torch.cuda.synchronize()
    else:
        eager = True
    runner()
    runner()
    torch.cuda.synchronize()
    ts = []
    for _ in range(replays):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        runner()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / (G * R))
    return statistics.median(ts), min(ts), eager


def check_variant(fn, x, w, ref_bf16, ref32):
    out = fn(x, w)
    torch.cuda.synchronize()
    out2 = fn(x, w)
    torch.cuda.synchronize()
    det = bool(torch.equal(out, out2))
    err = (out.float() - ref32).abs()
    scale = ref32.abs().max().clamp_min(1e-6)
    max_abs = err.max().item()
    return dict(max_abs=max_abs, max_rel=max_abs / scale.item(),
                mism=(out != ref_bf16).float().mean().item(), det=det,
                finite=bool(torch.isfinite(out.float()).all().item()))


# ------------------------------------------------------------------ config generators
def _prior(ctas, sm, extra=0.0):
    return abs(math.log2(max(ctas, 1) / (6.0 * sm))) + extra


def _spread(cands, cap):
    if cap is None or len(cands) <= cap:
        return cands
    step = len(cands) / float(cap)
    return [cands[int(i * step)] for i in range(cap)]


def gen_fma_configs(n, k, m, sm, cap, pdl_modes):
    if m > 8:
        return []
    out = []
    for bn in (1, 2, 4, 8):
        for bk in (128, 256, 512, 1024):
            if bk > max(128, 2 * k):
                continue
            for nw in (1, 2, 4, 8):
                ept = bn * bk // (32 * nw)
                if ept < 8 or ept * m > 96:
                    continue
                seen = set()
                for sk in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40):
                    kps, seff, even = split_geometry(k, bk, sk)
                    if seff != sk or (kps, seff) in seen:
                        continue
                    seen.add((kps, seff))
                    ctas = _cdiv(n, bn) * seff
                    if ctas < min(sm // 2, 8) or ctas > 512 * sm:
                        continue
                    for pdl in pdl_modes:
                        out.append((_prior(ctas, sm, (0 if even else 0.7) + (0.4 if seff > 1 else 0.0)),
                                    TConf("fma", bn, bk, seff, nw, 0, pdl)))
    out.sort(key=lambda t: t[0])
    return _spread([c for _, c in out], cap)


def gen_dot_configs(n, k, m, sm, cap, pdl_modes):
    if m > 16:
        return []
    out = []
    for bn in (16, 32, 64, 128):
        for bk in (64, 128, 256, 512):
            if bk > max(64, 2 * k):
                continue
            for nw in (2, 4, 8):
                for st in (2, 3, 4, 5, 6):
                    if st * (bn * bk + 16 * bk) * 2 > 96 * 1024:
                        continue
                    seen = set()
                    for sk in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40):
                        kps, seff, even = split_geometry(k, bk, sk)
                        if seff != sk or (kps, seff) in seen:
                            continue
                        seen.add((kps, seff))
                        ctas = _cdiv(n, bn) * seff
                        if ctas < min(sm // 2, 8) or ctas > 128 * sm:
                            continue
                        if seff > 1 and k // seff < bk:
                            continue
                        for pdl in pdl_modes:
                            out.append((_prior(ctas, sm, (0 if even else 0.7) + (0.4 if seff > 1 else 0.0)
                                               + 0.15 * abs(st - 4)),
                                        TConf("dot", bn, bk, seff, nw, st, pdl)))
    out.sort(key=lambda t: t[0])
    return _spread([c for _, c in out], cap)


def cute_valid(cfg, n, k, m):
    if cfg.num_rows != m or n % cfg.outputs_per_block or k % (cfg.block_size * cfg.vector_width):
        return False
    if cfg.static_k is not None and (cfg.static_k != k or k < 2 * cfg.block_size * cfg.vector_width):
        return False
    return True


def gen_cute_configs(n, k, m, sm, cap):
    try:
        from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
            SkinnyGemmConfig, shape_dynamic_skinny_gemm)
    except Exception:
        return [], None
    if not shape_dynamic_skinny_gemm.is_available():
        return [], None
    cands = []
    try:
        cands.append(("vllm_default", shape_dynamic_skinny_gemm._config(m, n, k)))
    except Exception:
        pass
    try:   # the TP=4 plans shipped for SM103 transplanted to this shape (same M)
        from vllm.models.qwen4_exp.nvidia.low_latency_gemm import QWEN4_EXP_GEMM_PLANS
        for (pn, pk), mp in QWEN4_EXP_GEMM_PLANS.items():
            if m in mp:
                cands.append(("tp4_plan_%dx%d" % (pn, pk), mp[m]))
    except Exception:
        pass
    grid = []
    for bs in (32, 64, 128, 256):
        for opb in (1, 2, 4):
            for vw in (8, 4):
                for ku in (1, 2, 4):
                    for sk in (None, k):
                        c = SkinnyGemmConfig(m, bs, opb, ku, vw, sk)
                        if cute_valid(c, n, k, m):
                            ctas = _cdiv(n, opb)
                            grid.append((_prior(ctas, sm, 0.1 * (ku != 2) + 0.2 * (vw != 8) + 0.1 * (sk is None)), c))
    grid.sort(key=lambda t: t[0])
    cands += [("sweep", c) for _, c in _spread(grid, cap)]
    out, seen = [], set()
    for tag, c in cands:
        if c in seen or not cute_valid(c, n, k, m):
            continue
        seen.add(c)
        out.append((tag, c))
    return out, shape_dynamic_skinny_gemm


# ------------------------------------------------------------------ variants
class Variant:
    def __init__(self, family, name, fn, cfg=None):
        self.family, self.name, self.fn, self.cfg = family, name, fn, cfg


def _bind_triton(cfg):
    return lambda x, w: triton_skinny_gemm(x, w, cfg)


def _bind_cute(op, cfg):
    return lambda x, w: op(x, w, cfg)


def _bind_fi(backend):
    import flashinfer
    return lambda x, w: flashinfer.mm_bf16(x, w.t(), backend=backend)


def build_variants(shape, m, args, sm, notes):
    n, k = shape.n, shape.k
    impls = set(args.impls)
    vs = []
    if "torch" in impls:
        vs.append(Variant("torch", "torch.F.linear (cuBLAS)", lambda x, w: F.linear(x, w)))
    if "torch_lt" in impls:
        def f_lt(x, w):
            prev = torch.backends.cuda.preferred_blas_library()
            torch.backends.cuda.preferred_blas_library("cublaslt")
            try:
                return F.linear(x, w)
            finally:
                torch.backends.cuda.preferred_blas_library(prev)
        vs.append(Variant("torch_lt", "torch.F.linear (cublasLt preferred)", f_lt))
    pdl_modes = {"0": [False], "1": [True], "both": [False, True]}[args.pdl]
    caps = {"quick": (10, 10, 4), "std": (36, 36, 8), "full": (None, None, None)}[args.budget]
    if "triton_fma" in impls and k % 8 == 0:
        for c in gen_fma_configs(n, k, m, sm, caps[0], pdl_modes):
            vs.append(Variant("triton_fma", "triton " + c.label(), _bind_triton(c), c))
    if "triton_dot" in impls and k % 8 == 0:
        for c in gen_dot_configs(n, k, m, sm, caps[1], pdl_modes):
            vs.append(Variant("triton_dot", "triton " + c.label(), _bind_triton(c), c))
    if "cute" in impls:
        cfgs, op = gen_cute_configs(n, k, m, sm, caps[2])
        if op is None:
            notes.add("cute: vllm CuTe skinny GEMM unavailable (import or cutlass-dsl missing)")
        for tag, c in cfgs:
            vs.append(Variant("cute", "cute[%s] bs=%d opb=%d ku=%d vw=%d sk=%s" % (
                tag, c.block_size, c.outputs_per_block, c.k_unroll, c.vector_width, c.static_k),
                _bind_cute(op, c), c))
    if "flashinfer" in impls:
        try:
            import flashinfer  # noqa: F401
            backs = ["tinygemm", "cublaslt", "cudnn"] + (["cutile"] if args.cutile else [])
            for be in backs:
                if n % 16 and be == "tinygemm":
                    continue
                vs.append(Variant("flashinfer", "flashinfer.mm_bf16[%s]" % be, _bind_fi(be)))
            if n % 16 == 0:
                vs.append(Variant("flashinfer", "flashinfer.mm_bf16[tinygemm,pdl]",
                                  lambda x, w: flashinfer.mm_bf16(x, w.t(), pdl=True, backend="tinygemm")))
        except Exception as e:
            notes.add("flashinfer unavailable: %s: %s" % (type(e).__name__, str(e)[:100]))
    if "hcvllm" in impls and shape.name == "hc_down" and m <= 4:
        try:
            from vllm.models.qwen4_exp.nvidia.ops.cute_dsl.hc_down_silu import HcDownSiluGemm
            for tb in (64, 128, 256, 512):
                kern = HcDownSiluGemm(320, 4, k)
                kern.dispatch = (lambda tb_: lambda mm: ("fma", mm, k, tb_))(tb)
                vs.append(Variant("hcvllm", "vllm hc_down_silu FMA tb=%d (silu epilogue, error n/a)" % tb,
                                  (lambda kk: lambda x, w: kk(x, w))(kern)))
        except Exception as e:
            notes.add("hcvllm unavailable: %s: %s" % (type(e).__name__, str(e)[:100]))
    return vs


# ------------------------------------------------------------------ driver
def show_torch_kernel_names(x, w):
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            F.linear(x, w)
            torch.cuda.synchronize()
        names = []
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                names.append(e.name[:110])
        return names
    except Exception as e:
        return ["<profiler failed: %s>" % e]


def stream_read_bw(dev, sm):
    nbytes = 1 << 30
    buf = torch.empty(nbytes // 2, dtype=torch.bfloat16, device=dev).normal_()
    best = 0.0
    for progs_mult in (4, 8, 16):
        progs = progs_mult * sm
        per_prog = (buf.numel() // progs) // 4096 * 4096
        o = torch.empty(progs, dtype=torch.float32, device=dev)
        for blk, nw in ((2048, 8), (4096, 8)):
            def run():
                _stream_read_kernel[(progs,)](buf, o, per_prog, BLOCK=blk, num_warps=nw)
            run()
            run()
            torch.cuda.synchronize()
            ts = []
            for _ in range(7):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                run()
                e.record()
                e.synchronize()
                ts.append(s.elapsed_time(e) * 1e-3)
            gbs = progs * per_prog * 2 / statistics.median(ts) / 1e9
            best = max(best, gbs)
    del buf
    return best


def fmt_row(r, base_us, peak):
    return "  %-62s %8.2f %8.0f %6.1f%% %6.2fx  rel=%.1e mism=%4.1f%% %s%s" % (
        r["name"][:62], r["us"], r["gbs"], 100 * r["gbs"] / peak, base_us / r["us"], r["max_rel"],
        100 * r["mism"], "" if r["ok"] else "FAIL ", "(eager!)" if r.get("eager") else "")


def run_shape(shape, ms, args, dev, sm, l2, notes, results):
    n, k = shape.n, shape.k
    wbytes = n * k * 2
    ws, big = alloc_weights(n, k, l2, args.flush_factor, args.max_gb, dev)
    print("\n== %-10s N=%-6d K=%-5d W=%.2f MB  copies R=%d (%.0f MB cycled)  calls/cycle=%g  %s" % (
        shape.name, n, k, wbytes / 1e6, len(ws), len(ws) * wbytes / 1e6, shape.calls, shape.layers))
    for m in ms:
        x = make_x(m, k, dev)
        ref_bf16 = F.linear(x, ws[0])
        ref32 = x.float() @ ws[0].float().t()
        if args.show_torch_kernels and m in (1, 4):
            print("  M=%d torch kernel(s): %s" % (m, " | ".join(show_torch_kernel_names(x, ws[0]))))
        vs = build_variants(shape, m, args, sm, notes)
        rows = []
        # phase 1: coarse timing of everything; phase 2: re-time the best few precisely
        for v in vs:
            try:
                chk = check_variant(v.fn, x, ws[0], ref_bf16, ref32)
                us, us_min, eager = time_calls(v.fn, x, ws, args.min_calls_coarse, args.replays_coarse,
                                               not args.no_graph)
            except Exception as e:
                msg = "%s: %s" % (type(e).__name__, str(e).split("\n")[0][:100])
                if args.verbose:
                    traceback.print_exc()
                notes.add("variant failed [%s %s]: %s" % (shape.name, v.name[:40], msg))
                continue
            ok = v.family == "hcvllm" or (chk["finite"] and chk["det"] and chk["max_rel"] <= args.tol)
            cfg = None
            if v.cfg is not None:
                cfg = dataclasses.asdict(v.cfg) if dataclasses.is_dataclass(v.cfg) else repr(v.cfg)
            rows.append(dict(family=v.family, name=v.name, us=us, us_min=us_min, eager=eager, ok=ok,
                             gbs=wbytes / us / 1e3, cfg=cfg, _v=v, **chk))
        rows.sort(key=lambda r: r["us"])
        top = [r for r in rows if r["ok"]][: args.refine] + [r for r in rows if r["family"] in ("torch", "torch_lt")]
        seen = set()
        for r in top:
            if id(r) in seen:
                continue
            seen.add(id(r))
            try:
                us, us_min, eager = time_calls(r["_v"].fn, x, ws, args.min_calls, args.replays, not args.no_graph)
                r["us"], r["us_min"], r["eager"], r["gbs"] = us, us_min, eager, wbytes / us / 1e3
            except Exception:
                pass
        if args.hot:
            wsh, bigh = alloc_weights(n, k, l2, 1, args.max_gb, dev, hot=True)
            for r in top:
                try:
                    r["us_hot"] = time_calls(r["_v"].fn, x, wsh, args.min_calls, args.replays, not args.no_graph)[0]
                except Exception:
                    pass
            del bigh, wsh
        rows.sort(key=lambda r: r["us"])
        base = next((r for r in rows if r["family"] == "torch"), None)
        base_us = base["us"] if base else rows[0]["us"]
        print(" M=%d   %-62s %8s %8s %7s %7s" % (m, "variant", "us/call", "GB/s", "%peak", "vs torch"))
        shown = 0
        for r in rows:
            if shown < args.show or r["family"] in ("torch", "torch_lt"):
                print(fmt_row(r, base_us, args.peak_gbps) + ("  hot=%.2fus" % r["us_hot"] if "us_hot" in r else ""))
                shown += 1
        best = next((r for r in rows if r["ok"] and r["family"] not in ("torch", "torch_lt", "hcvllm")), None)
        results.append(dict(shape=shape.name, n=n, k=k, m=m, calls=shape.calls, crit_lo=shape.crit_lo,
                            torch_us=base_us,
                            best=({kk: vv for kk, vv in best.items() if not kk.startswith("_")} if best else None),
                            rows=[{kk: vv for kk, vv in r.items() if not kk.startswith("_")} for r in rows]))
        del vs, rows, top
        torch.cuda.empty_cache()
    del ws, big
    torch.cuda.empty_cache()


def summarize(results, args):
    peak = args.peak_gbps
    plan = {}
    for mm in sorted(set(r["m"] for r in results)):
        sel = [r for r in results if r["m"] == mm]
        print("\n== SUMMARY  M=%d  (saving = calls/cycle * (torch - best); save_lo weights by crit_lo = lower bound "
              "accounting for kernels hidden behind a parallel stream)" % mm)
        print("  %-10s %5s %9s %9s %7s %9s %9s  %s" % ("shape", "calls", "torch_us", "best_us", "%peak",
                                                       "save_us", "save_lo", "best variant"))
        tot, tot_lo, tot_t = 0.0, 0.0, 0.0
        for r in sel:
            b = r["best"]
            tot_t += r["calls"] * r["torch_us"]
            if not b or b["us"] >= r["torch_us"] * 0.97:
                print("  %-10s %5g %9.2f %9s %7s %9s %9s  (torch stays)" % (
                    r["shape"], r["calls"], r["torch_us"], "-", "-", "0", "0"))
                continue
            sv = r["calls"] * (r["torch_us"] - b["us"])
            tot += sv
            tot_lo += sv * r["crit_lo"]
            print("  %-10s %5g %9.2f %9.2f %6.1f%% %9.1f %9.1f  %s" % (
                r["shape"], r["calls"], r["torch_us"], b["us"], 100 * b["gbs"] / peak, sv, sv * r["crit_lo"], b["name"]))
            plan.setdefault("%dx%d" % (r["n"], r["k"]), {})[str(mm)] = dict(
                shape=r["shape"], family=b["family"], name=b["name"], cfg=b["cfg"], us=b["us"], torch_us=r["torch_us"])
        print("  TOTAL torch GEMM time/cycle = %.0f us ; saving = %.0f us (lower bound %.0f us after overlap weighting)"
              % (tot_t, tot, tot_lo))
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default="all", help="comma list of shape names or 'all' (see --list)")
    ap.add_argument("--ms", default="1,2,4,8")
    ap.add_argument("--impls", default="torch,triton_fma,triton_dot,cute,flashinfer",
                    help="subset of torch,torch_lt,triton_fma,triton_dot,cute,flashinfer,hcvllm")
    ap.add_argument("--budget", choices=["quick", "std", "full"], default="std")
    ap.add_argument("--pdl", choices=["0", "1", "both"], default="0",
                    help="Triton PDL variants (griddepcontrol); gains only show when the preceding kernel is PDL-aware")
    ap.add_argument("--cutile", action="store_true", help="also try flashinfer cutile backend (slow exhaustive autotune)")
    ap.add_argument("--peak-gbps", type=float, default=PEAK_GBPS_DEFAULT)
    ap.add_argument("--flush-factor", type=float, default=3.0)
    ap.add_argument("--max-gb", type=float, default=3.0)
    ap.add_argument("--tol", type=float, default=6e-3, help="max |err| / max|ref| allowed vs fp32 reference")
    ap.add_argument("--min-calls", type=int, default=96)
    ap.add_argument("--min-calls-coarse", type=int, default=32)
    ap.add_argument("--replays", type=int, default=15)
    ap.add_argument("--replays-coarse", type=int, default=5)
    ap.add_argument("--refine", type=int, default=5)
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--hot", action="store_true", help="also time L2-resident (R=1) for the top variants")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-calib", action="store_true")
    ap.add_argument("--no-show-torch-kernels", dest="show_torch_kernels", action="store_false")
    ap.add_argument("--json", default=None)
    ap.add_argument("--plan-json", default=None, help="write the best-variant plan consumed by skinny_patch.py")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest-cpu", action="store_true")
    args = ap.parse_args()
    args.impls = [s.strip() for s in args.impls.split(",") if s.strip()]

    if args.list:
        for s in SHAPES:
            print("%-10s N=%-6d K=%-5d calls/cycle=%-4g crit_lo=%.2f  %s" % (s.name, s.n, s.k, s.calls, s.crit_lo, s.layers))
        return
    if args.selftest_cpu:
        sys.exit(0 if selftest_cpu() else 1)

    assert torch.cuda.is_available(), "needs a GPU (use --selftest-cpu for the interpreter check)"
    dev = torch.device("cuda", args.device)
    torch.cuda.set_device(dev)
    name, sm, l2 = device_info(dev)
    print("GPU %s  SMs=%d  L2=%.0f MB  torch %s  triton %s  peak=%.0f GB/s" % (
        name, sm, l2 / 1e6, torch.__version__, triton.__version__, args.peak_gbps))
    print("bf16 reduced-precision-reduction allowed: %s ; preferred blas: %s" % (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, torch.backends.cuda.preferred_blas_library()))
    if not args.no_calib:
        bw = stream_read_bw(dev, sm)
        print("calibration: Triton streaming-read kernel reaches %.0f GB/s = %.1f%% of nominal peak "
              "(this is the practical ceiling for a skinny GEMM on this card)" % (bw, 100 * bw / args.peak_gbps))
    ms = [int(v) for v in args.ms.split(",")]
    names = [s.name for s in SHAPES] if args.shapes == "all" else args.shapes.split(",")
    notes, results = set(), []
    t0 = time.time()
    for nm in names:
        run_shape(SHAPE_BY_NAME[nm], ms, args, dev, sm, l2, notes, results)
        print("   [elapsed %.0f s]" % (time.time() - t0))
    plan = summarize(results, args)
    if notes:
        print("\nNOTES:")
        for t in sorted(notes):
            print("  -", t)
    if args.json:
        json.dump(dict(gpu=name, sm=sm, l2=l2, peak=args.peak_gbps, results=results, plan=plan),
                  open(args.json, "w"), indent=1, default=str)
        print("wrote", args.json)
    if args.plan_json:
        json.dump(plan, open(args.plan_json, "w"), indent=1, default=str)
        print("wrote", args.plan_json)


# =====================================================================================
# CPU self-test (Triton interpreter) - validates kernel logic only, says nothing about speed
# =====================================================================================
def selftest_cpu():
    """Validates kernel logic with the Triton interpreter.  NOTE: the interpreter mis-handles bf16 operands of
    tl.dot (raw-bit reinterpretation, errors ~1e11), so the tensor-core kernel is checked in fp16/fp32 here;
    the kernel body is dtype-agnostic.  The FMA kernel is checked in bf16 and fp16.  The interpreter's
    fp32->bf16 cast truncates, hence the looser bf16 tolerance (GPU conversion is round-to-nearest-even)."""
    torch.manual_seed(0)
    cases = [(40, 1024), (96, 640), (33, 320), (24, 2560)]
    fma = [TConf("fma", 4, 128, 1, 1), TConf("fma", 2, 256, 2, 2), TConf("fma", 1, 128, 3, 1),
           TConf("fma", 8, 256, 4, 4), TConf("fma", 2, 512, 1, 2)]
    dot = [TConf("dot", 16, 64, 1, 2, 3), TConf("dot", 32, 128, 2, 4, 3), TConf("dot", 16, 64, 5, 4, 2)]
    bad = 0
    total = 0
    for (n, k) in cases:
        for m in (1, 3, 4, 8):
            for dt, confs, tol in ((torch.bfloat16, fma, 8e-3), (torch.float16, fma + dot, 1e-3)):
                w = (torch.randn(n, k) * k ** -0.5).to(dt)
                x = torch.randn(m, k).to(dt)
                ref = x.float() @ w.float().t()
                for c in confs:
                    out = triton_skinny_gemm(x, w, c)
                    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
                    total += 1
                    if not (err < tol):
                        bad += 1
                        print("FAIL n=%d k=%d m=%d %s %s rel_err=%.3e" % (n, k, m, dt, c.label(), err))
    print("selftest-cpu: %d cases, %d failures" % (total, bad))
    return bad == 0


if __name__ == "__main__":
    main()
