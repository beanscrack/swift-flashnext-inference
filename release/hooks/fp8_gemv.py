"""fp8_gemv.py - weight-only FP8 (e4m3) skinny GEMM for the Swift/Qwen4Exp decode path (W8A16).

out[M, N] = x[M, K] (bf16) @ (w8[N, K] * scale)^T, fp32 accumulation, bf16 output.  Activations are never quantized.

Scales: fp32, shape [N, K // GROUP] (GROUP = 0 means one scale per output row, shape [N, 1]).
Kernels (same structure as bench_skinny.py's BF16 kernels, so the two can be compared like for like):
  fma  CUDA-core FMA, one CTA = BLOCK_N rows x one K slice, 16-byte weight loads along K (16 fp8 values), M <= 8
  dot  tensor-core tl.dot on bf16 after an in-register fp8 -> bf16 convert, M padded to BLOCK_M (16 or 32)
Both support split-K over CTAs with a 2-stage (fp32 workspace + reduce) scheme: no atomics, deterministic.
CUDA-graph safe: no host sync, workspaces come from torch.empty.
"""
import dataclasses

import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def quantize_fp8(w, group=0):
    """bf16/fp32 [N, K] -> (w8 [N, K] e4m3, scale [N, K // group] fp32); group=0: per output row."""
    N, K = w.shape
    wf = w.float()
    if group == 0:
        amax = wf.abs().amax(dim=1, keepdim=True)
    else:
        assert K % group == 0
        amax = wf.view(N, K // group, group).abs().amax(dim=2)
    scale = (amax / FP8_MAX).clamp_min(1e-12)
    if group == 0:
        w8 = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    else:
        w8 = (wf.view(N, K // group, group) / scale[:, :, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8).view(N, K)
    return w8.contiguous(), scale.contiguous()


def dequantize_fp8(w8, scale, group=0, dtype=torch.bfloat16):
    N, K = w8.shape
    if group == 0:
        return (w8.float() * scale).to(dtype)
    return (w8.float().view(N, K // group, group) * scale[:, :, None]).view(N, K).to(dtype)


@triton.jit
def _acc_row(acc, w, x_ptr, k_off, row_off, k_mask, EVEN_K: tl.constexpr):
    if EVEN_K:
        xr = tl.load(x_ptr + row_off + k_off)
    else:
        xr = tl.load(x_ptr + row_off + k_off, mask=k_mask, other=0.0)
    return acc + w * xr.to(tl.float32)[None, :]


@triton.jit
def _store_row(acc, s_row, row: tl.constexpr, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om,
               DIRECT: tl.constexpr, ROWSCALE: tl.constexpr):
    o = tl.sum(acc, axis=1)
    if ROWSCALE:
        o = o * s_row
    if DIRECT:
        tl.store(out_ptr + row * stride_om + offs_n, o.to(out_ptr.dtype.element_ty), mask=n_mask)
    else:
        tl.store(ws_ptr + (pid_k * M + row) * N + offs_n, o, mask=n_mask)


@triton.jit
def _w8_fma_kernel(x_ptr, w_ptr, s_ptr, out_ptr, ws_ptr, M, N, K, k_per_split,
                   stride_xm, stride_wn, stride_sn, stride_om,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
                   EVEN_K: tl.constexpr, EVEN_N: tl.constexpr, DIRECT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_begin = pid_k * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, K)
    w_row = w_ptr + offs_n[:, None] * stride_wn
    ROWSCALE: tl.constexpr = GROUP == 0
    if ROWSCALE:
        s_row = tl.load(s_ptr + offs_n * stride_sn, mask=n_mask, other=0.0)
    else:
        s_row = tl.zeros([BLOCK_N], dtype=tl.float32)

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
                w = tl.load(w_row + k_off[None, :], mask=n_mask[:, None], other=0.0, eviction_policy="evict_first")
        else:
            w = tl.load(w_row + k_off[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                        eviction_policy="evict_first")
        w = w.to(tl.float32)
        if not ROWSCALE:   # BLOCK_K divides GROUP: one scale per row for this whole tile
            sg = tl.load(s_ptr + offs_n * stride_sn + (k_begin + k0) // GROUP, mask=n_mask, other=0.0)
            w = w * sg[:, None]
        acc0 = _acc_row(acc0, w, x_ptr, k_off, 0, k_mask, EVEN_K)
        if BLOCK_M > 1:
            acc1 = _acc_row(acc1, w, x_ptr, k_off, stride_xm, k_mask, EVEN_K)
        if BLOCK_M > 2:
            acc2 = _acc_row(acc2, w, x_ptr, k_off, 2 * stride_xm, k_mask, EVEN_K)
        if BLOCK_M > 3:
            acc3 = _acc_row(acc3, w, x_ptr, k_off, 3 * stride_xm, k_mask, EVEN_K)
        if BLOCK_M > 4:
            acc4 = _acc_row(acc4, w, x_ptr, k_off, 4 * stride_xm, k_mask, EVEN_K)
        if BLOCK_M > 5:
            acc5 = _acc_row(acc5, w, x_ptr, k_off, 5 * stride_xm, k_mask, EVEN_K)
        if BLOCK_M > 6:
            acc6 = _acc_row(acc6, w, x_ptr, k_off, 6 * stride_xm, k_mask, EVEN_K)
        if BLOCK_M > 7:
            acc7 = _acc_row(acc7, w, x_ptr, k_off, 7 * stride_xm, k_mask, EVEN_K)

    _store_row(acc0, s_row, 0, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT, ROWSCALE)
    if BLOCK_M > 1:
        _store_row(acc1, s_row, 1, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT, ROWSCALE)
    if BLOCK_M > 2:
        _store_row(acc2, s_row, 2, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT, ROWSCALE)
    if BLOCK_M > 3:
        _store_row(acc3, s_row, 3, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT, ROWSCALE)
    if BLOCK_M > 4:
        _store_row(acc4, s_row, 4, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT, ROWSCALE)
    if BLOCK_M > 5:
        _store_row(acc5, s_row, 5, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT, ROWSCALE)
    if BLOCK_M > 6:
        _store_row(acc6, s_row, 6, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT, ROWSCALE)
    if BLOCK_M > 7:
        _store_row(acc7, s_row, 7, M, N, offs_n, n_mask, pid_k, out_ptr, ws_ptr, stride_om, DIRECT, ROWSCALE)


@triton.jit
def _w8_dot_kernel(x_ptr, w_ptr, s_ptr, out_ptr, ws_ptr, M, N, K, k_per_split,
                   stride_xm, stride_wn, stride_sn, stride_om,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr,
                   EVEN_K: tl.constexpr, EVEN_N: tl.constexpr, DIRECT: tl.constexpr):
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
        w = w.to(tl.bfloat16)
        if GROUP == 0:
            acc = tl.dot(x, w, acc)
        else:
            sg = tl.load(s_ptr + offs_n * stride_sn + (k_begin + k0) // GROUP, mask=n_mask, other=0.0)
            acc += tl.dot(x, w) * sg[None, :]
    if GROUP == 0:
        acc = acc * tl.load(s_ptr + offs_n * stride_sn, mask=n_mask, other=0.0)[None, :]
    mn_mask = m_mask[:, None] & n_mask[None, :]
    if DIRECT:
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], acc.to(out_ptr.dtype.element_ty),
                 mask=mn_mask)
    else:
        tl.store(ws_ptr + (pid_k * M + offs_m[:, None]) * N + offs_n[None, :], acc, mask=mn_mask)


@triton.jit
def _reduce_kernel(ws_ptr, out_ptr, M, N, stride_om, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = M * N
    mask = offs < total
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(ws_ptr + s * total + offs, mask=mask, other=0.0)
    m = offs // N
    n = offs - m * N
    tl.store(out_ptr + m * stride_om + n, acc.to(out_ptr.dtype.element_ty), mask=mask)


@dataclasses.dataclass(frozen=True)
class W8Conf:
    kind: str                 # "fma" | "dot"
    block_n: int
    block_k: int
    split_k: int = 1
    num_warps: int = 4
    num_stages: int = 3
    block_m: int = 16         # dot only (16 or 32)

    def label(self):
        s = "w8%s bn=%d bk=%d sk=%d w=%d" % (self.kind, self.block_n, self.block_k, self.split_k, self.num_warps)
        if self.kind == "dot":
            s += " st=%d bm=%d" % (self.num_stages, self.block_m)
        return s


def _cdiv(a, b):
    return -(-a // b)


def split_geometry(K, block_k, split_k):
    kb = _cdiv(K, block_k)
    sk = max(1, min(split_k, kb))
    kps = _cdiv(kb, sk) * block_k
    seff = _cdiv(K, kps)
    return kps, seff, (K % kps == 0)


def w8_gemm(x, w8, scale, group, cfg, out=None):
    """x [M, K] bf16 (inner stride 1, row stride a multiple of 8), w8 [N, K] e4m3 contiguous, scale fp32."""
    M, K = x.shape
    N = w8.shape[0]
    assert w8.shape[1] == K and x.stride(1) == 1 and w8.stride(1) == 1
    if group:
        assert group % cfg.block_k == 0, "BLOCK_K must divide GROUP"
    if cfg.kind == "fma":
        assert 1 <= M <= 8
    else:
        assert 1 <= M <= cfg.block_m
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    kps, seff, even_k = split_geometry(K, cfg.block_k, cfg.split_k)
    if group:
        assert kps % group == 0 or seff == 1, "split boundary must align with scale groups"
    even_n = (N % cfg.block_n == 0)
    direct = (seff == 1)
    ws = out if direct else torch.empty((seff, M, N), dtype=torch.float32, device=x.device)
    grid = (_cdiv(N, cfg.block_n), seff)
    if cfg.kind == "fma":
        _w8_fma_kernel[grid](x, w8, scale, out, ws, M, N, K, kps, x.stride(0), w8.stride(0), scale.stride(0),
                             out.stride(0), BLOCK_M=M, BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k, GROUP=group,
                             EVEN_K=even_k, EVEN_N=even_n, DIRECT=direct, num_warps=cfg.num_warps)
    else:
        _w8_dot_kernel[grid](x, w8, scale, out, ws, M, N, K, kps, x.stride(0), w8.stride(0), scale.stride(0),
                             out.stride(0), BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k,
                             GROUP=group, EVEN_K=even_k, EVEN_N=even_n, DIRECT=direct,
                             num_warps=cfg.num_warps, num_stages=cfg.num_stages)
    if not direct:
        blk = 1024 if M * N >= 4096 else 256
        _reduce_kernel[(_cdiv(M * N, blk),)](ws, out, M, N, out.stride(0), SPLIT=seff, BLOCK=blk, num_warps=4)
    return out


def gen_configs(n, k, m, group, sm):
    """Candidate configs for one (N, K, M); pruned to keep >= ~1 wave of CTAs and BLOCK_K | GROUP."""
    out = []
    bks = [b for b in (128, 256, 512) if (group == 0 or group % b == 0) and b <= k]
    if m <= 8:
        for bn in (2, 4, 8, 16, 32):
            for bk in bks:
                for w in (2, 4, 8):
                    if bn * bk < 32 * w * 16:     # every thread needs >= one 16-byte vector
                        continue
                    if bn * bk // (32 * w) > 64:  # register budget per row-accumulator
                        continue
                    for sk in (1, 2, 4, 8):
                        ctas = _cdiv(n, bn) * min(sk, _cdiv(k, bk))
                        if ctas < sm // 2 or (sk > 1 and _cdiv(n, bn) >= 4 * sm):
                            continue
                        if group and sk > 1 and (_cdiv(_cdiv(k, bk), sk) * bk) % group:
                            continue
                        out.append(W8Conf("fma", bn, bk, sk, w))
    bm = 16 if m <= 16 else 32
    for bn in (16, 32, 64, 128):
        for bk in [b for b in (64, 128, 256) if group == 0 or group % b == 0]:
            for w in (4, 8):
                for st in (3, 4, 6):
                    for sk in (1, 2, 4):
                        ctas = _cdiv(n, bn) * min(sk, _cdiv(k, bk))
                        if ctas < sm // 2 or (sk > 1 and _cdiv(n, bn) >= 4 * sm):
                            continue
                        if group and sk > 1 and (_cdiv(_cdiv(k, bk), sk) * bk) % group:
                            continue
                        out.append(W8Conf("dot", bn, bk, sk, w, st, bm))
    return out
