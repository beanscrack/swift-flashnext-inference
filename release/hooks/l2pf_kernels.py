# SPDX-License-Identifier: Apache-2.0
"""Side-stream L2 weight-prefetch kernels for SM120 (RTX PRO 6000 Blackwell), Triton implementation.

A prefetch launch takes up to 4 byte ranges ("segments", e.g. o_proj.weight, HC weights, router weight) and pulls
them from DRAM into L2 so that a later GEMM hits in L2.  The kernel never writes memory (except a 4-byte sink that is
only written when write_sink != 0, which callers never set), so it cannot change model numerics.

Modes (all verified to assemble for sm_120a with Triton's ptxas-blackwell 13.1, see ptx_probe.py / compile_check.py):
  "touch"     16-B vector loads that do not allocate in L1 (ld.global.nc.L1::no_allocate.L2::256B.v4, SASS
              LDG.E.NA.LTC256B.128.CONSTANT).  Self-throttling: the achieved DRAM rate is set by bytes in flight
              (CTAs x BLOCK x 16 B) / latency, and the kernel duration in Nsight is the real transfer time.  DEFAULT.
  "touch_el"  same with an L2::cache_hint evict_last policy (createpolicy.fractional.L2::evict_last).
  "pf"        prefetch.global.L2 (SASS CCTL.E.PF2): fire-and-forget, one instruction per STRIDE bytes per lane.
              No register / scoreboard, so the kernel finishes long before the data arrives; MUST be paced
              (pace_ns) or it can flood the memory queues in front of the latency-bound kernels it overlaps.
  "pf_el"     prefetch.global.L2::evict_last (SASS CCTL.E.PML2).
  "bulk"      cp.async.bulk.prefetch.L2.global (SASS UBLKPF.L2): one TMA-unit instruction per CHUNK bytes per warp;
              lowest SM issue cost; fire-and-forget like "pf" -> pace it.
  "bulk_el"   bulk with an L2::cache_hint evict_last policy (createpolicy.fractional.L2::evict_last).
  "demote"    applypriority.global.L2::evict_normal (SASS CCTL.E.DML2) over the ranges: undo evict_last marks after use.

Work split: every segment is split into nprog contiguous slices, so all CTAs finish segment 0 (the most urgent
weight) before any CTA starts segment 1.  Optional start delay (delay_ns) and pacing (pace_ns per step per CTA)
use %globaltimer (wall-clock ns, independent of the power-capped SM clock).
"""
import triton
import triton.language as tl

MODES = ("touch", "touch_el", "pf", "pf_el", "bulk", "bulk_el", "demote")


@triton.jit
def _wait_until(deadline):
    t = tl.extra.cuda.globaltimer()
    while t < deadline:
        t = tl.extra.cuda.globaltimer()


@triton.jit
def _touch_range(base, nbytes, pid, nprog, acc, t0, step, pace_ns,
                 BLOCK: tl.constexpr, UNROLL: tl.constexpr, EL: tl.constexpr):
    # base: int64 byte address (16-B aligned); nbytes: multiple of 16.  BLOCK = 16-byte vectors per wave; with
    # 4 warps each thread owns BLOCK/128 of them, and Triton emits all of a thread's loads of one wave back to back
    # (independent asm statements) before the XOR that consumes them -> BLOCK*16 bytes in flight per CTA.
    # The loads are inline PTX so the exact instruction is fixed (Triton's .cg lowers to LDG.E.STRONG.GPU and
    # .cg cannot be combined with evict_last):  LDG.E.NA.LTC256B.128.CONSTANT  (probe: ptx_probe.py).
    STEP: tl.constexpr = BLOCK * UNROLL * 16
    per = tl.cdiv(tl.cdiv(nbytes, nprog), STEP) * STEP
    lo = pid.to(tl.int64) * per
    hi = tl.minimum(lo + per, nbytes)
    for off in range(lo, hi, STEP):
        for u in tl.static_range(UNROLL):
            o = off + (u * BLOCK + tl.arange(0, BLOCK)).to(tl.int64) * 16
            o = tl.minimum(o, hi - 16)          # tail lanes re-read the last vector (no masks in asm)
            if EL:
                a, b, c, d = tl.inline_asm_elementwise(
                    "{\n\t.reg .b64 pol;\n\tcreatepolicy.fractional.L2::evict_last.b64 pol, 1.0;\n\t"
                    "ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {$0,$1,$2,$3}, [$4], pol;\n\t}",
                    "=r,=r,=r,=r,l", [base + o], dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
                    is_pure=False, pack=1)
            else:
                a, b, c, d = tl.inline_asm_elementwise(
                    "ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {$0,$1,$2,$3}, [$4];",
                    "=r,=r,=r,=r,l", [base + o], dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
                    is_pure=False, pack=1)
            acc = acc ^ a ^ b ^ c ^ d
        step += 1
        if pace_ns > 0:
            _wait_until(t0 + step * pace_ns)
    return acc, step


@triton.jit
def _pf_range(base, nbytes, pid, nprog, acc, t0, step, pace_ns,
              BLOCK: tl.constexpr, UNROLL: tl.constexpr, STRIDE: tl.constexpr, KIND: tl.constexpr):
    # KIND 0: prefetch.global.L2   1: prefetch.global.L2::evict_last   2: applypriority.global.L2::evict_normal
    STEP: tl.constexpr = BLOCK * UNROLL * STRIDE
    per = tl.cdiv(tl.cdiv(nbytes, nprog), STEP) * STEP
    lo = pid.to(tl.int64) * per
    hi = tl.minimum(lo + per, nbytes)
    for off in range(lo, hi, STEP):
        for u in tl.static_range(UNROLL):
            o = off + (u * BLOCK + tl.arange(0, BLOCK)).to(tl.int64) * STRIDE
            o = tl.minimum(o, hi - 1)          # tail lanes re-touch the last line (harmless, no masks in asm)
            a = base + o
            if KIND == 0:
                r = tl.inline_asm_elementwise("prefetch.global.L2 [$1];\n\tmov.u32 $0, 0;", "=r,l", [a],
                                              dtype=tl.int32, is_pure=False, pack=1)
            elif KIND == 1:
                r = tl.inline_asm_elementwise("prefetch.global.L2::evict_last [$1];\n\tmov.u32 $0, 0;", "=r,l", [a],
                                              dtype=tl.int32, is_pure=False, pack=1)
            else:
                r = tl.inline_asm_elementwise("applypriority.global.L2::evict_normal [$1], 128;\n\tmov.u32 $0, 0;",
                                              "=r,l", [a], dtype=tl.int32, is_pure=False, pack=1)
            acc = acc ^ r
        step += 1
        if pace_ns > 0:
            _wait_until(t0 + step * pace_ns)
    return acc, step


@triton.jit
def _bulk_range(base, nbytes, pid, nprog, acc, t0, step, pace_ns, CHUNK: tl.constexpr, EVICT_LAST: tl.constexpr):
    # Scalar (warp-uniform) code: with num_warps=1 the UBLKPF executes once per CTA per chunk.
    per = tl.cdiv(tl.cdiv(nbytes, nprog), CHUNK) * CHUNK
    lo = pid.to(tl.int64) * per
    hi = tl.minimum(lo + per, nbytes)
    for off in range(lo, hi, CHUNK):
        sz = tl.minimum(hi - off, CHUNK).to(tl.int32)     # multiple of 16 because nbytes and CHUNK are
        if EVICT_LAST:
            r = tl.inline_asm_elementwise(
                "{\n\t.reg .b64 pol;\n\tcreatepolicy.fractional.L2::evict_last.b64 pol, 1.0;\n\t"
                "cp.async.bulk.prefetch.L2.global.L2::cache_hint [$1], $2, pol;\n\tmov.u32 $0, 0;\n\t}",
                "=r,l,r", [base + off, sz], dtype=tl.int32, is_pure=False, pack=1)
        else:
            r = tl.inline_asm_elementwise("cp.async.bulk.prefetch.L2.global [$1], $2;\n\tmov.u32 $0, 0;", "=r,l,r",
                                          [base + off, sz], dtype=tl.int32, is_pure=False, pack=1)
        acc = acc ^ r
        step += 1
        if pace_ns > 0:
            _wait_until(t0 + step * pace_ns)
    return acc, step


# No value specialization on the integer arguments: one compiled variant per constexpr combination, so a launch
# inside CUDA-graph capture can never trigger a fresh compile because some size/pace value has a different
# divisibility than in the eager warm-up.
@triton.jit(do_not_specialize=["b0", "n0", "b1", "n1", "b2", "n2", "b3", "n3", "write_sink", "delay_ns", "pace_ns"])
def l2pf_kernel(b0, n0, b1, n1, b2, n2, b3, n3, sink_ptr, write_sink, delay_ns, pace_ns,
                NSEG: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr, UNROLL: tl.constexpr,
                STRIDE: tl.constexpr, CHUNK: tl.constexpr):
    """MODE: 0 touch, 1 touch_el, 2 pf, 3 pf_el, 4 bulk, 5 bulk_el, 6 demote.  b*: int64 byte addresses, n*: bytes."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    t0 = tl.extra.cuda.globaltimer()
    if delay_ns > 0:
        _wait_until(t0 + delay_ns)
        t0 = t0 + delay_ns
    step = pid * 0
    if MODE <= 1:
        acc = tl.zeros([BLOCK], dtype=tl.int32)
        TEL: tl.constexpr = MODE == 1
        acc, step = _touch_range(b0, n0, pid, nprog, acc, t0, step, pace_ns, BLOCK, UNROLL, TEL)
        if NSEG > 1:
            acc, step = _touch_range(b1, n1, pid, nprog, acc, t0, step, pace_ns, BLOCK, UNROLL, TEL)
        if NSEG > 2:
            acc, step = _touch_range(b2, n2, pid, nprog, acc, t0, step, pace_ns, BLOCK, UNROLL, TEL)
        if NSEG > 3:
            acc, step = _touch_range(b3, n3, pid, nprog, acc, t0, step, pace_ns, BLOCK, UNROLL, TEL)
        tot = tl.xor_sum(acc, axis=0)
    elif MODE <= 3 or MODE == 6:
        KIND: tl.constexpr = 0 if MODE == 2 else (1 if MODE == 3 else 2)
        acc = tl.zeros([BLOCK], dtype=tl.int32)
        acc, step = _pf_range(b0, n0, pid, nprog, acc, t0, step, pace_ns, BLOCK, UNROLL, STRIDE, KIND)
        if NSEG > 1:
            acc, step = _pf_range(b1, n1, pid, nprog, acc, t0, step, pace_ns, BLOCK, UNROLL, STRIDE, KIND)
        if NSEG > 2:
            acc, step = _pf_range(b2, n2, pid, nprog, acc, t0, step, pace_ns, BLOCK, UNROLL, STRIDE, KIND)
        if NSEG > 3:
            acc, step = _pf_range(b3, n3, pid, nprog, acc, t0, step, pace_ns, BLOCK, UNROLL, STRIDE, KIND)
        tot = tl.xor_sum(acc, axis=0)
    else:
        EL: tl.constexpr = MODE == 5
        acc = pid * 0
        acc, step = _bulk_range(b0, n0, pid, nprog, acc, t0, step, pace_ns, CHUNK, EL)
        if NSEG > 1:
            acc, step = _bulk_range(b1, n1, pid, nprog, acc, t0, step, pace_ns, CHUNK, EL)
        if NSEG > 2:
            acc, step = _bulk_range(b2, n2, pid, nprog, acc, t0, step, pace_ns, CHUNK, EL)
        if NSEG > 3:
            acc, step = _bulk_range(b3, n3, pid, nprog, acc, t0, step, pace_ns, CHUNK, EL)
        tot = acc
    if write_sink != 0:          # never true at runtime; keeps the loads observable for the compiler
        tl.store(sink_ptr + pid, tot + step)


_MODE_ID = {m: i for i, m in enumerate(MODES)}


def kernel_config(mode, unroll=None, block=None, stride=None, chunk=None, num_warps=None):
    """Default meta-parameters per mode (all overridable)."""
    m = _MODE_ID[mode]
    if m <= 1:          # touch: BLOCK 16-B vectors per wave (8 per thread at 4 warps) = 16 KiB in flight per CTA
        d = dict(BLOCK=block or 1024, UNROLL=unroll or 1, STRIDE=stride or 128, CHUNK=chunk or 16384,
                 num_warps=num_warps or 4)
    elif m <= 3 or m == 6:
        d = dict(BLOCK=block or 128, UNROLL=unroll or 8, STRIDE=stride or 128, CHUNK=chunk or 16384,
                 num_warps=num_warps or 4)
    else:
        d = dict(BLOCK=block or 128, UNROLL=1, STRIDE=stride or 128, CHUNK=chunk or 16384, num_warps=1)
    return d


def bytes_per_step(mode, cfg):
    m = _MODE_ID[mode]
    if m <= 1:
        return cfg["BLOCK"] * cfg["UNROLL"] * 4
    if m <= 3 or m == 6:
        return cfg["BLOCK"] * cfg["UNROLL"] * cfg["STRIDE"]
    return cfg["CHUNK"]


def pace_for_rate(rate_tbps, ctas, mode, cfg):
    """ns per step per CTA that caps the aggregate issue rate at rate_tbps (TB/s == bytes/ns / 1e3)."""
    if not rate_tbps:
        return 0
    per_cta_bytes_per_ns = rate_tbps * 1e3 / ctas
    return max(1, int(bytes_per_step(mode, cfg) / per_cta_bytes_per_ns))


def segments_from_tensors(tensors, budget_bytes=None):
    """[(addr, nbytes)] for contiguous tensors, truncated to budget (16-B multiples, priority = list order)."""
    segs = []
    left = budget_bytes if budget_bytes is not None else 1 << 62
    for t in tensors:
        if t is None or left <= 0:
            continue
        if not t.is_contiguous() or not t.is_cuda:
            continue
        nb = t.numel() * t.element_size()
        if t.data_ptr() % 16:
            continue
        nb = min(nb, left) // 16 * 16
        if nb <= 0:
            continue
        segs.append((t.data_ptr(), nb))
        left -= nb
    return segs


def launch(segs, sink, *, mode="touch", ctas=32, delay_ns=0, pace_ns=0, rate_tbps=None, unroll=None, block=None,
           stride=None, chunk=None, num_warps=None):
    """Enqueue prefetch kernel(s) on the CURRENT stream for [(addr, nbytes)] (CUDA-graph capturable: no host sync,
    no allocation).  `sink` is a persistent int32 CUDA tensor with >= ctas elements (never written)."""
    cfg = kernel_config(mode, unroll, block, stride, chunk, num_warps)
    if rate_tbps and not pace_ns:
        pace_ns = pace_for_rate(rate_tbps, ctas, mode, cfg)
    for i in range(0, len(segs), 4):
        part = list(segs[i:i + 4])
        nseg = len(part)
        while len(part) < 4:
            part.append((part[0][0], 0))
        args = []
        for a, n in part:
            args += [a, n]
        l2pf_kernel[(ctas,)](*args, sink, 0, int(delay_ns if i == 0 else 0), int(pace_ns),
                             NSEG=nseg, MODE=_MODE_ID[mode], BLOCK=cfg["BLOCK"], UNROLL=cfg["UNROLL"],
                             STRIDE=cfg["STRIDE"], CHUNK=cfg["CHUNK"], num_warps=cfg["num_warps"], num_stages=1)
    return pace_ns
