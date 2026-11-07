#!/usr/bin/env python3
"""fp8_bench.py - phase-1 gate for weight-only FP8 dense layers on Swift (see docs/FP8_SCOPE.md).

For each real weight shape at decode M values, time (L2 flushed: weights cycled through enough copies to exceed
--flush x L2) and check error (real checkpoint weights, random bf16 activations, fp32 reference):
  torch    F.linear bf16 (cuBLAS)
  plan     the deployed BF16 skinny kernel from release/plan/skinny_plan_wide.json (CuTe or Triton)
  w8row    FP8 e4m3 weight-only, per-output-row scale   (best config of a sweep)
  w8g128   FP8 e4m3 weight-only, per-row per-128-K-group scale (best config of a sweep)
Usage: python fp8_bench.py --shapes qkvz,o_proj,qsa_qkv,lm_head --ms 1,4,5 --json out.json
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

HOOKS = os.environ.get("SWIFT_HOOKS", "/u701/j692wu/swift-opt/release/hooks")
sys.path.insert(0, HOOKS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_skinny as B  # noqa: E402
import fp8_gemv as Q      # noqa: E402

CKPT = open("/u701/j692wu/swift-local/model-path.txt").read().strip()
L = "model.language_model.layers."
REAL = {   # shape name -> list of tensors concatenated along N (vLLM merges these)
    "qkvz":       [L + "0.linear_attn.in_proj_qkv.weight", L + "0.linear_attn.in_proj_z.weight"],
    "o_proj":     [L + "0.linear_attn.out_proj.weight"],
    "qsa_o":      [L + "3.self_attn.o_proj.weight"],
    "qsa_qkv":    [L + "3.self_attn.q_proj.weight", L + "3.self_attn.k_proj.weight", L + "3.self_attn.v_proj.weight"],
    "sh_gate_up": [L + "0.mlp.shared_expert.gate_proj.weight", L + "0.mlp.shared_expert.up_proj.weight"],
    "sh_down":    [L + "0.mlp.shared_expert.down_proj.weight"],
    "lm_head":    ["lm_head.weight"],
}


def load_real(name):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    parts = []
    for t in REAL[name]:
        with safe_open(os.path.join(CKPT, idx[t]), framework="pt", device="cuda") as f:
            parts.append(f.get_tensor(t).to(torch.bfloat16))
    return torch.cat(parts, 0).contiguous()


def copies(t, flush_bytes, max_bytes):
    nb = t.numel() * t.element_size()
    r = max(2, math.ceil(flush_bytes / nb))
    r = max(1, min(r, max_bytes // nb, 256))
    return [t.clone() for _ in range(r)]


def err_stats(out, ref32):
    d = (out.float() - ref32)
    scale = ref32.abs().max().clamp_min(1e-6)
    return dict(max_rel=(d.abs().max() / scale).item(),
                rms_rel=(d.pow(2).mean().sqrt() / ref32.pow(2).mean().sqrt().clamp_min(1e-12)).item())


def plan_fn(plan, n, k, m):
    e = plan.get("%dx%d" % (n, k), {}).get(str(m))
    if e is None:
        return None, None
    if e["family"] == "cute":
        from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import SkinnyGemmConfig, shape_dynamic_skinny_gemm
        cfg = SkinnyGemmConfig(**e["cfg"])
        shape_dynamic_skinny_gemm.request_warmup_configs(torch.bfloat16, {cfg})
        return (lambda x, w: shape_dynamic_skinny_gemm(x, w, cfg)), e["name"]
    if e["family"] in ("triton_fma", "triton_dot"):
        cfg = B.TConf(**e["cfg"])
        return (lambda x, w: B.triton_skinny_gemm(x, w, cfg)), e["name"]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="qkvz,o_proj,qsa_qkv,lm_head")
    ap.add_argument("--ms", default="1,4,5")
    ap.add_argument("--groups", default="0,128")
    ap.add_argument("--flush", type=float, default=6.0, help="weights cycled per timing >= flush x L2")
    ap.add_argument("--max-gb", type=float, default=24.0)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--min-calls", type=int, default=64)
    ap.add_argument("--max-configs", type=int, default=400)
    ap.add_argument("--plan", default="/u701/j692wu/swift-opt/release/plan/skinny_plan_wide.json")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    name, sm, l2 = B.device_info(dev)
    print("device %s  SMs %d  L2 %.0f MiB" % (name, sm, l2 / 2**20), flush=True)
    plan = json.load(open(a.plan))
    flush_bytes = int(a.flush * l2)
    max_bytes = int(a.max_gb * 1e9)
    results = []
    for sname in a.shapes.split(","):
        t0 = time.time()
        w = load_real(sname)
        n, k = w.shape
        print("\n=== %s  [%d x %d]  bf16 %.1f MB  (loaded in %.1fs)" % (sname, n, k, n * k * 2 / 1e6, time.time() - t0),
              flush=True)
        q = {g: Q.quantize_fp8(w, g) for g in map(int, a.groups.split(","))}
        wc = copies(w, flush_bytes, max_bytes)
        qc = {g: list(zip(copies(q[g][0], flush_bytes, max_bytes), [q[g][1]] * 256)) for g in q}
        for m in map(int, a.ms.split(",")):
            x = torch.randn((m, k), dtype=torch.bfloat16, device=dev)
            ref32 = x.float() @ w.float().t()
            row = dict(shape=sname, n=n, k=k, m=m)
            t_torch = B.time_calls(lambda xx, ww: F.linear(xx, ww), x, wc, a.min_calls, a.replays)[0]
            row["torch_us"] = t_torch
            row["torch_err"] = err_stats(F.linear(x, w), ref32)
            fn, lab = plan_fn(plan, n, k, m)
            if fn is not None:
                try:
                    row["plan_us"] = B.time_calls(fn, x, wc, a.min_calls, a.replays)[0]
                    row["plan_name"] = lab
                    row["plan_err"] = err_stats(fn(x, w), ref32)
                except Exception as e:
                    row["plan_us"] = None
                    print("  plan kernel failed: %s" % e)
            bf16_bytes = n * k * 2
            print("  M=%d torch %.1f us (%.0f GB/s)   plan %s us  [%s]" % (
                m, t_torch, bf16_bytes / t_torch / 1e3, "%.1f" % row["plan_us"] if row.get("plan_us") else "-",
                row.get("plan_name", "-")), flush=True)
            for g, (w8, s) in q.items():
                cands = Q.gen_configs(n, k, m, g, sm)[: a.max_configs]
                best = None
                nfail = 0
                for cfg in cands:
                    try:
                        f = (lambda xx, ws, cfg=cfg, g=g: Q.w8_gemm(xx, ws[0], ws[1], g, cfg))
                        out = f(x, (w8, s))
                        torch.cuda.synchronize()
                        e = err_stats(out, ref32)
                        if not math.isfinite(e["max_rel"]) or e["max_rel"] > 0.1:
                            nfail += 1
                            continue
                        us = B.time_calls(f, x, qc[g], a.min_calls, max(5, a.replays // 4))[0]
                        if best is None or us < best[0]:
                            best = (us, cfg, e)
                    except Exception:
                        nfail += 1
                        if nfail <= 2:
                            print("    config %s failed:\n%s" % (cfg.label(), traceback.format_exc()[-600:]))
                if best is None:
                    print("  w8 g=%d: no valid config (%d failed)" % (g, nfail))
                    continue
                f = (lambda xx, ws, cfg=best[1], g=g: Q.w8_gemm(xx, ws[0], ws[1], g, cfg))
                us = B.time_calls(f, x, qc[g], a.min_calls, a.replays)[0]   # re-time the winner properly
                key = "w8row" if g == 0 else "w8g%d" % g
                row[key + "_us"] = us
                row[key + "_cfg"] = best[1].label()
                row[key + "_err"] = best[2]
                base = row.get("plan_us") or t_torch
                print("  M=%d %-7s %.1f us (%.0f GB/s fp8)  x%.2f vs %s  err max %.2e rms %.2e (bf16 torch: max %.2e rms %.2e)"
                      "  [%s; %d cfgs, %d failed]" % (
                          m, key, us, n * k / us / 1e3, base / us, "plan" if row.get("plan_us") else "torch",
                          best[2]["max_rel"], best[2]["rms_rel"], row["torch_err"]["max_rel"], row["torch_err"]["rms_rel"],
                          best[1].label(), len(cands), nfail), flush=True)
            results.append(row)
        del wc, qc, q, w
        torch.cuda.empty_cache()
        if a.json:
            json.dump(results, open(a.json, "w"), indent=1)
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
