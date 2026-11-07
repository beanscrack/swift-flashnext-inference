"""CPU replay of captured vLLM MTP calls through mtp_torch (copy in this dir).

Same trimming / KV filtering as parity_v2.py. Reports per-row relative error
||torch - vllm|| / ||vllm|| of sample_hidden, bucketed by absolute position.
Options: --window W (restrict history to last W positions; experiment only),
         --pad_write 1 (emulate vLLM writing K/V of trailing rows into their position slots).
"""
import argparse, collections, glob, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_torch as M

ap = argparse.ArgumentParser()
ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
ap.add_argument("--calls", type=int, default=150)
ap.add_argument("--config", required=True)
ap.add_argument("--weights", default="/u701/j692wu/swift-opt/mtp/view-identity/model-tail.safetensors")
ap.add_argument("--captures", default="/u701/j692wu/swift-opt/runs/p1-parity-v2/captures")
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--attn", default="fp32", choices=["fp32", "pbf16", "vllm"])
ap.add_argument("--rope", default="bf16")
ap.add_argument("--router", default="bf16")
ap.add_argument("--window", type=int, default=0)
ap.add_argument("--pad_write", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_num_threads(a.threads)
dt = torch.float32 if a.dtype == "fp32" else torch.bfloat16
M.ATTN_MODE = a.attn
M.ROPE_MODE = a.rope
M.ROUTER_MODE = a.router
model = M.MTPDraft.from_safetensors(a.weights, a.config, dtype=dt, device="cpu").eval()
files = sorted(glob.glob(os.path.join(a.captures, "call_*.pt")))[: a.calls]


def filt(kv, keep):
    return M.KVState(kv.k[:, keep], kv.v[:, keep], kv.pos[:, keep])


kv = None; rel = []; cos = []; info = []
t0 = time.time()
with torch.no_grad():
    for i, fn in enumerate(files):
        c = torch.load(fn, map_location="cpu")
        pf = c["positions"]; pf = pf[0] if pf.dim() == 2 else pf
        n = 1
        while n < pf.shape[0] and int(pf[n]) == int(pf[0]) + n: n += 1
        start = int(pf[0])
        if kv is not None:
            kv = filt(kv, kv.pos[0] < start)
            if a.window > 0:
                kv = filt(kv, kv.pos[0] >= start - a.window)
        if a.pad_write and n < pf.shape[0] and kv is not None:
            padpos = set(int(x) for x in pf[n:])
            kv = filt(kv, torch.tensor([int(x) not in padpos for x in kv.pos[0]]))
        if a.pad_write and n < pf.shape[0]:
            sh_all, _, kv = model(c["hidden_input" if False else "hidden_in"].to(dt)[None], c["input_ids"][None], pf[None], kv)
            sh = sh_all[0, :n].float()
        else:
            sh, _, kv = model(c["hidden_in"][:n].to(dt)[None], c["input_ids"][:n][None], pf[:n][None], kv)
            sh = sh[0].float()
        ref = c["sample_hidden"][:n].float()
        for r in range(n):
            rel.append(float((sh[r] - ref[r]).norm() / ref[r].norm()))
            cos.append(float(torch.nn.functional.cosine_similarity(sh[r], ref[r], dim=0)))
            info.append(int(pf[r]))
rel_t = torch.tensor(rel)
b = collections.defaultdict(list)
for e, p in zip(rel, info): b[p // 64].append(e)
res = {"dtype": a.dtype, "attn": a.attn, "rope": a.rope, "window": a.window, "pad_write": a.pad_write,
       "rows": len(rel), "calls": len(files),
       "relerr_median_all": float(rel_t.median()), "cos_median": float(torch.tensor(cos).median()),
       "cos_min": min(cos), "secs": round(time.time() - t0, 1),
       "relerr_median_by_pos64": {"%d" % (k * 64): round(float(torch.tensor(v).median()), 4) for k, v in sorted(b.items())},
       "relerr_mean_by_pos64": {"%d" % (k * 64): round(float(torch.tensor(v).mean()), 4) for k, v in sorted(b.items())}}
print(json.dumps(res))
if a.out:
    json.dump({"summary": res, "rel": rel, "pos": info}, open(a.out, "w"))
