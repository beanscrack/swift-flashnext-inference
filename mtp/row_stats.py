"""Per-row cosine / relative-error distribution of mtp_torch vs captured vLLM sample_hidden (same replay as parity_v2)."""
import glob, os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_torch as M
cap, cfg, w = sys.argv[1], sys.argv[2], sys.argv[3]
dev = torch.device("cuda")
model = M.MTPDraft.from_safetensors(w, cfg, dtype=torch.bfloat16, device=dev).eval()
kv = None; cs = []; rel = []; info = []
with torch.no_grad():
    for i, fn in enumerate(sorted(glob.glob(os.path.join(cap, "call_*.pt")))):
        c = torch.load(fn, map_location="cpu"); pos = c["positions"]; pos = pos[0] if pos.dim() == 2 else pos
        n = 1
        while n < pos.shape[0] and int(pos[n]) == int(pos[0]) + n: n += 1
        ids, pos, hin, ref = c["input_ids"][:n], pos[:n], c["hidden_in"][:n], c["sample_hidden"][:n].to(dev).float()
        if kv is not None:
            keep = kv.pos[0] < int(pos[0]); kv = M.KVState(kv.k[:, keep], kv.v[:, keep], kv.pos[:, keep])
        sh, _, kv = model(hin.to(dev, torch.bfloat16)[None], ids.to(dev)[None], pos.to(dev)[None], kv)
        sh = sh[0].float()
        for r in range(n):
            cc = float(torch.nn.functional.cosine_similarity(sh[r], ref[r], dim=0))
            re = float((sh[r] - ref[r]).norm() / ref[r].norm())
            cs.append(cc); rel.append(re); info.append((i, int(pos[r]), n))
t = torch.tensor(cs); e = torch.tensor(rel)
q = lambda x, p: float(torch.quantile(x, p))
out = {"rows": len(cs), "cos_median": q(t, .5), "cos_p01": q(t, .01), "cos_p05": q(t, .05), "cos_min": float(t.min()),
       "frac_cos_lt_0.999": float((t < .999).float().mean()), "frac_cos_lt_0.99": float((t < .99).float().mean()),
       "frac_cos_lt_0.98": float((t < .98).float().mean()),
       "relerr_median": q(e, .5), "relerr_p95": q(e, .95), "relerr_max": float(e.max()),
       "worst_rows": sorted(zip(cs, info))[:8]}
import collections
b = collections.defaultdict(list)
for re_, (ci, p, n) in zip(rel, info): b[min(p // 64, 7)].append(re_)
out["relerr_median_by_pos64"] = {"%d-%d" % (k*64, k*64+63 if k < 7 else 9999): round(float(torch.tensor(v).median()), 4) for k, v in sorted(b.items())}
out.pop("worst_rows")
print(json.dumps(out, indent=1))
