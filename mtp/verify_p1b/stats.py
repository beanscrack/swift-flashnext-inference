import glob, os, json, collections
import torch
D = "/u701/j692wu/swift-opt/runs/p1-parity-v2/captures"
files = sorted(glob.glob(os.path.join(D, "call_*.pt")))
calls = [torch.load(f, map_location="cpu") for f in files]
print("n files", len(files))
keys = collections.Counter(tuple(sorted(c.keys())) for c in calls)
print("key sets", keys)
recs = []
for i, c in enumerate(calls):
    pos = c["positions"]
    pdim = pos.dim()
    p = pos[0] if pdim == 2 else pos
    T = p.shape[0]
    nv = 1
    while nv < T and int(p[nv]) == int(p[0]) + nv:
        nv += 1
    recs.append(dict(i=i, T=T, pdim=pdim, nv=nv, dropped=T - nv, pos=[int(x) for x in p],
                     step=int(c.get("spec_step_idx", -1))))
print("position dims", collections.Counter(r["pdim"] for r in recs))
print("spec_step_idx", collections.Counter(r["step"] for r in recs))
print("T histogram", sorted(collections.Counter(r["T"] for r in recs).items()))
print("dropped histogram", sorted(collections.Counter(r["dropped"] for r in recs).items()))
print("calls with dropped>0:", sum(r["dropped"] > 0 for r in recs), " total dropped rows:", sum(r["dropped"] for r in recs), " total rows:", sum(r["T"] for r in recs), " kept rows:", sum(r["nv"] for r in recs))
ex = [r for r in recs if r["dropped"] > 0][:5]
for r in ex:
    print("EX call", r["i"], "T", r["T"], "nv", r["nv"], "pos", r["pos"], "step", r["step"])
rel = collections.Counter()
for r in recs:
    if r["dropped"] == 0:
        continue
    p0 = r["pos"][0]; kend = p0 + r["nv"]
    for x in r["pos"][r["nv"]:]:
        if x < p0: rel["< pos0"] += 1
        elif x < kend: rel["inside kept range"] += 1
        else: rel["> kept end"] += 1
print("dropped row position classes", dict(rel))
print("dropped rows equal to p0+nv:", sum(1 for r in recs if r["dropped"] and r["pos"][r["nv"]] == r["pos"][0] + r["nv"]))
chain_ok = chain_bad = pos_ok = pos_bad = 0
maxdiff = []
for k, c in enumerate(calls):
    r = recs[k]
    if r["T"] == 1 and k > 0 and "multi_hidden" in calls[k-1]:
        pr = recs[k-1]
        prev_row = calls[k-1]["multi_hidden"][pr["nv"] - 1].float()
        cur = c["hidden_in"][0].float()
        d = float((prev_row - cur).abs().max())
        maxdiff.append(d)
        if d == 0.0: chain_ok += 1
        else: chain_bad += 1
        if r["pos"][0] == pr["pos"][pr["nv"] - 1] + 1: pos_ok += 1
        else: pos_bad += 1
print("T==1 calls after a call: hidden_in == prev multi_hidden[last kept] exact:", chain_ok, " not exact:", chain_bad, " pos==prev last+1:", pos_ok, " not:", pos_bad)
if maxdiff: print("maxabs diff: min %.4g max %.4g" % (min(maxdiff), max(maxdiff)))
kinds = collections.Counter(("prefill" if r["nv"] > 1 else "step", r["step"]) for r in recs)
print("trimmed kinds (kind, spec_step_idx):", dict(kinds))
print("untrimmed T>1 vs T==1:", dict(collections.Counter(("T>1" if r["T"] > 1 else "T==1") for r in recs)))
json.dump(recs, open("/u701/j692wu/swift-opt/mtp/verify_p1b/recs.json", "w"))
print("first 12 calls kept pos:", [(r["i"], r["pos"][:r["nv"]]) for r in recs[:12]])
print("last 5 calls:", [(r["i"], r["T"], r["nv"], r["pos"][:r["nv"]]) for r in recs[-5:]])
