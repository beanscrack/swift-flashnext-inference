import json, glob, os, collections
import torch
from safetensors import safe_open
torch.manual_seed(0)
D = "/u701/j692wu/swift-opt/runs/p1-parity-v2/captures"
recs = json.load(open("/u701/j692wu/swift-opt/mtp/verify_p1b/recs.json"))
files = sorted(glob.glob(os.path.join(D, "call_*.pt")))
calls = [torch.load(f, map_location="cpu") for f in files]
with safe_open("/u701/j692wu/swift-opt/mtp/view-identity/model-tail.safetensors", framework="pt") as f:
    lm = f.get_tensor("lm_head.weight")
print("lm", tuple(lm.shape), lm.dtype)
lmf = lm.float()
def top2(h):
    lg = h.float() @ lmf.t()
    v, ix = lg.topk(2, dim=-1)
    return ix[:, 0], v[:, 0] - v[:, 1]
ref_tok, ref_gap = [], []
prev_sh = None
tot = collections.Counter()
b_prev = b_pert = b_const = 0
gaps = []
all_ref = []
pert_rel = []
rows = 0
for i, c in enumerate(calls):
    nv = recs[i]["nv"]
    sh = c["sample_hidden"][:nv]
    t, g = top2(sh)
    all_ref.append(t)
    gaps.append(g)
    rows += nv
    # B1: previous call's sample_hidden (same row index, clamped) as predictor
    if prev_sh is not None:
        idx = torch.clamp(torch.arange(nv), max=prev_sh.shape[0] - 1)
        tp, _ = top2(prev_sh[idx])
        b_prev += int((tp == t).sum()); tot["prev"] += nv
    # B2: relative 1% random perturbation of the reference hidden (cos ~ 0.99995)
    noise = torch.randn_like(sh.float())
    sp = sh.float() + noise * (0.01 * sh.float().norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True))
    cs = torch.nn.functional.cosine_similarity(sp, sh.float(), dim=-1).min().item()
    pert_rel.append(cs)
    tpert, _ = top2(sp)
    b_pert += int((tpert == t).sum()); tot["pert"] += nv
    prev_sh = sh
allt = torch.cat(all_ref); allg = torch.cat(gaps)
print("rows", rows, "ref argmax distinct tokens", int(allt.unique().numel()))
vals, cnt = allt.unique(return_counts=True)
top = cnt.argmax()
print("most common ref token id", int(vals[top]), "share", round(int(cnt[top]) / rows, 4))
b_const = int((allt == vals[top]).sum())
print("B0 constant-predictor (always most common token) agreement: %.4f" % (b_const / rows))
print("B1 previous-call sample_hidden same-row predictor agreement: %.4f over %d rows" % (b_prev / tot["prev"], tot["prev"]))
print("B2 1%%-relative random perturbation agreement: %.4f over %d rows (min cos of perturbed %.5f)" % (b_pert / tot["pert"], tot["pert"], min(pert_rel)))
print("top1-top2 logit gap (fp32 of ref hidden): min %.5f, #rows gap<0.05: %d, gap<0.2: %d" % (float(allg.min()), int((allg < 0.05).sum()), int((allg < 0.2).sum())))
