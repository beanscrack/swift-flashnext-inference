"""Where are the low-cosine rows between the main capture and the rerun capture of the same sequences?"""
import glob
import os

import torch

fs = [f for f in sorted(glob.glob("/tmp/mtp_hidden_rerun/*.pt")) if not f.endswith("labels.pt") and "bf16" not in f]


def deq(d):
    return d["fp8"].float() * d["scale"].float()[:, None]


allc, low, rel_pos, flips, n_lab = [], [], [], 0, 0
for f in fs:
    sid = os.path.basename(f)[:-3]
    a = torch.load(f)
    b = torch.load("/tmp/mtp_hidden/" + sid + ".pt")
    ha, hb = deq(a), deq(b)
    c = torch.nn.functional.cosine_similarity(ha, hb, dim=1)
    n = hb.norm(dim=1)
    allc.append(c)
    idx = torch.nonzero(c < 0.95).flatten().tolist()
    rel_pos += [i / ha.shape[0] for i in idx]
    for i in idx[:2]:
        low.append((round(float(c[i]), 3), i, ha.shape[0], round(float(n[i]), 1), round(float(n.median()), 1),
                    int(b["input_ids"][i])))
    la, lb = "/tmp/mtp_hidden_rerun/" + sid + ".labels.pt", "/tmp/mtp_hidden/" + sid + ".labels.pt"
    if os.path.exists(la) and os.path.exists(lb):
        ya, yb = torch.load(la), torch.load(lb)
        ya = ya["target_argmax"] if isinstance(ya, dict) else ya
        yb = yb["target_argmax"] if isinstance(yb, dict) else yb
        flips += int((ya != yb).sum())
        n_lab += int(ya.numel())
c = torch.cat(allc)
print("rows", c.numel(), "median %.5f" % float(c.median()),
      "frac<0.99 %.4f  frac<0.95 %.4f  frac<0.9 %.4f" % tuple(float((c < t).float().mean()) for t in (0.99, 0.95, 0.9)))
print("rows<0.95: n=%d, mean relative position %.2f" % (len(rel_pos), sum(rel_pos) / max(len(rel_pos), 1)))
print("label flips between captures: %d / %d = %.4f" % (flips, n_lab, flips / max(n_lab, 1)))
print("examples (cos, pos, T, row_norm, median_norm, token_id):")
for e in low[:14]:
    print("  ", e)
