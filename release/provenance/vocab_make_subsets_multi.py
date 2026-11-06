"""Multilingual-safe draft vocabularies: blend of the English/code a-priori mixture (v1) and a language-balanced family of the target's own
multilingual answers. Writes subset_<N>_m.npy (nested, int32, sorted, forced ids included) and manifest_multi.json.
Usage: make_subsets_multi.py [W_MULTI=0.35] [counts_prefix=counts_multi]"""
import json, sys
import numpy as np
from vocab_lib import *
w_multi = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
prefix = sys.argv[2] if len(sys.argv) > 2 else "counts_multi"
SIZES_M = [49152, 65536, 81920, 98304, 131072]
cnt, model = load_counts(); fam = families(cnt, model)
base = mixture(fam, W_DEFAULT)            # a-priori English/code mixture (the clean v1 reference)
multi = np.load(f"{D}/{prefix}.npy")
vec = (1.0 - w_multi) * base + w_multi * multi
G, gjson = guaranteed()
man = {"w_multi": w_multi, "base": "v1 a-priori mixture W_DEFAULT", "multi": prefix, "sizes": SIZES_M, "forced": len(G), "subsets": {}}
prev = None
for n in SIZES_M:
    ids = select(vec, n, G); np.save(f"{D}/subset_{n}_m.npy", ids)
    m = mask_of(ids)
    man["subsets"][str(n)] = {"file": f"subset_{n}_m.npy", "calib_mass_base": float(P(base)[m].sum()), "calib_mass_multi": float(multi[m].sum())}
    print(n, "mass coverage: base(en/code) %.4f  multilingual %.4f" % (P(base)[m].sum(), multi[m].sum()))
    s = set(ids.tolist()); assert prev is None or prev <= s, "not nested"; prev = s
json.dump(man, open(f"{D}/manifest_multi.json", "w"), indent=1)
