"""Multilingual-safe preset: union of (a) the clean English/code vocabulary subset_49152.npy, (b) per-script quotas of the first K tokens of
each non-Latin script by token id (BPE merge-rank prior), (c) the most frequent Latin-script tokens of the language-balanced multilingual calibration
outputs.  Usage: make_multi_preset.py NAME HAN CYR ARA LATIN_DATA   (e.g. A 24000 14000 8000 12000); other scripts are included whole."""
import sys, json
import numpy as np
from vocab_lib import NTOK, V, guaranteed
name = sys.argv[1]; K = {"han": int(sys.argv[2]), "cyrillic": int(sys.argv[3]), "arabic": int(sys.argv[4])}; n_latin = int(sys.argv[5])
cls = np.load("script_class.npy")
ids = set(np.load("subset_49152.npy").tolist())
for s in ("hangul", "kana", "thai", "devanagari", "hebrew", "bytes"):
    ids |= set(np.nonzero(cls == s)[0].tolist())
for s, k in K.items():
    ids |= set(np.nonzero(cls == s)[0][:k].tolist())
multi = np.load("counts_multi.npy")                       # language-balanced P over the calibration outputs
lat = np.nonzero(cls == "latin")[0]
order = lat[np.argsort(-multi[lat], kind="stable")]
order = [int(i) for i in order if multi[i] > 0 and int(i) not in ids][:n_latin]
ids |= set(order)
G, _ = guaranteed(); ids |= set(G)
arr = np.array(sorted(i for i in ids if i < NTOK), dtype=np.int32)
np.save(f"subset_multi_{name}.npy", arr)
print(name, "size", len(arr), "(%.1f%% of vocab)" % (100.0 * len(arr) / V), "| quotas", K, "latin_data", len(order))
