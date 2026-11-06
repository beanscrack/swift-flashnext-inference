"""Rebuild the multilingual draft vocabulary 'preset D' (126,868 rows) from its inputs and verify it equals the shipped file.
D = A  U  first 32,000 Latin-script tokens by token id, where
A = subset_49152.npy (clean English/code vocab)
    U first 24,000 Han / 14,000 Cyrillic / 8,000 Arabic tokens by token id
    U ALL Hangul, kana, Thai, Devanagari, Hebrew and byte-fragment tokens
    U the most frequent Latin tokens of the language-balanced multilingual calibration outputs (counts_multi.npy; up to 12,000, in practice 4,717 new)
    U the always-included ids (special/added tokens, EOS, single bytes, digits, whitespace)
Inputs: script_class.npy (script_prior.py), counts_multi.npy (build_multi_counts.py on calib/outputs.jsonl), subset_49152.npy, guaranteed_ids.json."""
import numpy as np, sys
from vocab_lib import NTOK, guaranteed
cls = np.load("script_class.npy")
ids = set(np.load("subset_49152.npy").tolist())
for s in ("hangul", "kana", "thai", "devanagari", "hebrew", "bytes"): ids |= set(np.nonzero(cls == s)[0].tolist())
for s, k in (("han", 24000), ("cyrillic", 14000), ("arabic", 8000)): ids |= set(np.nonzero(cls == s)[0][:k].tolist())
multi = np.load("counts_multi.npy"); lat = np.nonzero(cls == "latin")[0]
order = lat[np.argsort(-multi[lat], kind="stable")]
ids |= set([int(i) for i in order if multi[i] > 0 and int(i) not in ids][:12000])
G, _ = guaranteed(); ids |= set(G)
ids |= set(lat[:32000].tolist())
D = np.array(sorted(i for i in ids if i < NTOK), dtype=np.int32)
ref = np.load("subset_multi_D.npy")
print("rebuilt size", len(D), "| shipped size", len(ref), "| identical:", bool(len(D) == len(ref) and np.array_equal(D, ref)))
np.save("subset_multi_D_rebuilt.npy", D)
