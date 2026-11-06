"""Coverage of candidate draft vocabularies on HELD-OUT target outputs: the coding benchmark runs and the held-out (ood) runs.
Coverage = fraction of the target's generated tokens that lie inside the subset (upper bound on per-position acceptance retention).
Usage: eval_multi.py subset1.npy subset2.npy ..."""
import json, sys, glob, collections
import numpy as np
from transformers import AutoTokenizer
from vocab_lib import V
model = open("/u701/j692wu/swift-local/model-path.txt").read().strip()
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
R = "/u701/j692wu/swift-profile/runs"
sets = {}
for label, pattern in (("coding", f"{R}/base-132-marlin/rep*/requests.json"), ("heldout", f"{R}/ood-base/rep*/requests.json")):
    for f in sorted(glob.glob(pattern)):
        for r in json.load(open(f)):
            ids = np.asarray(tok.encode(r["output"], add_special_tokens=False), dtype=np.int64)
            sets.setdefault((label, r["case"]), []).append(ids)
cases = sorted(sets)
print("%-28s" % "subset" + "".join("%-18s" % (c[1][:16]) for c in cases))
for path in sys.argv[1:]:
    ids = np.load(path); m = np.zeros(V, dtype=bool); m[ids] = True
    row = []
    for c in cases:
        a = np.concatenate(sets[c]); row.append(m[a].mean())
    print("%-28s" % (path.split("/")[-1] + " (%d)" % len(ids)) + "".join("%-18s" % ("%.3f" % x) for x in row))
