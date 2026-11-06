"""Per-language token counts from the target's own sampled answers to the multilingual calibration prompts.
Usage: build_multi_counts.py OUTPUTS.jsonl [OUT_PREFIX]   -> counts_multi_<lang>.npy, counts_multi.npy (language-balanced mean of P), multi_stats.json"""
import json, sys, collections
import numpy as np
from transformers import AutoTokenizer
from vocab_lib import D, V, P
src = sys.argv[1]; prefix = sys.argv[2] if len(sys.argv) > 2 else "counts_multi"
model = open("/u701/j692wu/swift-local/model-path.txt").read().strip()
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
by_lang = collections.defaultdict(lambda: np.zeros(V, dtype=np.int64)); ntok = collections.Counter(); nreq = collections.Counter()
for ln in open(src, encoding="utf-8"):
    r = json.loads(ln)
    if not r["output"].strip() or r["n_tokens"] < 20: continue
    ids = np.asarray(tok.encode(r["output"], add_special_tokens=False), dtype=np.int64)
    by_lang[r["lang"]] += np.bincount(ids, minlength=V)[:V]; ntok[r["lang"]] += len(ids); nreq[r["lang"]] += 1
langs = sorted(by_lang)
for l in langs: np.save(f"{D}/{prefix}_{l}.npy", by_lang[l])
bal = np.mean([P(by_lang[l]) for l in langs], axis=0)
np.save(f"{D}/{prefix}.npy", bal)
json.dump({"languages": langs, "tokens": dict(ntok), "requests": dict(nreq), "source": src}, open(f"{D}/{prefix}_stats.json", "w"), indent=1)
print("languages:", len(langs), "total tokens:", sum(ntok.values())); print({l: (nreq[l], ntok[l]) for l in langs})
