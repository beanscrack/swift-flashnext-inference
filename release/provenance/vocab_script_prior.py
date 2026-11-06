"""How well does 'first K tokens of a script, by token id' cover each language's real outputs? (BPE merge-rank prior)"""
import json, glob, unicodedata, collections
import numpy as np
from transformers import AutoTokenizer
from vocab_lib import NTOK
model = open("/u701/j692wu/swift-local/model-path.txt").read().strip()
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
def script_of(s):
    if not s or "�" in s: return "bytes"
    c = collections.Counter()
    for ch in s:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF: c["han"] += 1
        elif 0x3040 <= o <= 0x30FF: c["kana"] += 1
        elif 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF: c["hangul"] += 1
        elif 0x0400 <= o <= 0x04FF: c["cyrillic"] += 1
        elif 0x0600 <= o <= 0x06FF: c["arabic"] += 1
        elif 0x0590 <= o <= 0x05FF: c["hebrew"] += 1
        elif 0x0900 <= o <= 0x097F: c["devanagari"] += 1
        elif 0x0E00 <= o <= 0x0E7F: c["thai"] += 1
        elif ch.isalpha() and o < 0x250 or 0x1E00 <= o <= 0x1EFF: c["latin"] += 1
    return c.most_common(1)[0][0] if c else "other"
cls = np.array([script_of(tok.decode([i])) for i in range(NTOK)])
np.save("script_class.npy", cls)
cnt = collections.Counter(cls.tolist()); print("tokens per script class:", dict(cnt.most_common()))
lang_scripts = {"zh-Hans": ["han"], "zh-Hant": ["han"], "ja": ["han", "kana"], "ko": ["hangul"], "ru": ["cyrillic"], "uk": ["cyrillic"], "ar": ["arabic"], "he": ["hebrew"], "hi": ["devanagari"], "th": ["thai"]}
outs = collections.defaultdict(list)
for ln in open("../calib/outputs.jsonl", encoding="utf-8"):
    r = json.loads(ln); outs[r["lang"]].append(np.asarray(tok.encode(r["output"], add_special_tokens=False)))
held = {}
for f in sorted(glob.glob("/u701/j692wu/swift-profile/runs/ood-base/rep*/requests.json")):
    for r in json.load(open(f)):
        if r["case"].startswith("chinese"): held.setdefault(r["case"], []).append(np.asarray(tok.encode(r["output"], add_special_tokens=False)))
def curve(name, arrs, scripts, Ks):
    a = np.concatenate(arrs); sc = cls[np.minimum(a, NTOK - 1)]
    mask_lang = np.isin(sc, scripts + ["bytes"]); frac_script = mask_lang.mean()
    ids_sorted = {s: np.sort(np.nonzero(cls == s)[0]) for s in scripts}
    out = []
    for K in Ks:
        cov = np.zeros(NTOK, dtype=bool)
        for s in scripts: cov[ids_sorted[s][:K]] = True
        cov |= (cls == "bytes")
        out.append("%dK:%.3f" % (K // 1000, cov[np.minimum(a, NTOK - 1)][mask_lang].mean()))
    print("%-22s script-token share %.2f | coverage of its script tokens (by id prefix): %s" % (name, frac_script, "  ".join(out)))
Ks = [2000, 4000, 8000, 12000, 16000, 24000, 32000]
for l, sc in lang_scripts.items(): curve("calib " + l, outs[l], sc, Ks)
for c, arrs in held.items(): curve("HELD-OUT " + c, arrs, ["han"], Ks)
