"""Token-exactness between runs on a DETERMINISTIC MoE backend (CUTLASS): per (round, case) identical output? first divergence (char index)?
Usage: compare_exact.py REFLABEL LABEL [LABEL ...]  (compares rep1 and rep2 of each against rep1 of REFLABEL, and rep1 vs rep2 within each)"""
import json, sys
P = "/u701/j692wu/swift-profile/runs"
def load(label, rep):
    return {(r["phase"], r["case"]): r["output"] for r in json.load(open(f"{P}/{label}/rep{rep}/requests.json"))}
def cmp(a, b):
    same, div = 0, []
    for k in a:
        if k not in b: continue
        if a[k] == b[k]: same += 1
        else:
            n = next((i for i, (x, y) in enumerate(zip(a[k], b[k])) if x != y), min(len(a[k]), len(b[k])))
            div.append((k[1], n, len(a[k])))
    return same, len(a), div
ref = sys.argv[1]
R1 = load(ref, 1)
for lab in [ref] + sys.argv[2:]:
    for rep in (1, 2):
        if lab == ref and rep == 1: continue
        try: other = load(lab, rep)
        except Exception as e: print(lab, "rep%d" % rep, "unavailable", e); continue
        s, n, d = cmp(R1, other)
        print("%-22s rep%d vs %s rep1: %2d/%d byte-identical outputs%s" % (lab, rep, ref, s, n, ("" if not d else " | first divergence at char: " + ", ".join("%s@%d/%d" % x for x in d[:8]))))
for lab in sys.argv[1:]:
    try:
        s, n, d = cmp(load(lab, 1), load(lab, 2)); print("%-22s rep1 vs rep2 (same server): %2d/%d identical" % (lab, s, n))
    except Exception as e: print(lab, "rep pair unavailable", e)
