#!/usr/bin/env python3
"""make_fp8_plan.py bench.json [bench2.json ...] --key w8row --out plan.json
Turn fp8_bench.py winners into the SWIFT_FP8_PLAN format {"NxK": {"M": W8Conf fields}} (later files win)."""
import argparse
import json
import re


def parse(label):
    # "w8fma bn=8 bk=512 sk=1 w=4"  |  "w8dot bn=32 bk=128 sk=1 w=4 st=3 bm=16"
    kind = label.split()[0][2:]
    kv = dict(re.findall(r"(\w+)=(\d+)", label))
    c = {"kind": kind, "block_n": int(kv["bn"]), "block_k": int(kv["bk"]), "split_k": int(kv["sk"]),
         "num_warps": int(kv["w"])}
    if kind == "dot":
        c["num_stages"] = int(kv["st"])
        c["block_m"] = int(kv["bm"])
    return c


ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--key", default="w8row")
ap.add_argument("--out", required=True)
a = ap.parse_args()
plan = {}
for f in a.files:
    for r in json.load(open(f)):
        lab = r.get(a.key + "_cfg")
        if lab:
            plan.setdefault("%dx%d" % (r["n"], r["k"]), {})[str(r["m"])] = parse(lab)
json.dump(plan, open(a.out, "w"), indent=1)
print({k: sorted(v, key=int) for k, v in plan.items()})
