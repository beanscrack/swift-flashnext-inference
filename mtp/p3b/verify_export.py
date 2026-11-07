"""Verification (b): exported checkpoint -> fresh MTPDraft -> simulate.py ROUND-END on heldout.

usage: python -I verify_export.py <ckpt.safetensors> <step>
Checks the export format (31 names == mtp_original, shapes, dtype bf16), reruns simulate.py on a
fresh model that has only the exported tensors loaded, and compares with the in-training eval.
"""
import json
import sys

from safetensors import safe_open

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3b")
sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
from common import P3B, ORIG, load_base, load_ckpt_into, eval_round_end  # noqa: E402
from data import TAIL, CONFIG  # noqa: E402
from mtp_torch import MTPDraft  # noqa: E402
import torch  # noqa: E402

CKPT, STEP = sys.argv[1], int(sys.argv[2])
OUT = f"{P3B}/verify"


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    with safe_open(ORIG, framework="pt", device="cpu") as fo, safe_open(CKPT, framework="pt", device="cpu") as fc:
        orig_names = set(fo.keys())
        ck_names = set(fc.keys())
        shapes_ok = all(tuple(fo.get_slice(n).get_shape()) == tuple(fc.get_slice(n).get_shape()) for n in orig_names & ck_names)
        dtypes = {str(fc.get_slice(n).get_dtype()) for n in ck_names}
    fmt = {"n_tensors": len(ck_names), "names_equal_orig": ck_names == orig_names, "shapes_equal_orig": shapes_ok,
           "dtypes": sorted(dtypes), "has_embed_or_lm_head": any("embed" in n or "lm_head" in n for n in ck_names)}
    print("format", fmt, flush=True)
    assert fmt["n_tensors"] == 31 and fmt["names_equal_orig"] and fmt["shapes_equal_orig"] and dtypes == {"BF16"}

    model = MTPDraft.from_safetensors(TAIL, CONFIG, dtype=torch.bfloat16, device="cuda").eval()
    names = load_ckpt_into(model, CKPT)
    _, lm_head, lm_rows, vocab = load_base()  # frozen lm_head rows (same as training)
    d = eval_round_end(model, lm_rows, vocab, "verify_export", OUT)
    re = d["round_end"]
    ev = [json.loads(l) for l in open(f"{P3B}/logs/evals.jsonl") if json.loads(l)["step"] == STEP][0]
    diffs = [abs(a - b) for a, b in zip(re["acc"], ev["acc"])] + [abs(re["tokens_per_cycle"] - ev["tokens_per_cycle"])]
    res = {"ckpt": CKPT, "step": STEP, "format": fmt, "n_loaded": len(names),
           "reloaded_round_end": {"acc": re["acc"], "tokens_per_cycle": re["tokens_per_cycle"]},
           "in_training_eval": {"acc": ev["acc"], "tokens_per_cycle": ev["tokens_per_cycle"]},
           "max_abs_diff": max(diffs), "PASS_within_0.002": max(diffs) <= 0.002}
    json.dump(res, open(f"{OUT}/export_check.json", "w"), indent=1)
    print("EXPORT_CHECK", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
