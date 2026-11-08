#!/u701/j692wu/.conda/envs/inference/bin/python
"""Verify a view built by make_view.py.

  verify_view.py --view DIR --ck CK [--ref-tail FILE] [--expect-tensors 369]
      checks: (a) tail tensors vs reference tail, (c) symlinks/file count, (d) index + config/tokenizer files
  verify_view.py --tails-only --view-tail FILE --ref-tail FILE
      tensor-by-tensor compare only (used for the negative test)
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))
from stlib import sha256_file  # noqa: E402

TAIL = "model-tail.safetensors"
INDEX = "model.safetensors.index.json"
SIDE_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "chat_template.jinja", "generation_config.json", "preprocessor_config.json",
              "video_preprocessor_config.json", "hf_quant_config.json"]


def compare_tails(a, b):
    import torch
    from safetensors import safe_open
    with safe_open(str(a), "pt") as fa, safe_open(str(b), "pt") as fb:
        ka, kb = set(fa.keys()), set(fb.keys())
        missing = sorted(ka ^ kb)
        diffs = []
        for k in sorted(ka & kb):
            ta, tb = fa.get_tensor(k), fb.get_tensor(k)
            if ta.dtype != tb.dtype or tuple(ta.shape) != tuple(tb.shape) or not torch.equal(ta, tb):
                diffs.append(k)
    return len(ka & kb), diffs, missing


def files_under(root):
    out = set()
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            out.add(os.path.relpath(os.path.join(dp, f), root))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view")
    ap.add_argument("--ck")
    ap.add_argument("--ref-tail")
    ap.add_argument("--expect-tensors", type=int, default=369)
    ap.add_argument("--tails-only", action="store_true")
    ap.add_argument("--view-tail")
    a = ap.parse_args()

    if a.tails_only:
        n, diffs, missing = compare_tails(a.view_tail, a.ref_tail)
        print(json.dumps({"tensors_compared": n, "equal": n - len(diffs), "differing": diffs,
                          "key_mismatch": missing}, indent=2))
        return 0 if not missing else 1

    view = Path(os.path.abspath(a.view))
    ck = Path(a.ck)
    ref = Path(a.ref_tail) if a.ref_tail else (ck / TAIL).resolve()
    res, ok = {}, True

    # (a) tensor-by-tensor
    n, diffs, missing = compare_tails(view / TAIL, ref)
    res["a_tensors_equal"] = f"{n - len(diffs)}/{n}"
    res["a_differing"] = diffs
    res["a_key_mismatch"] = missing
    ok &= not diffs and not missing and n == a.expect_tensors
    res["a_tail_sha256"] = sha256_file(view / TAIL)

    # (c) symlinks and file count
    ck_files = files_under(ck)
    view_files = files_under(view) - {"VIEW.json"}
    bad = []
    if ck_files != view_files:
        bad.append({"set_mismatch": sorted(ck_files ^ view_files)[:20]})
    for rel in sorted(ck_files - {TAIL, INDEX}):
        vp, cp = view / rel, (ck / rel).resolve()
        if not vp.is_symlink():
            bad.append((rel, "not a symlink"))
            continue
        rp = Path(os.path.realpath(vp))
        if rp != cp or not rp.is_file():
            bad.append((rel, "bad target", str(rp)))
        elif rp.stat().st_size != cp.stat().st_size:
            bad.append((rel, "size mismatch"))
    res["c_file_count"] = {"ck": len(ck_files), "view": len(view_files)}
    res["c_failures"] = bad
    ok &= not bad and len(ck_files) == len(view_files)

    # (d) index and small files
    ci = json.loads((ck / INDEX).read_text())
    vi = json.loads((view / INDEX).read_text())
    wm_same = ci["weight_map"] == vi["weight_map"]
    rest_same = {k: v for k, v in ci.items() if k != "metadata"} == {k: v for k, v in vi.items() if k != "metadata"}
    res["d_weight_map_identical"] = wm_same
    res["d_other_index_keys_identical"] = rest_same
    res["d_total_size"] = {"ck": ci.get("metadata", {}).get("total_size"),
                           "view": vi.get("metadata", {}).get("total_size")}
    side = {}
    for rel in SIDE_FILES:
        if not (ck / rel).exists():
            continue
        vp = view / rel
        same = vp.exists() and sha256_file(vp.resolve()) == sha256_file((ck / rel).resolve())
        side[rel] = {"symlink": vp.is_symlink(), "identical": bool(same)}
    res["d_side_files"] = side
    ok &= wm_same and rest_same and all(v["symlink"] and v["identical"] for v in side.values())

    res["PASS"] = bool(ok)
    print(json.dumps(res, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
