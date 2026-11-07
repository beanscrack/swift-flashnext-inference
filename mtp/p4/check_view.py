#!/u701/j692wu/.conda/envs/inference/bin/python
"""Independent P4 checks on a view built by make_view.py (no torch needed; raw safetensors parsing).

  check_view.py --view DIR --orig-tail FILE --mtp-src FILE
  - sha256(DIR/model-tail.safetensors) == DIR/VIEW.json tail_sha256
  - every non-mtp tensor in DIR tail is byte-identical to orig tail
  - the 31 mtp.* tensors in DIR tail are byte-identical to mtp-src (best.safetensors)
  - reports which mtp tensors are byte-identical to the original (i.e. not replaced by a changed value)
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base = 8 + n
    meta = hdr.pop("__metadata__", None)
    return base, hdr, meta


def tensor_bytes(path, base, ent):
    s, e = ent["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        return f.read(e - s)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True)
    ap.add_argument("--orig-tail", required=True)
    ap.add_argument("--mtp-src", required=True)
    a = ap.parse_args()
    view = Path(a.view)
    tail = view / "model-tail.safetensors"
    out = {}

    meta = json.loads((view / "VIEW.json").read_text())
    got = sha256(tail)
    out["shard_sha256"] = got
    out["VIEW.json_tail_sha256"] = meta["tail_sha256"]
    out["check1_shard_sha256_matches_VIEW"] = got == meta["tail_sha256"]

    vb, vh, _ = header(tail)
    ob, oh, _ = header(a.orig_tail)
    sb, sh, _ = header(a.mtp_src)
    mtp_names = sorted(n for n in oh if n.startswith("mtp."))
    assert len(mtp_names) == 31, len(mtp_names)
    assert set(vh) == set(oh), "tensor name sets differ"

    non_mtp_diff, mtp_vs_src_diff, identical_to_orig = [], [], []
    for n in sorted(oh):
        if n.startswith("mtp."):
            continue
        if tensor_bytes(tail, vb, vh[n]) != tensor_bytes(a.orig_tail, ob, oh[n]):
            non_mtp_diff.append(n)
    n_non = len(oh) - len(mtp_names)
    for n in mtp_names:
        if tensor_bytes(tail, vb, vh[n]) != tensor_bytes(a.mtp_src, sb, sh[n]):
            mtp_vs_src_diff.append(n)
        if tensor_bytes(tail, vb, vh[n]) == tensor_bytes(a.orig_tail, ob, oh[n]):
            identical_to_orig.append(n)
    out["check2_non_mtp_tensors_compared"] = n_non
    out["check2_non_mtp_byte_diffs"] = non_mtp_diff
    out["check2_pass"] = not non_mtp_diff
    out["check3_mtp_tensors_compared"] = len(mtp_names)
    out["check3_mtp_byte_diffs_vs_src"] = mtp_vs_src_diff
    out["check3_pass"] = not mtp_vs_src_diff
    out["mtp_tensors_byte_identical_to_original"] = identical_to_orig
    out["mtp_tensors_changed_vs_original"] = len(mtp_names) - len(identical_to_orig)
    print(json.dumps(out, indent=2))
    ok = out["check1_shard_sha256_matches_VIEW"] and out["check2_pass"] and out["check3_pass"]
    print("ALL PASS" if ok else "FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
