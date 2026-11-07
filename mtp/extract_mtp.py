#!/u701/j692wu/.conda/envs/inference/bin/python
"""Write the 31 original mtp.* tensors from the checkpoint tail to --out (raw bytes, bitwise copy)."""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))
from stlib import check_under, sha256_file, tensor_entries, write_safetensors  # noqa: E402

TAIL = "model-tail.safetensors"
N_MTP = 31


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output .safetensors (must be under swift-opt/mtp/)")
    ap.add_argument("--ck", help="checkpoint snapshot dir (default: ~/swift-local/model-path.txt)")
    a = ap.parse_args()
    ck = Path(a.ck or Path(os.path.expanduser("~/swift-local/model-path.txt")).read_text().strip())
    assert ck.is_dir(), f"checkpoint snapshot missing: {ck}"
    out = Path(os.path.abspath(a.out))
    check_under(HERE, out)
    tail = (ck / TAIL).resolve()
    meta, ents = tensor_entries(tail)
    names = sorted(n for n in ents if n.startswith("mtp."))
    assert len(names) == N_MTP, f"expected {N_MTP} mtp tensors, found {len(names)}"
    entries = [dict(ents[n], name=n) for n in sorted(names, key=lambda k: ents[k]["offset"])]
    hashes = write_safetensors(str(out), entries, meta if meta else {"format": "pt"})
    print(json.dumps({"out": str(out), "file_sha256": sha256_file(out), "tensors": len(names),
                      "tensor_sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
