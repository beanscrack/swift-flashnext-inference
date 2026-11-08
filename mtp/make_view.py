#!/u701/j692wu/.conda/envs/inference/bin/python
"""Build a checkpoint view whose MTP draft weights come from --mtp, without copying the checkpoint.

  make_view.py --mtp <file.safetensors with exactly the 31 mtp.* tensors | original> --out <view_dir> [--ck SNAPSHOT_DIR]

view_dir gets:
  - symlinks (to resolved blob paths) for every file of CK except model-tail.safetensors and the index
  - a new model-tail.safetensors: all tensors of the original tail, with the 31 mtp.* tensors taken from --mtp
    (raw bytes; same names/dtypes/shapes asserted), original __metadata__ preserved
  - model.safetensors.index.json: same weight_map, metadata.total_size recomputed when present
  - VIEW.json: source paths, sha256 of new tail, per-tensor sha256 of the mtp tensors
--ck defaults to the contents of ~/swift-local/model-path.txt. --out must be under swift-opt/mtp/.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))
from stlib import check_under, sha256_file, tensor_entries, write_safetensors  # noqa: E402

TAIL = "model-tail.safetensors"
INDEX = "model.safetensors.index.json"
N_MTP = 31


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtp", required=True, help='safetensors with exactly the 31 mtp.* tensors, or "original"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--ck", help="checkpoint snapshot dir")
    a = ap.parse_args()

    ck = Path(a.ck or Path(os.path.expanduser("~/swift-local/model-path.txt")).read_text().strip())
    assert ck.is_dir(), f"checkpoint snapshot missing: {ck}"
    out = Path(os.path.abspath(a.out))
    check_under(HERE, out)

    index_path = ck / INDEX
    index = json.loads(index_path.read_text())
    wm = index["weight_map"]
    tail_src = (ck / TAIL).resolve()
    assert tail_src.is_file(), f"tail missing: {tail_src}"
    orig_meta, orig = tensor_entries(tail_src)
    assert {n for n, s in wm.items() if s == TAIL} == set(orig), "index tail names != tail header names"
    mtp_names = sorted(n for n in orig if n.startswith("mtp."))
    assert len(mtp_names) == N_MTP, f"expected {N_MTP} mtp tensors in tail, found {len(mtp_names)}"

    if a.mtp == "original":
        mtp_file = tail_src
        new = {n: orig[n] for n in mtp_names}
    else:
        mtp_file = Path(os.path.abspath(a.mtp))
        assert mtp_file.is_file(), f"--mtp missing: {mtp_file}"
        _, new = tensor_entries(mtp_file)
        assert set(new) == set(mtp_names), (
            f"missing={sorted(set(mtp_names) - set(new))} extra={sorted(set(new) - set(mtp_names))}")
        for n in mtp_names:
            assert new[n]["dtype"] == orig[n]["dtype"], f"{n} dtype {new[n]['dtype']} != {orig[n]['dtype']}"
            assert new[n]["shape"] == orig[n]["shape"], f"{n} shape {new[n]['shape']} != {orig[n]['shape']}"
            assert new[n]["nbytes"] == orig[n]["nbytes"], n

    # New tail: original tensor order, mtp tensors swapped in.
    entries = []
    for n in sorted(orig, key=lambda k: orig[k]["offset"]):
        src = new[n] if n in new else orig[n]
        entries.append({"name": n, "dtype": src["dtype"], "shape": src["shape"],
                        "file": src["file"], "offset": src["offset"], "nbytes": src["nbytes"]})

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    hashes = write_safetensors(str(out / TAIL), entries, orig_meta)

    # Symlinks for everything else in CK.
    n_links = 0
    for dirpath, _dirs, files in os.walk(ck):
        rel_dir = Path(dirpath).relative_to(ck)
        (out / rel_dir).mkdir(parents=True, exist_ok=True)
        for fn in files:
            rel = rel_dir / fn
            if str(rel) in (TAIL, INDEX):
                continue
            target = Path(os.path.realpath(ck / rel))
            assert target.is_file(), f"{rel} does not resolve to a file: {target}"
            os.symlink(target, out / rel)
            n_links += 1

    # Index: same weight_map; total_size recomputed from tensor bytes across all shards.
    total = sum(e["nbytes"] for e in entries)
    for shard in sorted(set(wm.values()) - {TAIL}):
        _, ents = tensor_entries((ck / shard).resolve())
        names = {n for n, s in wm.items() if s == shard}
        assert set(ents) == names, f"{shard}: header names != index names"
        total += sum(e["nbytes"] for e in ents.values())
    orig_total = index.get("metadata", {}).get("total_size")
    if "total_size" in index.get("metadata", {}):
        index["metadata"]["total_size"] = total
        (out / INDEX).write_text(json.dumps(index, indent=2) + "\n")
    else:
        shutil.copyfile(index_path, out / INDEX)

    mtp_hashes = {n: hashes[n] for n in mtp_names}
    view = {
        "source_ck": str(ck),
        "source_tail": str(tail_src),
        "tail_file": TAIL,
        "tail_sha256": sha256_file(out / TAIL),
        "tail_tensor_count": len(entries),
        "mtp_source": str(mtp_file),
        "mtp_source_sha256": "same as source_tail" if a.mtp == "original" else sha256_file(mtp_file),
        "mtp_tensor_sha256": mtp_hashes,
        "index_total_size_original": orig_total,
        "index_total_size_recomputed": total,
        "symlinked_files": n_links,
    }
    (out / "VIEW.json").write_text(json.dumps(view, indent=2) + "\n")
    print(json.dumps({"out": str(out), "tail_sha256": view["tail_sha256"], "symlinks": n_links,
                      "tail_tensors": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
