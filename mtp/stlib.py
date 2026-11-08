"""Raw-byte safetensors helpers: parse headers and copy tensor bytes without re-encoding."""
import hashlib
import json
import math
import os
import struct

ITEMSIZE = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
            "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
            "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def check_under(root, path):
    root = os.path.abspath(root)
    path = os.path.abspath(path)
    assert path != root and os.path.commonpath([root, path]) == root, f"{path} is not under {root}"


def sha256_file(path, chunk=64 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n).decode("utf-8"))
    return header, 8 + n


def tensor_entries(path):
    """Return (metadata, {name: entry}); entry has dtype/shape/file/offset/nbytes (offset is absolute)."""
    path = os.path.abspath(path)
    header, base = read_header(path)
    meta = header.pop("__metadata__", None)
    size = os.path.getsize(path)
    out = {}
    for name, info in header.items():
        start, end = info["data_offsets"]
        nbytes = end - start
        assert base + end <= size, f"{name} overruns {path}"
        if info["dtype"] in ITEMSIZE:
            expect = math.prod(info["shape"]) * ITEMSIZE[info["dtype"]]
            assert expect == nbytes, f"{name}: header size {nbytes} != {expect}"
        out[name] = {"dtype": info["dtype"], "shape": info["shape"], "file": path,
                     "offset": base + start, "nbytes": nbytes}
    return meta, out


def write_safetensors(out_path, entries, metadata, chunk=64 << 20):
    """Write entries (dicts with name/dtype/shape/file/offset/nbytes) in the given order.

    Returns {name: sha256 of that tensor's bytes}. Written via a temp file then renamed.
    """
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    pos = 0
    for e in entries:
        assert e["name"] not in header, f"duplicate tensor {e['name']}"
        header[e["name"]] = {"dtype": e["dtype"], "shape": e["shape"],
                             "data_offsets": [pos, pos + e["nbytes"]]}
        pos += e["nbytes"]
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hjson += b" " * ((8 - len(hjson) % 8) % 8)
    hashes = {}
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for e in entries:
            h = hashlib.sha256()
            remaining = e["nbytes"]
            with open(e["file"], "rb") as src:
                src.seek(e["offset"])
                while remaining:
                    b = src.read(min(chunk, remaining))
                    assert b, f"short read on {e['file']} for {e['name']}"
                    f.write(b)
                    h.update(b)
                    remaining -= len(b)
            hashes[e["name"]] = h.hexdigest()
    os.replace(tmp, out_path)
    return hashes
