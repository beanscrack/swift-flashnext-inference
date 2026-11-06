"""Download the pinned Swift checkpoint once into the shared HF cache."""
import json
import os
from pathlib import Path

os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
MODEL = "ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4"
REVISION = "3ff0520224f264a2d0ac4ab56ece8f2f13aadb38"

if __name__ == "__main__":
    print(f"Downloading {MODEL}@{REVISION}", flush=True)
    snapshot = snapshot_download(
        MODEL,
        revision=REVISION,
        ignore_patterns=["*.mp4", "*.png"],
        max_workers=4,
    )
    (ROOT / "model-path.txt").write_text(snapshot + "\n")
    (ROOT / "model-revision.json").write_text(
        json.dumps({"model": MODEL, "revision": REVISION, "path": snapshot}, indent=2) + "\n"
    )
    print(f"Download complete: {snapshot}", flush=True)
