"""P3b shared helpers (no training logic).

- load_base(): MTPDraft loaded from the same TAIL file the P3 eval path uses (bf16, cuda),
  the full frozen lm_head and the 49,152 draft-vocab rows.
- sha(): sha256 of a tensor's bytes (frozen-weight checks).
- eval_round_end(): runs p3/simulate.py main() unchanged, with the in-memory model substituted
  for its loader, so ROUND-END numbers come from simulate.py's own code.
- export_bf16() / load_ckpt_into(): the 31 mtp.* tensors with original names, bf16, safetensors.
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp")
sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
from mtp_torch import MTPDraft  # noqa: E402
from data import TAIL, CONFIG, VOCAB  # noqa: E402
import simulate  # noqa: E402  (import only: verify_ttt reads sys.argv, writes nothing)

ORIG = "/u701/j692wu/swift-opt/mtp/mtp_original.safetensors"
P3B = "/u701/j692wu/swift-opt/mtp/p3b"
# simulate.py round-end numbers for the ORIGINAL weights (p3/sim_results_bf16.json)
ORIG_RT = {
    "acc": [0.8119258747942442, 0.6549540699835396, 0.5309430255402751, 0.43122444645037966],
    "tokens_per_cycle": 3.429047416768438,
}


def load_base(device="cuda"):
    model = MTPDraft.from_safetensors(TAIL, CONFIG, dtype=torch.bfloat16, device=device).eval()
    with safe_open(TAIL, framework="pt", device="cpu") as f:
        lm_head = f.get_tensor("lm_head.weight").to(device, torch.bfloat16)
    vocab = torch.from_numpy(np.load(VOCAB).astype(np.int64)).to(device)
    lm_rows = lm_head[vocab].contiguous()
    return model, lm_head, lm_rows, vocab


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.int16).numpy().tobytes()).hexdigest()


def eval_round_end(model, lm_rows, vocab, tag, outdir):
    """Run simulate.py main() (ROUND-END + overall + by-mode) on the given in-memory model."""
    os.makedirs(outdir, exist_ok=True)
    out_json = os.path.join(outdir, f"sim_{tag}.json")
    out_npz = os.path.join(outdir, f"preds_{tag}.npz")
    simulate.load_model_and_head = lambda: (model, lm_rows, vocab)
    simulate.OUT = out_json
    simulate.PREDS = out_npz
    simulate.main()
    with open(out_json) as f:
        return json.load(f)


def export_bf16(named, path):
    """named: list of (name without 'mtp.' prefix, param). Writes atomically via a .tmp file."""
    tensors = {"mtp." + n: p.detach().to(torch.bfloat16).contiguous().cpu() for n, p in named}
    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata={"format": "pt"})
    os.replace(tmp, path)


def load_ckpt_into(model, path):
    with safe_open(path, framework="pt", device="cpu") as f:
        tensors = {n: f.get_tensor(n) for n in f.keys()}
    model.load_mtp_tensors(tensors)
    return sorted(tensors)
