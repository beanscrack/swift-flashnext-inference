"""Sample loading for P3: gen.jsonl metadata + /tmp/mtp_hidden (gpu-1-vm) hidden states and labels."""
import json

import torch

GEN = "/u701/j692wu/swift-opt/mtp/p2/gen.jsonl"
HID = "/tmp/mtp_hidden"
TAIL = "/u701/j692wu/swift-opt/mtp/view-identity/model-tail.safetensors"
CONFIG = "/u701/j692wu/swift-opt/mtp/view-identity/config.json"
VOCAB = "/u701/j692wu/swift-opt/release/vocab/swift_draft_vocab_code49152.npy"


def load_gen(split=None):
    """Metadata only (no token-id lists kept beyond what is needed)."""
    out = []
    with open(GEN) as f:
        for line in f:
            r = json.loads(line)
            if split is not None and r["split"] != split:
                continue
            out.append({
                "id": r["id"],
                "split": r["split"],
                "mode": r["mode"],
                "finish_reason": r["finish_reason"],
                "P": len(r["prompt_token_ids"]),
                "C": len(r["completion_token_ids"]),
                "prompt_token_ids": r["prompt_token_ids"],
                "completion_token_ids": r["completion_token_ids"],
                "messages": r["messages"],
                "request": r["request"],
                "max_tokens": r["max_tokens"],
            })
    return out


def load_sample(sid, device="cuda"):
    """Returns hidden [T, 10240] bf16 (dequantised fp8 rowscale), x [T] int64, labels [T-1] int64, meta."""
    d = torch.load(f"{HID}/{sid}.pt", map_location="cpu", weights_only=False)
    assert d["format"] == "fp8_e4m3_rowscale_v1" and d["nonfinite"] == 0
    fp8 = d["fp8"]
    T = fp8.shape[0]
    pos = d["positions"].long()
    assert torch.equal(pos, torch.arange(T)), "positions must be arange(T)"
    hid = (fp8.float() * d["scale"][:, None]).to(device=device, dtype=torch.bfloat16)
    x = d["input_ids"].long().to(device)
    lab = torch.load(f"{HID}/{sid}.labels.pt", map_location="cpu", weights_only=False)
    assert lab["T"] == T
    labels = lab["target_argmax"].long().to(device)
    assert labels.shape[0] == T - 1
    return hid, x, labels, T, int(lab["P"]), int(lab["C"])
