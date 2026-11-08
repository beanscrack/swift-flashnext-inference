#!/usr/bin/env python3
"""parity_v2.py - hidden-level parity of mtp_torch.MTPDraft against vLLM's Qwen4ExpMTP.forward (V2 runner).

Input: call_XXXXX.pt files written by swift_mtp_capture.py in standalone mode (SWIFT_MTP_CAPTURE_STANDALONE=1),
one per MTP forward of a single greedy request, in call order. Each holds input_ids [T], positions [T] (or [3,T]),
hidden_in [T, 10240], sample_hidden [T, 2560], multi_hidden [T, 10240].

Calls are replayed in order through MTPDraft with a carried draft KV cache. Before each call only cache entries at
positions < min(call positions) are kept: vLLM writes each call's K/V into the slots of its positions, so later
writes at the same positions replace earlier ones (rejected draft rows are rewritten next round).
Reports per call: cosine and max-rel error of sample_hidden / multi_hidden, and top-1 agreement of
argmax(lm_head @ sample_hidden) between vLLM and torch (full vocabulary, bf16 lm_head).
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_torch as M  # noqa: E402


def cos(a, b):
    a, b = a.float().flatten(1), b.float().flatten(1)
    return torch.nn.functional.cosine_similarity(a, b, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True, help="safetensors with mtp.*, embed_tokens and lm_head (model-tail)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.captures, "call_*.pt")))
    if not files:
        raise SystemExit("no call_*.pt captures in %s" % a.captures)
    dev = torch.device(a.device)
    model = M.MTPDraft.from_safetensors(a.weights, a.config, dtype=torch.bfloat16, device=dev).eval()
    from safetensors import safe_open
    with safe_open(a.weights, framework="pt", device=str(dev)) as f:
        lm = f.get_tensor("lm_head.weight").to(torch.bfloat16)
    kv = None
    rows = []
    agree = total = 0
    agree_by = {"prefill": [0, 0], "step": [0, 0]}
    with torch.no_grad():
        for i, fn in enumerate(files):
            c = torch.load(fn, map_location="cpu")
            ids, pos, hin = c["input_ids"], c["positions"], c["hidden_in"]
            if pos.dim() == 2:
                pos = pos[0]
            # V2 always runs k+1 rows per verify call; only the leading rows with consecutive positions are real.
            # Trailing rows (rejected / padding, e.g. positions [.., 3, 4]) are computed but not kept, so drop them.
            n_valid = 1
            while n_valid < pos.shape[0] and int(pos[n_valid]) == int(pos[0]) + n_valid:
                n_valid += 1
            dropped = pos.shape[0] - n_valid
            ids, pos, hin = ids[:n_valid], pos[:n_valid], hin[:n_valid]
            for key in ("sample_hidden", "multi_hidden"):
                if key in c:
                    c[key] = c[key][:n_valid]
            T = ids.shape[0]
            start = int(pos[0])
            if kv is None and start != 0:
                raise SystemExit("first call starts at position %d (capture must start at the request start)" % start)
            if kv is not None:
                keep = kv.pos[0] < start
                kv = M.KVState(kv.k[:, keep], kv.v[:, keep], kv.pos[:, keep])
            sh, mh, kv = model(hin.to(dev, torch.bfloat16)[None], ids.to(dev)[None], pos.to(dev)[None], kv)
            sh, mh = sh[0], mh[0]
            r = {"call": i, "T": T, "dropped": dropped, "pos0": start, "spec_step_idx": int(c.get("spec_step_idx", -1))}
            if "sample_hidden" in c:
                ref_s = c["sample_hidden"].to(dev)
                ref_m = c["multi_hidden"].to(dev)
                r["cos_sample_min"] = float(cos(sh, ref_s).min())
                r["cos_multi_min"] = float(cos(mh, ref_m).min())
                r["rel_sample"] = float((sh.float() - ref_s.float()).abs().max() / ref_s.float().abs().max())
                t_ours = (sh.to(torch.bfloat16) @ lm.t()).argmax(-1)
                t_ref = (ref_s.to(torch.bfloat16) @ lm.t()).argmax(-1)
                n_ok = int((t_ours == t_ref).sum())
                r["top1_agree"] = n_ok / T
                agree += n_ok
                total += T
                kind = "prefill" if T > 1 else "step"
                agree_by[kind][0] += n_ok
                agree_by[kind][1] += T
            rows.append(r)
    summary = {
        "calls": len(rows), "rows_compared": total,
        "top1_agreement": agree / max(total, 1),
        "top1_agreement_prefill_calls": agree_by["prefill"][0] / max(agree_by["prefill"][1], 1),
        "top1_agreement_step_calls": agree_by["step"][0] / max(agree_by["step"][1], 1),
        "rows_prefill": agree_by["prefill"][1], "rows_step": agree_by["step"][1],
        "cos_sample_min_median": float(torch.tensor([r["cos_sample_min"] for r in rows if "cos_sample_min" in r]).median()),
        "cos_sample_min_worst": min(r["cos_sample_min"] for r in rows if "cos_sample_min" in r),
    }
    json.dump({"summary": summary, "calls": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("first calls:", json.dumps(rows[:6]))


if __name__ == "__main__":
    main()
