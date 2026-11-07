"""parity_check.py - per-step top-1 agreement of MTPDraft against vLLM draft tokens.

Input: .pt rounds written by swift_mtp_capture.py (V1 runner, eager).  Needs CUDA (or --device cpu,
slow).  For every captured request row it runs MTPDraft.unroll() with
  * the target hidden / token ids / positions / next token that vLLM fed to step 0,
  * the trimmed draft vocab: W_sub = lm_head[ids] (bf16, or fp8 round-trip with --lm-head-source fp8,
    via release/hooks/fp8_gemv.py quantize_fp8/dequantize_fp8, the same path as swift_draft_vocab.py),
  * ids = the kept-id .npy (sorted unique; argmax ties -> lowest id, as in swift_draft_vocab.py).
It then reports, per draft step j, the fraction of requests where my token == vLLM's token, and
sanity checks on the captured inputs (step-0 ids/hidden, chain ids/positions for steps >= 1).

Example (GPU node, after a capture run with SWIFT_MTP_CAPTURE_DIR=... VLLM_USE_V2_MODEL_RUNNER=0):
  python parity_check.py --captures /path/to/captures \
      --config $CK/config.json --weights $CK/model-tail.safetensors \
      --vocab full     # V1 captures are full-vocabulary (trimmed-vocab hook only patches the V2 speculator)
      --lm-head-source bf16 --device cuda --out /path/to/parity.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mtp_torch as M  # noqa: E402
from mtp_torch import MTPDraft  # noqa: E402

HOOKS_DIR = "/u701/j692wu/swift-opt/release/hooks"


def load_lm_head(path_st: str, source: str, group: int) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(path_st, framework="pt", device="cpu") as f:
        if "lm_head.weight" not in f.keys():
            raise KeyError(f"lm_head.weight not in {path_st}")
        w = f.get_tensor("lm_head.weight")
    if source == "fp8":
        sys.path.insert(0, HOOKS_DIR)
        import fp8_gemv  # release/hooks/fp8_gemv.py

        w8, scale = fp8_gemv.quantize_fp8(w.to("cuda" if torch.cuda.is_available() else "cpu"), group)
        w = fp8_gemv.dequantize_fp8(w8, scale, group).cpu()
    elif source != "bf16":
        raise ValueError(source)
    return w.to(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True, help="safetensors holding mtp.* and embed_tokens (model-tail)")
    ap.add_argument("--lm-head-file", default=None, help="safetensors with lm_head.weight (default: --weights)")
    ap.add_argument("--vocab", required=True, help="'full' (V1 capture) or .npy kept target token ids")
    ap.add_argument("--lm-head-source", choices=["bf16", "fp8"], default="bf16")
    ap.add_argument("--fp8-group", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-rounds", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device)
    dtype = torch.bfloat16
    model = MTPDraft.from_safetensors(args.weights, args.config, dtype=dtype, device=dev)
    if not model.embed_loaded:
        raise RuntimeError("embed_tokens.weight not found in --weights; the draft needs the target embedding")
    W = load_lm_head(args.lm_head_file or args.weights, args.lm_head_source, args.fp8_group)
    # "full": full-vocabulary argmax, which is what the V1 runner produces (swift_draft_vocab.py only
    # patches the V2 MTPSpeculator, so a V1 capture is never trimmed).  A .npy path gives the trimmed head.
    ids = np.arange(W.shape[0], dtype=np.int64) if args.vocab == "full" else np.load(args.vocab).astype(np.int64)
    if not (np.all(ids[:-1] < ids[1:]) and ids.min() >= 0 and ids.max() < W.shape[0]):
        raise ValueError("vocab ids must be sorted, unique and within lm_head rows")
    ids_t = torch.from_numpy(ids).to(dev)
    W_sub = W[torch.from_numpy(ids)].to(dev)

    files = sorted(glob.glob(os.path.join(args.captures, "mtp_round_*.pt")))
    if args.max_rounds:
        files = files[: args.max_rounds]
    if not files:
        raise SystemExit(f"no captures in {args.captures}")

    k_hist = {}
    step_match = None
    full_match = 0
    n_req = 0
    step0_in_bad = 0
    step0_in_total = 0
    chain_bad = 0
    chain_total = 0
    hid_diff = []
    hid_cos = []
    per_round = []
    kv = None  # draft KV of the single request, carried across rounds (see MTPDraft.unroll)
    for path in files:
        rec = torch.load(path, map_location="cpu", weights_only=False)
        a = rec["propose_args"]
        draft = rec["draft_token_ids"]  # [B, k] vLLM
        B, k = draft.shape
        if B != 1:
            raise SystemExit(f"{path}: B={B}; parity needs one request at a time (--max-num-seqs 1, one client request)")
        k_hist[k] = k_hist.get(k, 0) + 1
        if step_match is None:
            step_match = np.zeros(k, dtype=np.int64)
        tok = a["target_token_ids"].long()
        nxt = a["next_token_ids"].long()
        pos = a["target_positions"]
        if pos.dim() == 2:  # M-RoPE target: text positions are identical across axes
            pos = pos[0]
        pos = pos.long()
        hid = a["target_hidden_states"]
        cad = a["common_attn_metadata"]
        qsl = cad["query_start_loc"].long().tolist()
        tti = a.get("token_indices_to_sample")
        tti = tti.long().tolist() if tti is not None else [qsl[1] - 1]
        calls = rec["calls"]
        c0 = calls[0]
        s, e = qsl[0], tti[0]

        # sanity: captured step-0 inputs equal what set_inputs_first_pass builds (llm_base_proposer.py:839-847)
        exp_ids = torch.empty_like(tok)
        exp_ids[:-1] = tok[1:]
        exp_ids[e] = nxt[0]
        rows0 = torch.arange(s, e + 1)
        step0_in_total += rows0.numel()
        step0_in_bad += int((c0["input_ids"][rows0] != exp_ids[rows0]).sum())

        # KV of earlier rounds: keep only positions before this round's first row (later writes overwrite)
        start = int(pos[s])
        if kv is None and start != 0:
            raise SystemExit("first captured round starts at position %d, not 0: capture must begin at the request start" % start)
        if kv is not None:
            keep = kv.pos[0] < start
            kv = M.KVState(kv.k[:, keep], kv.v[:, keep], kv.pos[:, keep])
        seg_tok = tok[s : e + 1].to(dev)[None]
        seg_hid = hid[s : e + 1].to(dev, dtype)[None]
        seg_pos = pos[s : e + 1].to(dev)[None]
        with torch.no_grad():
            out = model.unroll(seg_hid, seg_tok, k=k, lm_head_rows=W_sub, id_table=ids_t,
                               next_token_ids=nxt[:1].to(dev), positions=seg_pos, kv=kv)
        kv = out["kv"]
        mine = out["draft_ids"][0].cpu()
        theirs = draft[0].long()
        eq = (mine == theirs)
        step_match[:k] += eq.numpy().astype(np.int64)
        full_match += int(bool(eq.all()))
        n_req += 1
        # chain consistency: vLLM's step-j input id must be draft_{j-1}, position pos[e] + j
        for j in range(1, min(k, len(calls))):
            cj = calls[j]
            chain_total += 1
            if int(cj["input_ids"][0]) != int(theirs[j - 1]) or int(cj["positions"][0]) != int(pos[e]) + j:
                chain_bad += 1
        if "sample_hidden" in c0:
            mine_h = out["sample_hidden"][0, 0].float().cpu()
            ref_h = c0["sample_hidden"][e].float()
            hid_diff.append(float((mine_h - ref_h).abs().max().detach()))
            hid_cos.append(float(torch.nn.functional.cosine_similarity(mine_h, ref_h.detach(), dim=0)))
        per_round.append({"file": os.path.basename(path), "start": start, "end": e, "k": k,
                          "match_per_step": eq.numpy().astype(np.int64).tolist(),
                          "vllm": theirs.tolist(), "mine": mine.tolist()})

    report = {
        "rounds": len(files),
        "requests": n_req,
        "k_values": k_hist,
        "top1_agreement_per_step": (step_match / max(n_req, 1)).round(6).tolist() if step_match is not None else [],
        "full_chain_agreement": full_match / max(n_req, 1),
        "step0_input_mismatch": f"{step0_in_bad}/{step0_in_total}",
        "chain_input_position_mismatch": f"{chain_bad}/{chain_total}",
        "sample_hidden_max_abs_diff_step0": max(hid_diff) if hid_diff else None,
        "sample_hidden_min_cos_step0": min(hid_cos) if hid_cos else None,
        "lm_head_source": args.lm_head_source,
        "vocab_rows": int(ids.size),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"report": report, "per_round": per_round}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
