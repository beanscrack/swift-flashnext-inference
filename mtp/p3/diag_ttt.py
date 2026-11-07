"""Diagnostic: per-step TTT vs sequential reference error, fp32 and bf16, one heldout sequence."""
import json
import random
import sys

import torch

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp")
sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
from mtp_torch import MTPDraft  # noqa: E402
from ttt import ttt_forward, K_STEPS  # noqa: E402
from data import load_gen, load_sample, TAIL, CONFIG  # noqa: E402
from verify_ttt import reference_anchor  # noqa: E402

rng = random.Random(1)
recs = [r for r in load_gen("heldout") if 800 <= r["P"] + r["C"] <= 1000]
rec = sorted(recs, key=lambda r: r["id"])[0]
out = {}
for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
    model = MTPDraft.from_safetensors(TAIL, CONFIG, dtype=dt, device="cuda").eval()
    hid, x, labels, T, P, C = load_sample(rec["id"])
    hid = hid.to(dt)
    with torch.no_grad():
        S = ttt_forward(model, hid, x)
        # step-0 check: TTT step 0 vs plain full-sequence forward
        s0_full, _, _ = model(hid[None], torch.cat([x[1:], x[-1:]])[None], torch.arange(T, device="cuda")[None], kv=None)
        e_list = [70, 300, 500, T - 6]
        rows = {}
        for e in e_list:
            ref = reference_anchor(model, hid, x, e)
            errs = []
            for j in range(K_STEPS):
                a = S[j, e].float()
                b = ref[j].float()
                errs.append(((a - b).norm() / b.norm()).item())
            rows[str(e)] = errs
        step0_full_err = ((s0_full[0, : T - 5].float() - S[0].float()).norm() / s0_full[0, : T - 5].float().norm()).item()
        # full-seq step0 vs reference step0 for e (prefix)
        ref0 = [reference_anchor(model, hid, x, e)[0] for e in e_list]
        ref0_vs_full = [((s0_full[0, e].float() - r.float()).norm() / r.float().norm()).item() for e, r in zip(e_list, ref0)]
    out[dt_name] = {"per_anchor_per_step_rel": rows, "step0_full_vs_ttt_step0_rel": step0_full_err,
                    "step0_full_vs_reference_step0_rel": ref0_vs_full}
    print(dt_name, json.dumps(out[dt_name]), flush=True)
    del model
    torch.cuda.empty_cache()
with open("/u701/j692wu/swift-opt/mtp/p3/diag_ttt.json", "w") as f:
    json.dump(out, f, indent=1)
