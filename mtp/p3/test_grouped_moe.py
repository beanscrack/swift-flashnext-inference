"""Unit test: grouped (sort-by-expert) MoE == per-expert loop MoEBlock.forward.

fp32 model (real weights from model-tail.safetensors), several token counts, plus a bf16 pass
for information.  Pass criterion (fp32): max|grouped - loop| / max|loop| < 1e-3.
"""
import json
import sys

import torch

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
sys.path.insert(0, "/u701/j692wu/swift-opt/mtp")
from mtp_torch import MTPDraft  # noqa: E402
from grouped_moe import grouped_moe_forward  # noqa: E402

TAIL = "/u701/j692wu/swift-opt/mtp/view-identity/model-tail.safetensors"
CONFIG = "/u701/j692wu/swift-opt/mtp/view-identity/config.json"
OUT = "/u701/j692wu/swift-opt/mtp/p3/test_grouped_moe.json"


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


def main():
    torch.manual_seed(0)
    res = {}
    for dtype, tol_name in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        model = MTPDraft.from_safetensors(TAIL, CONFIG, dtype=dtype, device="cuda").eval()
        moe = model.layers[0].mlp
        H = model.cfg.hidden_size
        worst = 0.0
        rows = {}
        with torch.no_grad():
            for N in (1, 7, 64, 513, 2048):
                x = (torch.randn(N, H, device="cuda") * 0.5).to(dtype)
                ref = moe(x)
                got = grouped_moe_forward(moe, x)
                r = rel(got, ref)
                exact = torch.equal(got, ref)
                rows[str(N)] = {"max_rel_err": r, "bitwise_equal": exact}
                worst = max(worst, r)
            # real activations: post-attention-norm style inputs from a real sequence are not needed here;
            # the TTT verification exercises the MoE on real hidden states.
        res[tol_name] = {"worst_max_rel_err": worst, "by_N": rows, "pass": worst < 1e-3 if tol_name == "fp32" else None}
        print(tol_name, json.dumps(res[tol_name]), flush=True)
        del model, moe
        torch.cuda.empty_cache()
    ok = res["fp32"]["pass"]
    res["PASS"] = bool(ok)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print("PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
