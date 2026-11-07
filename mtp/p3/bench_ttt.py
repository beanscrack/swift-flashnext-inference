"""Throughput and memory of the TTT forward (bf16 model, production router) on heldout sequences."""
import json, sys, time
import torch
sys.path.insert(0, "/u701/j692wu/swift-opt/mtp")
sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
from ttt import ttt_forward
from data import load_gen, load_sample
from verify_ttt import load_model_and_head

model, lm_rows, vocab = load_model_and_head()
base_mem = torch.cuda.memory_allocated()
recs = sorted([r for r in load_gen("heldout") if 1900 <= r["P"] + r["C"] <= 2048], key=lambda r: r["id"])
rec = recs[0]
hid, x, labels, T, P, C = load_sample(rec["id"])
res = {"T": T, "weights_bytes": base_mem}
with torch.no_grad():
    for _ in range(2):
        ttt_forward(model, hid, x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.time()
        S = ttt_forward(model, hid, x)
        torch.cuda.synchronize(); times.append(time.time() - t0)
    res["median_seconds"] = sorted(times)[2]
    res["tokens_per_s"] = T / res["median_seconds"]
    res["peak_mem_bytes_above_weights"] = torch.cuda.max_memory_allocated() - base_mem
    res["peak_mem_bytes_total"] = torch.cuda.max_memory_allocated()
    res["anchors_per_s"] = (T - 5) / res["median_seconds"]
print(json.dumps(res), flush=True)
json.dump(res, open("/u701/j692wu/swift-opt/mtp/p3/bench_ttt.json", "w"), indent=1)
