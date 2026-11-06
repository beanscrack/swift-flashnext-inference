"""Concurrency test against a live server: C simultaneous streams, each cycling through the coding prompts.
Reports aggregate output tokens/s (all streams) and mean per-request decode tps.  Usage: bench_conc.py OUT.json C1 C2 ..."""
import json, sys, threading, time, statistics as st
sys.path.insert(0, "/u701/j692wu/swift-local")
from experiment_bench import request
cases = json.load(open("/u701/j692wu/swift-local/experiments/20261021/decode-cases.json"))
base = "http://127.0.0.1:8000"
request(base, dict(cases[0], output_tokens=32))            # warm
result = {}
for C in [int(x) for x in sys.argv[2:]]:
    for rep in range(2):
        res, lock = [], threading.Lock()
        def worker(w):
            for k in range(len(cases)):
                item = cases[(k + w) % len(cases)]
                r = request(base, item)
                with lock: res.append(r)
        t0 = time.perf_counter()
        ts = [threading.Thread(target=worker, args=(w,)) for w in range(C)]
        [t.start() for t in ts]; [t.join() for t in ts]
        wall = time.perf_counter() - t0
        toks = sum(r["usage"]["completion_tokens"] for r in res)
        result[f"c{C}_rep{rep}"] = {"requests": len(res), "wall_s": wall, "aggregate_tps": toks / wall,
                                    "mean_request_decode_tps": st.mean(r["decode_tps"] for r in res), "mean_ttft_s": st.mean(r["ttft_s"] for r in res)}
        print("C=%d rep%d: aggregate %.1f tok/s, mean per-request decode %.1f tok/s" % (C, rep, toks / wall, result[f"c{C}_rep{rep}"]["mean_request_decode_tps"]), flush=True)
json.dump(result, open(sys.argv[1], "w"), indent=1)
