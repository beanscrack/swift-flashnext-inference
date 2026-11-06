"""Summarize runs/*: decode TPS (median/variability), TTFT, MTP acceptance, power/clocks, backend, output sanity."""
import json, statistics as st, sys, glob, os, csv

P = "/u701/j692wu/swift-profile/runs"
BASE_REF = "base-orig-marlin"

def pct(xs, q):
    xs = sorted(xs); k = (len(xs) - 1) * q; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def load(label):
    d = os.path.join(P, label)
    status = json.load(open(os.path.join(d, "status.json"))) if os.path.exists(os.path.join(d, "status.json")) else {"status": "missing", "valid": False}
    reps = []
    for f in sorted(glob.glob(os.path.join(d, os.environ.get("SUBDIR", ""), "rep*/requests.json")), key=lambda x: int(os.path.basename(os.path.dirname(x))[3:])):
        reps.append(json.load(open(f)))
    return d, status, reps

def tele(d):
    f = os.path.join(d, "telemetry.csv")
    if not os.path.exists(f): return {}
    rows = list(csv.reader(open(f)))[1:]
    def num(s):
        try: return float(s.split()[0])
        except Exception: return None
    pw = [num(r[1]) for r in rows if len(r) > 5 and (num(r[5]) or 0) > 50]
    sm = [num(r[2]) for r in rows if len(r) > 5 and (num(r[5]) or 0) > 50]
    mem = [num(r[4]) for r in rows if len(r) > 5]
    pw = [x for x in pw if x]; sm = [x for x in sm if x]; mem = [x for x in mem if x]
    return {"power_w_mean": round(st.mean(pw), 1) if pw else None, "sm_clock_mhz_mean": round(st.mean(sm)) if sm else None,
            "gpu_mem_mib_max": max(mem) if mem else None}

def spec(reps):
    tot = {}
    for rep in reps:
        for r in rep:
            for k, v in r.get("counter_deltas", {}).items():
                tot[k] = tot.get(k, 0) + v
    g = lambda s: next((v for k, v in tot.items() if k.endswith(s)), None)
    drafts, prop, acc = g("spec_decode_num_drafts_total"), g("spec_decode_num_draft_tokens_total"), g("spec_decode_num_accepted_tokens_total")
    if not drafts: return {}
    return {"acceptance": round(acc / prop, 4), "tokens_per_verify": round(1 + acc / drafts, 3), "verify_steps": int(drafts)}

def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

def main():
    labels = sys.argv[1:] or sorted(os.listdir(P))
    ref_out = {}
    d0, s0, r0 = load(BASE_REF)
    for rep in r0:
        for r in rep:
            ref_out.setdefault(r["case"], r["output"])
    out = []
    for label in labels:
        d, status, reps = load(label)
        row = {"label": label, "status": status.get("status"), "valid": status.get("valid"), "detail": status.get("detail"),
               "backend": status.get("backend"), "nspec": status.get("nspec"), "cuda_env": status.get("cuda_env")}
        if reps:
            allr = [r for rep in reps for r in rep]
            tps = [r["decode_tps"] for r in allr]
            means = [st.mean(r["decode_tps"] for r in rep) for rep in reps]
            row.update({"reps": len(reps), "requests": len(allr),
                        "median_tps": round(st.median(tps), 2), "mean_tps": round(st.mean(tps), 2),
                        "stdev_tps": round(st.pstdev(tps), 2), "p10": round(pct(tps, .1), 2), "p90": round(pct(tps, .9), 2),
                        "min": round(min(tps), 2), "max": round(max(tps), 2),
                        "rep_means": [round(m, 2) for m in means], "median_of_rep_means": round(st.median(means), 2),
                        "ttft_mean_s": round(st.mean(r["ttft_s"] for r in allr), 3)})
            row.update(spec(reps))
            if row.get("tokens_per_verify"):
                row["cycle_ms"] = round(1000 * row["tokens_per_verify"] / row["mean_tps"], 2)
            if label != BASE_REF and ref_out:
                agree = [lcp(r["output"], ref_out[r["case"]]) / max(1, len(ref_out[r["case"]])) for r in allr if r["case"] in ref_out]
                row["mean_prefix_agreement_with_baseline"] = round(st.mean(agree), 3) if agree else None
            row["sample_output_head"] = allr[0]["output"][:160].replace("\n", " ")
            row.update(tele(d))
        out.append(row)
    print(json.dumps(out, indent=1))
    hdr = f"{'label':26} {'valid':5} {'backend':20} {'n':3} {'median':>7} {'mean':>7} {'sd':>5} {'p10':>7} {'p90':>7} {'accept':>6} {'tok/cyc':>7} {'cyc_ms':>6} {'ttft':>6} {'W':>5} {'MHz':>5}"
    print("\n" + hdr)
    for r in out:
        print(f"{r['label']:26} {str(r['valid'])[:5]:5} {str(r.get('backend')):20} {r.get('requests', 0):3} {r.get('median_tps', ''):>7} {r.get('mean_tps', ''):>7} "
              f"{r.get('stdev_tps', ''):>5} {r.get('p10', ''):>7} {r.get('p90', ''):>7} {r.get('acceptance', ''):>6} {r.get('tokens_per_verify', ''):>7} "
              f"{r.get('cycle_ms', ''):>6} {r.get('ttft_mean_s', ''):>6} {str(r.get('power_w_mean', '')):>5} {str(r.get('sm_clock_mhz_mean', '')):>5}")

main()
