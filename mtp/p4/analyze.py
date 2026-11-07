#!/u701/j692wu/.conda/envs/inference/bin/python
"""P4 analysis: compare p4-identity vs p4-mtp-v1 (and p0-identity reference).

  analyze.py [--hash] [--check-p0-only]
Reads only runs under /u701/j692wu/swift-opt/runs/<LABEL>/ and writes /u701/j692wu/swift-opt/mtp/p4/analysis.json.
"""
import argparse
import hashlib
import json
import re
import statistics as st
from pathlib import Path

RUNS = Path("/u701/j692wu/swift-opt/runs")
MTP = Path("/u701/j692wu/swift-opt/mtp")
OUT = MTP / "p4" / "analysis.json"
ID, MV, P0 = "p4-identity", "p4-mtp-v1", "p0-identity"
REF_LP, REF_T1 = -0.2124, 0.9753
P0_LP, P0_T1 = -0.2115, 0.9775
LP_SPREAD, T1_SPREAD = 0.0014, 0.004  # from the brief: ~0.0014 logprob, ~0.4 pt top1
OOD_TOL = 0.03

SPEC = re.compile(r'^(vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens)_total'
                  r'|vllm:spec_decode_num_accepted_tokens_per_pos_total)(\{[^}]*\})? (\S+)$')


def last_json(path):
    lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
    return json.loads(lines[-1])


def counters(path):
    c = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0, "pos": {}}
    for line in Path(path).read_text().splitlines():
        m = SPEC.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        if name.endswith("per_pos_total"):
            pos = int(re.search(r'position="(\d+)"', labels).group(1))
            c["pos"][pos] = c["pos"].get(pos, 0.0) + val
        elif "draft_tokens" in name:
            c["draft_tokens"] += val
        elif "accepted" in name:
            c["accepted"] += val
        else:
            c["drafts"] += val
    return c


def delta(a, b):
    d = {"drafts": b["drafts"] - a["drafts"], "draft_tokens": b["draft_tokens"] - a["draft_tokens"],
         "accepted": b["accepted"] - a["accepted"], "pos": {}}
    for k in sorted(set(a["pos"]) | set(b["pos"])):
        d["pos"][k] = b["pos"].get(k, 0.0) - a["pos"].get(k, 0.0)
    return d


def add(a, b):
    d = {"drafts": a["drafts"] + b["drafts"], "draft_tokens": a["draft_tokens"] + b["draft_tokens"],
         "accepted": a["accepted"] + b["accepted"], "pos": {}}
    for k in set(a["pos"]) | set(b["pos"]):
        d["pos"][k] = a["pos"].get(k, 0.0) + b["pos"].get(k, 0.0)
    return d


def zero():
    return {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0, "pos": {}}


def acc_summary(c):
    if c["drafts"] == 0:
        return None
    return {"drafts": c["drafts"], "accepted": c["accepted"],
            "tokens_per_cycle": c["accepted"] / c["drafts"] + 1.0,
            "pos_accept_rate": {k: c["pos"][k] / c["drafts"] for k in sorted(c["pos"])}}


def phase_counters(run, sub):
    """Sum before/after deltas of every rep dir under run/sub (sub='' for coding, 'ood' for OOD)."""
    base = RUNS / run / sub if sub else RUNS / run
    tot = zero()
    n = 0
    for d in sorted(base.glob("rep[0-9]*")):
        if not d.is_dir():
            continue
        b, a = d / "metrics-before.txt", d / "metrics-after.txt"
        if b.exists() and a.exists():
            tot = add(tot, delta(counters(b), counters(a)))
            n += 1
    return tot, n


def reps_tps(run, sub=""):
    base = RUNS / run / sub if sub else RUNS / run
    vals = []
    for f in sorted(base.glob("rep[0-9].out"), key=lambda p: int(p.stem[3:])):
        vals.append(last_json(f)["mean_decode_tps"])
    return vals


def score(run):
    txt = (RUNS / run / "extra.out").read_text()
    m = re.search(r"mean logprob (-?[\d.]+); top1 fraction ([\d.]+)", txt)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def conc_lines(run):
    f = RUNS / run / "conc.out"
    if not f.exists():
        return {}, 0
    txt = f.read_text()
    res = {}
    for m in re.finditer(r"C=(\d+) rep(\d): aggregate ([\d.]+) tok/s, mean per-request decode ([\d.]+)", txt):
        res.setdefault(int(m.group(1)), []).append(float(m.group(3)))
    return res, txt.count("Traceback")


def conc_expected_and_done(run):
    cases = json.loads((RUNS / run / "decode-cases.json").read_text())
    f = RUNS / run / "conc.json"
    done = json.loads(f.read_text()) if f.exists() else {}
    out = {}
    for k, v in done.items():
        C = int(k.split("_")[0][1:])
        out.setdefault(C, []).append((v["requests"], len(cases) * C))
    return out


def server_checks(run):
    txt = (RUNS / run / "server.log").read_text(errors="replace")
    lines = txt.splitlines()
    tracebacks = sum(1 for l in lines if "Traceback" in l)
    errors = [l[:200] for l in lines if re.search(r"\bERROR\b", l)]
    models = sorted(set(re.findall(r"Initializing a V1 LLM engine .*? model='([^']+)'", txt)))
    return {"tracebacks": tracebacks, "error_lines": len(errors), "error_samples": errors[:3], "engine_model_paths": models}


def driver_failures(run):
    txt = (RUNS / run / "driver.log").read_text()
    return [l for l in txt.splitlines() if "FAILED" in l or "ABORT" in l or "NOT VALID" in l or "CLEANUP PROBLEM" in l]


def texts(run, sub, rep):
    f = RUNS / run / sub / f"rep{rep}" / "requests.json" if sub else RUNS / run / f"rep{rep}" / "requests.json"
    return {r["case"]: r.get("output") or "" for r in json.loads(f.read_text())}


def compare_texts(tok, A, B):
    same, div = 0, []
    for case in sorted(set(A) & set(B)):
        if A[case] == B[case]:
            same += 1
            continue
        ta = tok.encode(A[case]).ids
        tb = tok.encode(B[case]).ids
        pos = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        div.append({"case": case, "first_div_token": pos, "tokens_a": len(ta), "tokens_b": len(tb)})
    return same, len(set(A) & set(B)), div


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true", help="recompute sha256 of the view tail (slow)")
    a = ap.parse_args()

    res = {"runs": {}}
    for run in (ID, MV, P0):
        r = {}
        tps = reps_tps(run)
        r["coding_tps_reps"] = tps
        if tps:
            r["coding_tps_mean"], r["coding_tps_min"], r["coding_tps_max"] = st.mean(tps), min(tps), max(tps)
        ood = reps_tps(run, "ood")
        r["ood_tps_reps"] = ood
        if ood:
            r["ood_tps_mean"] = st.mean(ood)
        final = counters(RUNS / run / "metrics-final.txt")
        r["acc_final_all_phases"] = acc_summary(final)
        coding, nc = phase_counters(run, "")
        r["acc_coding_phase"] = acc_summary(coding)
        r["coding_phase_reps"] = nc
        oodc, no = phase_counters(run, "ood")
        r["acc_ood_phase"] = acc_summary(oodc)
        lp, t1 = score(run)
        r["logprob_mean"], r["top1_fraction"] = lp, t1
        conc, ctb = conc_lines(run)
        r["conc_aggregate_tps"] = {C: v for C, v in sorted(conc.items())}
        r["conc_expected_vs_done"] = conc_expected_and_done(run)
        r["conc_tracebacks"] = ctb
        r["server"] = server_checks(run)
        r["driver_failures"] = driver_failures(run)
        res["runs"][run] = r

    # output sanity
    from tokenizers import Tokenizer
    view_identity = json.loads((MTP / "view-identity" / "VIEW.json").read_text())
    tok = Tokenizer.from_file(str(Path(view_identity["source_ck"]) / "tokenizer.json"))
    reps = [1, 2, 3, 4, 5]
    san = {"pairs_identity_vs_mtp_same_rep": [], "noise_identity_rep1_vs_rep2": None}
    id_rep = {k: texts(ID, "", k) for k in reps if (RUNS / ID / f"rep{k}" / "requests.json").exists()}
    mv_rep = {k: texts(MV, "", k) for k in reps if (RUNS / MV / f"rep{k}" / "requests.json").exists()}
    for k in sorted(set(id_rep) & set(mv_rep)):
        same, n, div = compare_texts(tok, id_rep[k], mv_rep[k])
        san["pairs_identity_vs_mtp_same_rep"].append({"rep": k, "identical": same, "cases": n, "divergent": div})
    if 1 in id_rep and 2 in id_rep:
        same, n, div = compare_texts(tok, id_rep[1], id_rep[2])
        san["noise_identity_rep1_vs_rep2"] = {"identical": same, "cases": n, "divergent": div}
    res["output_sanity"] = san

    # view tail hash
    vh = {"VIEW_json_tail_sha256": json.loads((MTP / "view-mtp-v1" / "VIEW.json").read_text())["tail_sha256"]}
    if a.hash:
        h = hashlib.sha256()
        with open(MTP / "view-mtp-v1" / "model-tail.safetensors", "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        vh["file_sha256_now"] = h.hexdigest()
        vh["match"] = vh["file_sha256_now"] == vh["VIEW_json_tail_sha256"]
    res["shard"] = vh

    # gates
    gi, gm = res["runs"][ID], res["runs"][MV]
    spread_id = gi["coding_tps_max"] - gi["coding_tps_min"]
    lo_lp, hi_lp = min(REF_LP, P0_LP) - LP_SPREAD, max(REF_LP, P0_LP) + LP_SPREAD
    lo_t1, hi_t1 = min(REF_T1, P0_T1) - T1_SPREAD, max(REF_T1, P0_T1) + T1_SPREAD

    def stress_ok(r):
        ok = r["conc_tracebacks"] == 0 and r["server"]["tracebacks"] == 0 and not r["driver_failures"]
        for C, pairs in r["conc_expected_vs_done"].items():
            ok &= all(d == e for d, e in pairs)
        ok &= set(r["conc_aggregate_tps"]) == {1, 2, 4, 8}
        return bool(ok)

    gates = [
        ("coding tok/s: mtp-v1 mean > identity mean by > identity spread",
         gm["coding_tps_mean"] - gi["coding_tps_mean"] > spread_id,
         f"mtp {gm['coding_tps_mean']:.2f} vs id {gi['coding_tps_mean']:.2f}, diff {gm['coding_tps_mean'] - gi['coding_tps_mean']:+.2f}, id spread {spread_id:.2f}"),
        ("tokens/cycle (coding phase) higher for mtp-v1",
         gm["acc_coding_phase"]["tokens_per_cycle"] > gi["acc_coding_phase"]["tokens_per_cycle"],
         f"mtp {gm['acc_coding_phase']['tokens_per_cycle']:.4f} vs id {gi['acc_coding_phase']['tokens_per_cycle']:.4f}"),
        ("OOD tok/s: mtp-v1 not > 3% below identity",
         gm["ood_tps_mean"] >= (1 - OOD_TOL) * gi["ood_tps_mean"],
         f"mtp {gm['ood_tps_mean']:.2f} vs id {gi['ood_tps_mean']:.2f}, ratio {gm['ood_tps_mean'] / gi['ood_tps_mean']:.4f}"),
        ("logprob within reference spread (both runs)",
         all(lo_lp <= r["logprob_mean"] <= hi_lp and lo_t1 <= r["top1_fraction"] <= hi_t1 for r in (gi, gm)),
         f"id {gi['logprob_mean']:.4f}/{gi['top1_fraction']*100:.2f}%, mtp {gm['logprob_mean']:.4f}/{gm['top1_fraction']*100:.2f}%; window lp [{lo_lp:.4f},{hi_lp:.4f}] top1 [{lo_t1*100:.2f},{hi_t1*100:.2f}]%"),
        ("concurrency stress clean (both runs)",
         stress_ok(gi) and stress_ok(gm),
         f"id done={gi['conc_expected_vs_done']} tb={gi['conc_tracebacks']}/{gi['server']['tracebacks']}; mtp done={gm['conc_expected_vs_done']} tb={gm['conc_tracebacks']}/{gm['server']['tracebacks']}"),
        ("replacement shard loaded (engine path = view-mtp-v1, tail hash = VIEW.json)",
         gm["server"]["engine_model_paths"] == [str(MTP / "view-mtp-v1")] and vh.get("match", True),
         f"engine paths {gm['server']['engine_model_paths']}; hash match {vh.get('match', 'not rechecked (run with --hash)')}"),
    ]
    res["gates"] = [{"item": g[0], "pass": bool(g[1]), "detail": g[2]} for g in gates]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str) + "\n")

    # human-readable
    for run in (P0, ID, MV):
        r = res["runs"][run]
        print(f"== {run}")
        if "coding_tps_mean" in r:
            print(f"  coding tok/s reps {[round(x,2) for x in r['coding_tps_reps']]} mean {r['coding_tps_mean']:.2f} min {r['coding_tps_min']:.2f} max {r['coding_tps_max']:.2f}")
        if "ood_tps_mean" in r:
            print(f"  ood tok/s reps {[round(x,2) for x in r['ood_tps_reps']]} mean {r['ood_tps_mean']:.2f}")
        for key in ("acc_final_all_phases", "acc_coding_phase", "acc_ood_phase"):
            s = r[key]
            if s:
                pr = " ".join(f"p{k}={v:.3f}" for k, v in s["pos_accept_rate"].items())
                print(f"  {key}: drafts={s['drafts']:.0f} accepted={s['accepted']:.0f} tok/cycle={s['tokens_per_cycle']:.4f} pos[{pr}]")
        print(f"  logprob {r['logprob_mean']} top1 {r['top1_fraction']}")
        print(f"  conc aggregate tok/s by level {r['conc_aggregate_tps']} done {r['conc_expected_vs_done']}")
        print(f"  server tracebacks {r['server']['tracebacks']} error_lines {r['server']['error_lines']} engine paths {r['server']['engine_model_paths']}")
        if r["driver_failures"]:
            print(f"  driver failures: {r['driver_failures']}")
    print("== output sanity")
    for p in san["pairs_identity_vs_mtp_same_rep"]:
        print(f"  identity vs mtp-v1 rep{p['rep']}: identical {p['identical']}/{p['cases']}; divergent {[(d['case'], d['first_div_token']) for d in p['divergent']]}")
    if san["noise_identity_rep1_vs_rep2"]:
        n = san["noise_identity_rep1_vs_rep2"]
        print(f"  noise identity rep1 vs rep2: identical {n['identical']}/{n['cases']}; divergent {[(d['case'], d['first_div_token']) for d in n['divergent']]}")
    print("== shard", vh)
    print("== GATES")
    for g in res["gates"]:
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {g['item']}  [{g['detail']}]")


if __name__ == "__main__":
    main()
