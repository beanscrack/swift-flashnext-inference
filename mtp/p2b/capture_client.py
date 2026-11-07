#!/usr/bin/env python3
"""P2b client: capture the target's pre-final-mixer hidden state for every gen.jsonl sequence.

Modes:
  main [--limit N]   one request per record (sequential). Writes /tmp/mtp_hidden/<id>.pt (hook),
                     /tmp/mtp_hidden/<id>.labels.pt (target argmax), capture_status.jsonl, failures.jsonl.
  rerun              20 random samples re-captured into /tmp/mtp_hidden_rerun (5 also get bf16 copies).
  verify             aggregate sidecars, save log, server log, disk -> manifest.json (+ verify_rerun.json).
Run with the inference python, detached, on gpu-1-vm.
"""
import argparse
import http.client
import json
import os
import random
import statistics
import sys
import time

import torch

SEQ_SRC = "/u701/j692wu/swift-opt/mtp/p2/gen.jsonl"
OUT = "/u701/j692wu/swift-opt/mtp/p2b"
HID = "/tmp/mtp_hidden"
RERUN = "/tmp/mtp_hidden_rerun"
ARM = os.path.join(HID, "ARM")
STATUS = os.path.join(OUT, "capture_status.jsonl")
FAILS = os.path.join(OUT, "failures.jsonl")
SERVER_LOG = os.path.join(OUT, "server.log")
HC_H = 10240
PORT = 8000


def log(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%S") + " " + msg, flush=True)


def load_records():
    recs = []
    with open(SEQ_SRC) as f:
        for line in f:
            r = json.loads(line)
            P = len(r["prompt_token_ids"])
            C = len(r["completion_token_ids"])
            recs.append({
                "id": r["id"], "split": r["split"], "mode": r["mode"],
                "seq": list(r["prompt_token_ids"]) + list(r["completion_token_ids"]),
                "P": P, "C": C, "gen": list(r["completion_token_ids"]),
            })
    return recs


def hidden_path(sid, d=HID):
    return os.path.join(d, sid + ".pt")


def labels_path(sid):
    return os.path.join(HID, sid + ".labels.pt")


def post(seq):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=900)
    body = json.dumps({"model": "swift", "prompt": seq, "max_tokens": 1, "temperature": 0,
                       "prompt_logprobs": 1, "stream": False})
    conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {data[:300]!r}")
    return json.loads(data)


def arm_and_post(sid, seq, outdir=None, bf16=False):
    """Write ARM, send the request, and return (response, consumed). The hook renames ARM to ARM.done on capture."""
    os.makedirs(HID, exist_ok=True)
    for p in (ARM, ARM + ".done"):
        if os.path.exists(p):
            os.remove(p)
    lines = [sid, outdir or "", "bf16" if bf16 else ""]
    while lines and lines[-1] == "":
        lines.pop()
    with open(ARM, "w") as f:
        f.write("\n".join(lines) + "\n")
    try:
        resp = post(seq)
        consumed = os.path.exists(ARM + ".done")
    finally:
        for p in (ARM, ARM + ".done"):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
    return resp, consumed


def labels_from_resp(resp, seq):
    """target_argmax[t] = rank-1 entry of prompt_logprobs[t+1]: the target's argmax prediction for token t+1."""
    T = len(seq)
    if resp["usage"]["prompt_tokens"] != T:
        raise ValueError(f"usage.prompt_tokens {resp['usage']['prompt_tokens']} != T {T}")
    pl = resp["choices"][0].get("prompt_logprobs")
    if pl is None or len(pl) != T:
        raise ValueError(f"prompt_logprobs length {None if pl is None else len(pl)} != T {T}")
    if pl[0] is not None:
        raise ValueError("prompt_logprobs[0] not null")
    am = []
    missing_actual = 0
    for t in range(T - 1):
        d = pl[t + 1]
        if d is None:
            raise ValueError(f"prompt_logprobs[{t + 1}] is null")
        if str(seq[t + 1]) not in d:
            missing_actual += 1
        ones = [int(k) for k, v in d.items() if v.get("rank") == 1]
        if len(ones) != 1:
            raise ValueError(f"position {t + 1}: {len(ones)} rank-1 entries")
        am.append(ones[0])
    return am, missing_actual


def check_hidden(rec, seq):
    T = len(seq)
    probs = []
    if rec.get("format") != "fp8_e4m3_rowscale_v1":
        probs.append("format")
    fp8 = rec["fp8"]
    if tuple(fp8.shape) != (T, HC_H) or fp8.dtype != torch.float8_e4m3fn:
        probs.append(f"fp8 shape/dtype {tuple(fp8.shape)} {fp8.dtype}")
    sc = rec["scale"]
    if tuple(sc.shape) != (T,) or sc.dtype != torch.float32 or not torch.isfinite(sc).all():
        probs.append("scale")
    if rec["input_ids"] is None or rec["input_ids"].tolist() != seq:
        probs.append("input_ids != sequence")
    pos = rec["positions"]
    if tuple(pos.shape) != (T,) or not torch.equal(pos.long(), torch.arange(T)):
        probs.append("positions != arange(T)")
    if rec.get("nonfinite", -1) != 0:
        probs.append(f"nonfinite={rec.get('nonfinite')}")
    if rec["shape"] != [T, HC_H]:
        probs.append(f"shape {rec['shape']}")
    if rec["buffer_shape"][1] != HC_H:
        probs.append(f"buffer_shape {rec['buffer_shape']}")
    return probs


def dequant(rec):
    return rec["fp8"].to(torch.float32) * rec["scale"][:, None]


def capture_one(rec):
    sid, seq = rec["id"], rec["seq"]
    T = len(seq)
    resp, consumed = arm_and_post(sid, seq)
    if not consumed:
        raise RuntimeError("hook did not capture (ARM not consumed)")
    am, missing_actual = labels_from_resp(resp, seq)
    h = torch.load(hidden_path(sid), map_location="cpu", weights_only=True)
    probs = check_hidden(h, seq)
    if probs:
        raise ValueError("hidden check failed: " + "; ".join(probs))
    tmp = labels_path(sid) + ".tmp"
    torch.save({
        "sample_id": sid, "split": rec["split"], "mode": rec["mode"],
        "T": T, "P": rec["P"], "C": rec["C"],
        "target_argmax": torch.tensor(am, dtype=torch.int32),
        "generated": torch.tensor(rec["gen"], dtype=torch.int32),
        "missing_actual_in_prompt_logprobs": missing_actual,
    }, tmp)
    os.replace(tmp, labels_path(sid))
    return T, missing_actual


def done_ids():
    done = set()
    if os.path.exists(STATUS):
        with open(STATUS) as f:
            for line in f:
                o = json.loads(line)
                if o.get("status") == "ok":
                    done.add(o["id"])
    return done


def cmd_main(limit):
    recs = load_records()
    if limit:
        recs = recs[:limit]
    done = done_ids()
    todo = [r for r in recs if not (r["id"] in done and os.path.exists(hidden_path(r["id"]))
                                    and os.path.exists(labels_path(r["id"])))]
    log(f"main: records={len(recs)} todo={len(todo)} already_done={len(recs) - len(todo)}")
    n_ok = n_fail = tokens = 0
    t0 = time.time()
    with open(STATUS, "a") as st:
        for rec in todo:
            err = None
            for attempt in (1, 2):
                try:
                    T, miss = capture_one(rec)
                    st.write(json.dumps({"id": rec["id"], "status": "ok", "attempt": attempt, "T": T,
                                         "missing_actual": miss, "time": time.time()}) + "\n")
                    st.flush()
                    tokens += T
                    n_ok += 1
                    err = None
                    break
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    st.write(json.dumps({"id": rec["id"], "status": "error", "attempt": attempt,
                                         "error": err, "time": time.time()}) + "\n")
                    st.flush()
                    log(f"  error {rec['id']} attempt {attempt}: {err}")
            if err is not None:
                n_fail += 1
                with open(FAILS, "a") as f:
                    f.write(json.dumps({"id": rec["id"], "error": err}) + "\n")
            if (n_ok + n_fail) % 100 == 0:
                el = time.time() - t0
                log(f"  progress ok={n_ok} fail={n_fail} tokens={tokens} "
                    f"elapsed={el:.0f}s tok/s={tokens / max(el, 1e-9):.1f}")
    el = time.time() - t0
    log(f"main done: ok={n_ok} fail={n_fail} tokens={tokens} elapsed={el:.0f}s tok/s={tokens / max(el, 1e-9):.1f}")


def cmd_rerun():
    os.makedirs(RERUN, exist_ok=True)
    recs = {r["id"]: r for r in load_records()}
    ok_ids = sorted(done_ids())
    sample = random.Random(20261107).sample(ok_ids, 20)
    per = []
    for i, sid in enumerate(sample):
        rec = recs[sid]
        seq = rec["seq"]
        bf = i < 5
        resp, consumed = arm_and_post(sid, seq, outdir=RERUN, bf16=bf)
        if not consumed:
            raise RuntimeError(f"rerun {sid}: not captured")
        am_new, _ = labels_from_resp(resp, seq)
        am_old = torch.load(labels_path(sid), weights_only=True)["target_argmax"].tolist()
        new = torch.load(hidden_path(sid, RERUN), map_location="cpu", weights_only=True)
        old = torch.load(hidden_path(sid), map_location="cpu", weights_only=True)
        probs = check_hidden(new, seq)
        if probs:
            raise ValueError(f"rerun {sid}: " + "; ".join(probs))
        dn, do = dequant(new), dequant(old)
        cos = torch.nn.functional.cosine_similarity(dn, do, dim=1)
        entry = {
            "id": sid, "T": len(seq), "cos_median": float(cos.median()), "cos_min": float(cos.min()),
            "cos_rows": cos.tolist(),
            "fp8_bytes_identical_rows": float((new["fp8"].view(torch.uint8) == old["fp8"].view(torch.uint8))
                                              .all(dim=1).float().mean()),
            "labels_identical": am_new == am_old,
        }
        if bf:
            ref = torch.load(hidden_path(sid, RERUN).replace(".pt", ".bf16.pt"), map_location="cpu",
                             weights_only=True)["bf16"].float()
            err = (dn - ref).norm(dim=1) / ref.norm(dim=1).clamp(min=1e-30)
            entry["fp8_vs_bf16_rel_err_rows"] = err.tolist()
            entry["fp8_vs_bf16_rel_err_median"] = float(err.median())
            entry["fp8_vs_bf16_rel_err_max"] = float(err.max())
        per.append(entry)
        log(f"rerun {i + 1}/20 {sid} T={len(seq)} cos_median={entry['cos_median']:.6f} "
            f"identical_fp8_rows={entry['fp8_bytes_identical_rows']:.3f} labels_identical={entry['labels_identical']}")
    allcos = [c for e in per for c in e["cos_rows"]]
    bf_err = [x for e in per if "fp8_vs_bf16_rel_err_rows" in e for x in e["fp8_vs_bf16_rel_err_rows"]]
    summary = {
        "n_samples": len(per),
        "n_rows": len(allcos),
        "cos_median": statistics.median(allcos),
        "cos_min": min(allcos),
        "frac_cos_lt_0_999": sum(1 for c in allcos if c < 0.999) / len(allcos),
        "fp8_bytes_identical_rows_mean": statistics.mean(e["fp8_bytes_identical_rows"] for e in per),
        "labels_identical_samples": sum(1 for e in per if e["labels_identical"]),
        "bf16_samples": [e["id"] for e in per if "fp8_vs_bf16_rel_err_rows" in e],
        "fp8_vs_bf16_rel_err_row_median": statistics.median(bf_err) if bf_err else None,
        "fp8_vs_bf16_rel_err_row_max": max(bf_err) if bf_err else None,
        "fp8_vs_bf16_per_sample": [
            {"id": e["id"], "median": e["fp8_vs_bf16_rel_err_median"], "max": e["fp8_vs_bf16_rel_err_max"]}
            for e in per if "fp8_vs_bf16_rel_err_rows" in e],
    }
    for e in per:
        e.pop("cos_rows")
        e.pop("fp8_vs_bf16_rel_err_rows", None)
    with open(os.path.join(OUT, "verify_rerun.json"), "w") as f:
        json.dump({"summary": summary, "samples": per}, f, indent=1)
    log("rerun summary " + json.dumps({k: v for k, v in summary.items() if k != "fp8_vs_bf16_per_sample"}))


def cmd_verify():
    recs = {r["id"]: r for r in load_records()}
    ids = sorted(done_ids())
    stats = {}
    tot = {"samples": 0, "rows": 0, "prompt_tokens": 0, "completion_tokens": 0, "bytes_hidden": 0,
           "labels_missing": 0}
    cmp_all = {"n": 0, "match": 0}
    by = {}
    missing_actual = 0
    for sid in ids:
        r = recs[sid]
        lab = torch.load(labels_path(sid), weights_only=True)
        am = lab["target_argmax"].tolist()
        seq = r["seq"]
        T, P, C = len(seq), r["P"], r["C"]
        if len(am) != T - 1 or lab["T"] != T:
            tot["labels_missing"] += 1
            continue
        key = f"{r['split']}/{r['mode']}"
        b = by.setdefault(key, {"positions": 0, "match": 0, "samples": 0, "completion_tokens": 0,
                                "prompt_tokens": 0})
        for t in range(P - 1, T - 1):  # predicts completion tokens seq[P..T-1]
            b["positions"] += 1
            b["match"] += int(am[t] == seq[t + 1])
        b["samples"] += 1
        b["completion_tokens"] += C
        b["prompt_tokens"] += P
        tot["samples"] += 1
        tot["rows"] += T
        tot["prompt_tokens"] += P
        tot["completion_tokens"] += C
        tot["bytes_hidden"] += os.path.getsize(hidden_path(sid))
        missing_actual += lab["missing_actual_in_prompt_logprobs"]
    agree = {}
    for k, b in by.items():
        agree[k] = {"positions": b["positions"], "match": b["match"],
                    "rate": b["match"] / b["positions"] if b["positions"] else None,
                    "samples": b["samples"], "prompt_tokens": b["prompt_tokens"],
                    "completion_tokens": b["completion_tokens"]}
    P_all = sum(v["positions"] for v in by.values())
    M_all = sum(v["match"] for v in by.values())
    # hook isolation: saves in the main run must equal requests (one save per request, none from MTP forwards)
    attempts = saves = errors = 0
    save_counts = {}
    sl = os.path.join(HID, "saves.log")
    if os.path.exists(sl):
        with open(sl) as f:
            for line in f:
                o = json.loads(line)
                if o["event"] == "save" and o["outdir"] == HID:
                    save_counts[o["sample_id"]] = save_counts.get(o["sample_id"], 0) + 1
                    saves += 1
                elif o["event"] == "error":
                    errors += 1
    with open(STATUS) as f:
        attempts = sum(1 for _ in f)
    # server log
    server = {}
    if os.path.exists(SERVER_LOG):
        with open(SERVER_LOG, errors="replace") as f:
            txt = f.read()
        server = {
            "tracebacks": txt.count("Traceback (most recent call last)"),
            "draft_vocab_active": "[swift-draft-vocab] ACTIVE" in txt,
            "skinny_patched": "[swift_skinny] patched" in txt,
            "hidden_capture_patched": "[swift-hidden-capture] patched" in txt,
        }
    # disk
    du = 0
    nfiles = 0
    for name in os.listdir(HID):
        if name.endswith(".pt") and not name.endswith(".labels.pt"):
            du += os.path.getsize(os.path.join(HID, name))
            nfiles += 1
    st = os.statvfs(HID)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    with open(os.path.join(OUT, "verify_rerun.json")) as f:
        rerun = json.load(f)["summary"]
    failures = 0
    if os.path.exists(FAILS):
        with open(FAILS) as f:
            failures = sum(1 for _ in f)
    manifest = {
        "phase": "P2b",
        "source": SEQ_SRC,
        "records_in_source": len(recs),
        "captured_samples": tot["samples"],
        "failures_logged": failures,
        "rows_total": tot["rows"],
        "prompt_tokens_total": tot["prompt_tokens"],
        "completion_tokens_total": tot["completion_tokens"],
        "hidden_files": nfiles,
        "hidden_bytes_total": du,
        "labels_missing_or_mismatch": tot["labels_missing"],
        "per_split_mode": {k: {kk: vv for kk, vv in v.items()} for k, v in by.items()},
        "token_totals_by_split_mode": {k: {"samples": v["samples"], "prompt_tokens": v["prompt_tokens"],
                                           "completion_tokens": v["completion_tokens"], "rows": v["prompt_tokens"] + v["completion_tokens"]}
                                       for k, v in by.items()},
        "labels": {
            "argmax_agreement_completion_positions": agree,
            "argmax_agreement_overall": {"positions": P_all, "match": M_all,
                                         "rate": M_all / P_all if P_all else None},
            "actual_token_absent_from_prompt_logprobs": missing_actual,
            "every_position_exactly_one_rank1_entry": True,  # enforced per-sample in capture_one
        },
        "hook_isolation": {
            "requests_in_status_log": attempts,
            "saves_to_capture_dir_logged": saves,
            "distinct_samples_saved": len(save_counts),
            "max_saves_per_sample": max(save_counts.values()) if save_counts else 0,
            "hook_errors": errors,
            "server_log": server,
        },
        "rerun_verification": rerun,
        "disk": {"hidden_dir": HID, "free_GB_after": round(free_gb, 1)},
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    log("manifest written: " + json.dumps({k: manifest[k] for k in
                                             ["captured_samples", "failures_logged", "rows_total", "hidden_bytes_total"]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["main", "rerun", "verify", "smoke"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(HID, exist_ok=True)
    if a.mode == "main":
        cmd_main(a.limit)
    elif a.mode == "rerun":
        cmd_rerun()
    elif a.mode == "verify":
        cmd_verify()
    else:
        recs = load_records()[:2]
        for r in recs:
            resp, consumed = arm_and_post(r["id"], r["seq"])
            pl = resp["choices"][0]["prompt_logprobs"]
            log(f"smoke {r['id']} T={len(r['seq'])} consumed={consumed} pl[1]={json.dumps(pl[1])[:300]}")
            am, miss = labels_from_resp(resp, r["seq"])
            log(f"smoke labels ok n={len(am)} missing_actual={miss} first={am[:5]} gen_next={r['gen'][:5]}")
            log(f"smoke hidden probs={check_hidden(torch.load(hidden_path(r['id']), weights_only=True), r['seq'])}")


if __name__ == "__main__":
    main()
