"""Online gate client (run ON gpu-1-vm against the BF16 release server on 127.0.0.1:8000).

1. GET /metrics (raw text) -> spec_decode drafts and accepted-per-position counters (before).
2. Send every heldout record of gen.jsonl, one at a time, with the SAME messages and the SAME
   request fields (temperature 0, max_tokens, return_token_ids, chat_template_kwargs) as gen.jsonl.
3. GET /metrics again (after); deltas; acc_j = accepted_per_pos[j] / drafts; tokens/cycle = 1 + sum acc_j.
Writes online_results.jsonl (per request) and online_gate.json (summary).
"""
import json
import re
import sys
import time
import urllib.request

sys.path.insert(0, "/u701/j692wu/swift-opt/mtp/p3")
from data import load_gen  # noqa: E402

URL = "http://127.0.0.1:8000"
OUT_DIR = "/u701/j692wu/swift-opt/mtp/p3/online"
MODEL = "swift"
K = 4

DRAFTS_RE = re.compile(r"^vllm:spec_decode_num_drafts_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M)
POS_RE = re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\}\s+([0-9.eE+-]+)\s*$', re.M)
ACC_RE = re.compile(r"^vllm:spec_decode_num_accepted_tokens_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M)
EMIT_RE = re.compile(r"^vllm:spec_decode_num_emitted_tokens_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M)


def get_metrics_text():
    with urllib.request.urlopen(URL + "/metrics", timeout=30) as r:
        return r.read().decode("utf-8")


def parse(text):
    drafts = sum(float(v) for v in DRAFTS_RE.findall(text))
    pos = {}
    for p, v in POS_RE.findall(text):
        pos[int(p)] = pos.get(int(p), 0.0) + float(v)
    acc = sum(float(v) for v in ACC_RE.findall(text))
    emit = sum(float(v) for v in EMIT_RE.findall(text))
    return {"drafts": drafts, "accepted_per_pos": [pos.get(j, 0.0) for j in range(K)],
            "accepted_total": acc, "emitted_total": emit}


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    before_text = get_metrics_text()
    with open(f"{OUT_DIR}/metrics_before.txt", "w") as f:
        f.write(before_text)
    before = parse(before_text)
    recs = load_gen("heldout")
    res_f = open(f"{OUT_DIR}/online_results.jsonl", "w")
    n_ok = n_err = n_prompt_mismatch = n_token_prefix_match = 0
    t0 = time.time()
    for i, rec in enumerate(recs):
        body = {"model": MODEL, "messages": rec["messages"], "stream": False}
        body.update(rec["request"])  # temperature, max_tokens, return_token_ids, [chat_template_kwargs]
        try:
            resp = post("/v1/chat/completions", body)
            ch = resp["choices"][0]
            out_ids = ch.get("token_ids") or []
            p_ids = resp.get("prompt_token_ids")
            ok = True
        except Exception as e:  # keep going; counted as error
            out_ids, p_ids, ch = [], None, {"finish_reason": "error:" + repr(e)[:200]}
            ok = False
            n_err += 1
        if ok:
            n_ok += 1
            if p_ids is not None and p_ids != rec["prompt_token_ids"]:
                n_prompt_mismatch += 1
            k = 0
            while k < min(len(out_ids), len(rec["completion_token_ids"])) and out_ids[k] == rec["completion_token_ids"][k]:
                k += 1
            if k == min(len(out_ids), len(rec["completion_token_ids"])):
                n_token_prefix_match += 1
        res_f.write(json.dumps({"id": rec["id"], "mode": rec["mode"], "ok": ok,
                                "n_out": len(out_ids), "finish_reason": ch.get("finish_reason"),
                                "out_ids": out_ids}) + "\n")
        res_f.flush()
        if i % 10 == 0:
            print(f"{i+1}/{len(recs)} {time.time()-t0:.0f}s ok={n_ok} err={n_err}", flush=True)
    res_f.close()
    after_text = get_metrics_text()
    with open(f"{OUT_DIR}/metrics_after.txt", "w") as f:
        f.write(after_text)
    after = parse(after_text)
    drafts = after["drafts"] - before["drafts"]
    per_pos = [after["accepted_per_pos"][j] - before["accepted_per_pos"][j] for j in range(K)]
    acc = [p / drafts for p in per_pos] if drafts else [float("nan")] * K
    summary = {
        "n_requests": len(recs), "ok": n_ok, "errors": n_err,
        "prompt_mismatch_vs_gen": n_prompt_mismatch,
        "requests_with_completion_identical_to_gen": n_token_prefix_match,
        "drafts": drafts, "accepted_per_pos": per_pos, "acc": acc,
        "tokens_per_cycle": 1 + sum(acc),
        "emitted_delta": after["emitted_total"] - before["emitted_total"],
        "accepted_total_delta": after["accepted_total"] - before["accepted_total"],
        "wall_seconds": time.time() - t0,
        "before": before, "after": after,
    }
    with open(f"{OUT_DIR}/online_gate.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k not in ("before", "after")}), flush=True)


if __name__ == "__main__":
    main()
