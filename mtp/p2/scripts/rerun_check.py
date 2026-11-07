#!/usr/bin/env python3
# Determinism check (verification b). Run ON gpu-1-vm after the main run, server still up:
#   python -I rerun_check.py <gen.jsonl> <out.json>
# Re-generates 5 nothink + 5 think samples chosen with a fixed seed, using the stored request parameters, all
# 10 concurrently. Compares prompt_token_ids and completion_token_ids exactly with the stored run.
import concurrent.futures as cf, json, random, sys, urllib.request

URL = "http://127.0.0.1:8000"
GEN, OUT = sys.argv[1], sys.argv[2]
SEED = 20261107


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


recs = [json.loads(l) for l in open(GEN, encoding="utf-8") if l.strip()]
rng = random.Random(SEED)
nothink = [r for r in recs if r["mode"] == "nothink"]
think = [r for r in recs if r["mode"] == "think"]
sel = rng.sample(nothink, 5) + rng.sample(think, 5)


def rerun(r):
    body = {"model": "swift", "messages": r["messages"], "temperature": 0,
            "max_tokens": r["request"]["max_tokens"], "return_token_ids": True, "stream": False}
    if r["request"]["chat_template_kwargs"] is not None:
        body["chat_template_kwargs"] = r["request"]["chat_template_kwargs"]
    resp = post("/v1/chat/completions", body)
    ch = resp["choices"][0]
    new_ids = ch.get("token_ids") or []
    new_p = resp.get("prompt_token_ids")
    old_ids = r["completion_token_ids"]
    first_div = None
    for i, (a, b) in enumerate(zip(old_ids, new_ids)):
        if a != b:
            first_div = i
            break
    if first_div is None and len(old_ids) != len(new_ids):
        first_div = min(len(old_ids), len(new_ids))
    return {"id": r["id"], "mode": r["mode"], "old_len": len(old_ids), "new_len": len(new_ids),
            "prompt_ids_identical": new_p == r["prompt_token_ids"],
            "completion_ids_identical": new_ids == old_ids,
            "first_divergence_index": first_div,
            "old_finish": r["finish_reason"], "new_finish": ch.get("finish_reason")}


with cf.ThreadPoolExecutor(max_workers=10) as ex:
    results = list(ex.map(rerun, sel))
summary = {
    "seed": SEED, "n": len(results),
    "identical": sum(1 for x in results if x["completion_ids_identical"]),
    "mismatch": sum(1 for x in results if not x["completion_ids_identical"]),
    "mismatch_fraction": round(sum(1 for x in results if not x["completion_ids_identical"]) / len(results), 4),
    "prompt_mismatch": sum(1 for x in results if not x["prompt_ids_identical"]),
    "by_mode": {m: {"n": sum(1 for x in results if x["mode"] == m),
                    "mismatch": sum(1 for x in results if x["mode"] == m and not x["completion_ids_identical"])}
                for m in ("nothink", "think")},
    "results": results,
}
json.dump(summary, open(OUT, "w", encoding="utf-8"), indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
