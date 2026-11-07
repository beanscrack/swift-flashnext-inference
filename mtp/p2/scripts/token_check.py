#!/usr/bin/env python3
# Token accounting check (verification c). Run ON gpu-1-vm, server up:
#   python -I token_check.py <gen.jsonl> <out.json>
# For 20 seeded random samples:
#  1. re-tokenize the chat messages via /tokenize -> must equal prompt_tokens (and the server's prompt_token_ids);
#  2. detokenize prompt ids and completion ids, concatenate, re-tokenize the text (no special tokens) and compare
#     the count with total_tokens = prompt_tokens + completion_tokens (round-trip check, reported separately);
#  3. usage.prompt_tokens / usage.completion_tokens / total <= 2048.
import json, random, sys, urllib.request

URL = "http://127.0.0.1:8000"
GEN, OUT = sys.argv[1], sys.argv[2]
SEED = 20261108


def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


recs = [json.loads(l) for l in open(GEN, encoding="utf-8") if l.strip()]
sel = random.Random(SEED).sample(recs, min(20, len(recs)))
res = []
for r in sel:
    tb = {"model": "swift", "messages": r["messages"], "add_generation_prompt": True}
    if r["mode"] == "nothink":
        tb["chat_template_kwargs"] = {"enable_thinking": False}
    t_prompt = post("/tokenize", tb)["tokens"]
    detok_p = post("/detokenize", {"model": "swift", "tokens": r["prompt_token_ids"]})
    detok_c = post("/detokenize", {"model": "swift", "tokens": r["completion_token_ids"]})
    tp = detok_p.get("prompt", detok_p.get("text"))
    tc = detok_c.get("prompt", detok_c.get("text"))
    retok = post("/tokenize", {"model": "swift", "prompt": tp + tc, "add_special_tokens": False})["count"]
    u = r["usage"]
    res.append({
        "id": r["id"], "mode": r["mode"], "finish_reason": r["finish_reason"],
        "prompt_tokens": r["prompt_tokens"], "retok_prompt_matches": len(t_prompt) == r["prompt_tokens"],
        "prompt_ids_equal_server": t_prompt == r["prompt_token_ids"],
        "usage_prompt": u["prompt_tokens"], "usage_completion": u["completion_tokens"],
        "usage_total": u["total_tokens"],
        "usage_consistent": u["prompt_tokens"] == len(r["prompt_token_ids"]) and u["completion_tokens"] == len(r["completion_token_ids"]) and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"],
        "total_le_2048": u["total_tokens"] <= 2048,
        "completion_tokens": r["completion_tokens"], "retok_prompt_plus_completion": retok,
        "retok_equals_total": retok == u["total_tokens"],
        "retok_delta": retok - u["total_tokens"],
    })
summary = {
    "seed": SEED, "n": len(res),
    "retok_prompt_matches": sum(x["retok_prompt_matches"] for x in res),
    "prompt_ids_equal_server": sum(x["prompt_ids_equal_server"] for x in res),
    "usage_consistent": sum(x["usage_consistent"] for x in res),
    "total_le_2048": sum(x["total_le_2048"] for x in res),
    "retok_equals_total": sum(x["retok_equals_total"] for x in res),
    "retok_delta_values": [x["retok_delta"] for x in res],
    "samples": res,
}
json.dump(summary, open(OUT, "w", encoding="utf-8"), indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "samples"}), flush=True)
