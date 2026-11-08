#!/usr/bin/env python3
# P2a generation client. Run ON gpu-1-vm:  python -I gen.py --pool pool.jsonl --out gen.jsonl --deadline-s 18000
# Greedy (temperature 0) chat completions against the local release server (model name "swift").
# Per prompt: POST /tokenize (chat, same template kwargs as generation) -> prompt length check (<=400 tokens)
# -> POST /v1/chat/completions with return_token_ids=true (server returns prompt_token_ids + completion token_ids).
# Resumable: ids already present in --out are skipped.
import argparse, collections, concurrent.futures as cf, datetime, json, os, sys, time, urllib.request

URL = "http://127.0.0.1:8000"
MODEL = "swift"
MAX_PROMPT = 400          # prompt alone must be <= 400 tokens
TOTAL_CAP = 2048          # prompt + completion <= 2048
NOTHINK_MAX_TOKENS = 1536 # nothink: max_tokens 1536 (400 + 1536 <= 2048 always)
PREPARED_CAP = 4000       # prepared-* prompts accepted (after the 400-token filter)

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--drops", required=True)
ap.add_argument("--errors", required=True)
ap.add_argument("--deadline-s", type=float, required=True)
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--prepared-cap", type=int, default=PREPARED_CAP)
ap.add_argument("--progress", required=True)
args = ap.parse_args()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def kw(mode):
    # nothink: enable_thinking=false; think: no chat_template_kwargs (template default)
    return {"enable_thinking": False} if mode == "nothink" else None


def tokenize(rec):
    body = {"model": MODEL, "messages": rec["messages"], "add_generation_prompt": True}
    if kw(rec["mode"]):
        body["chat_template_kwargs"] = kw(rec["mode"])
    return post("/tokenize", body)["tokens"]


def generate(rec, ptoks):
    n = len(ptoks)
    mt = NOTHINK_MAX_TOKENS if rec["mode"] == "nothink" else TOTAL_CAP - n
    body = {"model": MODEL, "messages": rec["messages"], "temperature": 0, "max_tokens": mt,
            "return_token_ids": True, "stream": False}
    if kw(rec["mode"]):
        body["chat_template_kwargs"] = kw(rec["mode"])
    t0 = time.time()
    resp = post("/v1/chat/completions", body)
    dt = time.time() - t0
    ch = resp["choices"][0]
    msg = ch["message"]
    u = resp["usage"]
    out_ids = ch.get("token_ids") or []
    p_ids = resp.get("prompt_token_ids")
    reasoning = msg.get("reasoning") if msg.get("reasoning") is not None else msg.get("reasoning_content")
    rec_out = {
        "id": rec["id"], "source": rec["source"], "origin": rec["origin"], "tag": rec["tag"],
        "license": rec["license"], "split": rec["split"], "mode": rec["mode"],
        "messages": rec["messages"],
        "prompt_token_ids": p_ids, "prompt_tokens": n,
        "prompt_ids_match_tokenize": p_ids == ptoks,
        "completion_text": msg.get("content"), "reasoning_text": reasoning,
        "completion_token_ids": out_ids, "completion_tokens": len(out_ids),
        "finish_reason": ch.get("finish_reason"), "stop_reason": ch.get("stop_reason"),
        "total_tokens": n + len(out_ids), "max_tokens": mt,
        "request": {"temperature": 0, "max_tokens": mt, "return_token_ids": True,
                    "chat_template_kwargs": kw(rec["mode"])},
        "usage": u, "usage_matches_ids": (u["prompt_tokens"] == n and u["completion_tokens"] == len(out_ids)),
        "within_2048": (n + len(out_ids)) <= TOTAL_CAP,
        "gen_seconds": round(dt, 3), "generated_at": now(),
    }
    return rec_out


def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = r["source"]
    return done


pool = [json.loads(l) for l in open(args.pool, encoding="utf-8") if l.strip()]
done = load_done(args.out)
dropped_ids = set()
if os.path.exists(args.drops):
    with open(args.drops, encoding="utf-8") as f:
        dropped_ids = {json.loads(l)["id"] for l in f if l.strip()}
prep_accepted = sum(1 for s in done.values() if s in ("code_reasoning", "opencode"))
print(f"[{now()}] pool={len(pool)} already_done={len(done)} prepared_accepted_so_far={prep_accepted}", flush=True)

out_f = open(args.out, "a", encoding="utf-8")
drop_f = open(args.drops, "a", encoding="utf-8")
err_f = open(args.errors, "a", encoding="utf-8")
prog_f = open(args.progress, "a", encoding="utf-8")

stats = collections.Counter()
agg_completion = 0
agg_prompt = 0
t_start = time.time()
deadline = t_start + args.deadline_s
ex = cf.ThreadPoolExecutor(max_workers=args.concurrency)
inflight = {}


def handle(fut, rec):
    global agg_completion, agg_prompt
    try:
        r = fut.result()
    except Exception as e:  # network / server error: logged, id not written (rerun on resume)
        stats["error"] += 1
        err_f.write(json.dumps({"id": rec["id"], "source": rec["source"], "mode": rec["mode"],
                                "error": repr(e)[:500], "at": now()}) + "\n")
        err_f.flush()
        return
    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_f.flush()
    stats["ok"] += 1
    stats["ok_" + r["mode"]] += 1
    stats["finish_" + str(r["finish_reason"])] += 1
    if not r["usage_matches_ids"] or not r["prompt_ids_match_tokenize"] or not r["within_2048"]:
        stats["CHECK_FAILED"] += 1
    agg_completion += r["completion_tokens"]
    agg_prompt += r["prompt_tokens"]
    if stats["ok"] % 50 == 0:
        el = time.time() - t_start
        line = {"at": now(), "ok": stats["ok"], "elapsed_s": round(el, 1),
                "completion_tokens": agg_completion, "prompt_tokens": agg_prompt,
                "completion_tok_per_s": round(agg_completion / el, 2), "errors": stats["error"]}
        prog_f.write(json.dumps(line) + "\n")
        prog_f.flush()
        print(line, flush=True)


n_submitted = 0
for rec in pool:
    if time.time() > deadline:
        stats["deadline_reached_stop_submitting"] = 1
        break
    rid = rec["id"]
    if rid in done:
        continue
    is_prep = rec["source"] in ("code_reasoning", "opencode")
    if is_prep and prep_accepted >= args.prepared_cap:
        continue
    if rid in dropped_ids:
        continue
    try:
        ptoks = tokenize(rec)
    except Exception as e:
        stats["tokenize_error"] += 1
        err_f.write(json.dumps({"id": rid, "error": "tokenize " + repr(e)[:400], "at": now()}) + "\n")
        err_f.flush()
        continue
    if len(ptoks) > MAX_PROMPT:
        stats["drop_prompt_gt400_" + rec["source"]] += 1
        drop_f.write(json.dumps({"id": rid, "source": rec["source"], "mode": rec["mode"],
                                 "prompt_tokens": len(ptoks), "reason": "prompt_gt_400"}) + "\n")
        drop_f.flush()
        continue
    if is_prep:
        prep_accepted += 1
    # bounded in-flight window
    while len(inflight) >= args.concurrency:
        finished, _ = cf.wait(list(inflight.keys()), return_when=cf.FIRST_COMPLETED)
        for f in finished:
            handle(f, inflight.pop(f))
    fut = ex.submit(generate, rec, ptoks)
    inflight[fut] = rec
    n_submitted += 1

while inflight:
    finished, _ = cf.wait(list(inflight.keys()), return_when=cf.FIRST_COMPLETED)
    for f in finished:
        handle(f, inflight.pop(f))

el = time.time() - t_start
summary = {"at": now(), "done": True, "elapsed_s": round(el, 1), "submitted_this_run": n_submitted,
           "completion_tokens_this_run": agg_completion, "prompt_tokens_this_run": agg_prompt,
           "completion_tok_per_s": round(agg_completion / el, 2) if el > 0 else 0, "stats": dict(stats),
           "prepared_accepted_total": prep_accepted}
prog_f.write(json.dumps(summary) + "\n")
prog_f.flush()
print(json.dumps(summary), flush=True)
ex.shutdown(wait=True)
