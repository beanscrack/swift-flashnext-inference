#!/usr/bin/env python3
# Builds manifest.json from gen.jsonl, drops.jsonl, pool.jsonl and build_manifest.json (run with -I on datasci).
import collections, json, os, sys

P = "/u701/j692wu/swift-opt/mtp/p2"


def rows(name):
    p = os.path.join(P, name)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


gen = rows("gen.jsonl")
drops = rows("drops.jsonl")
pool = rows("pool.jsonl")
build = json.load(open(os.path.join(P, "build_manifest.json"), encoding="utf-8"))
progress = rows("progress.jsonl")
summary = [p for p in progress if p.get("done")]

by = collections.Counter()
tok = collections.defaultdict(lambda: collections.Counter())
fin = collections.defaultdict(collections.Counter)
lic = collections.defaultdict(set)
for r in gen:
    k = f'{r["source"]}|{r["mode"]}|{r["split"]}'
    by[k] += 1
    t = tok[k]
    t["prompt_tokens"] += r["prompt_tokens"]
    t["completion_tokens"] += r["completion_tokens"]
    t["total_tokens"] += r["total_tokens"]
    fin[f'{r["source"]}|{r["mode"]}'][str(r["finish_reason"])] += 1
    lic[r["source"]].add(r["license"])

drop_by = collections.Counter(f'{d["source"]}|{d["mode"]}' for d in drops)
done_ids = {r["id"] for r in gen}
drop_ids = {d["id"] for d in drops}
not_attempted = collections.Counter()
for p in pool:
    if p["id"] not in done_ids and p["id"] not in drop_ids:
        not_attempted[p["source"]] += 1

manifest = {
    "phase": "P2a self-distillation prompts: target greedy outputs",
    "target_model": "ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4 (served as 'swift'; --dtype bfloat16, NVFP4 weights)",
    "server_preset": "PRESET=final-code MAX_SEQS=8 CAPTURE_SIZES=1,2,3,4,5,10,15,20,25,30,35,40 (no FP8 hooks)",
    "sampling": {"temperature": 0, "nothink_max_tokens": 1536,
                 "think_max_tokens": "2048 - prompt_tokens", "total_cap": 2048, "max_prompt_tokens": 400,
                 "nothink_chat_template_kwargs": {"enable_thinking": False},
                 "think_chat_template_kwargs": None, "mode_rule": "think iff int(sha256('mode|'+prompt_sha),16) % 5 == 0",
                 "heldout_rule": "heldout iff int(sha256(prompt),16) % 100 < 5"},
    "token_id_source": "prompt_token_ids and completion_token_ids returned by /v1/chat/completions (return_token_ids=true); "
                       "prompt ids cross-checked against POST /tokenize (chat) for every record",
    "records": len(gen),
    "records_by_source_mode_split": dict(by),
    "records_by_source": dict(collections.Counter(r["source"] for r in gen)),
    "records_by_mode": dict(collections.Counter(r["mode"] for r in gen)),
    "records_by_split": dict(collections.Counter(r["split"] for r in gen)),
    "tokens_by_source_mode_split": {k: dict(v) for k, v in tok.items()},
    "tokens_total": {
        "prompt_tokens": sum(r["prompt_tokens"] for r in gen),
        "completion_tokens": sum(r["completion_tokens"] for r in gen),
        "total_tokens": sum(r["total_tokens"] for r in gen),
    },
    "finish_reasons_by_source_mode": {k: dict(v) for k, v in fin.items()},
    "finish_reasons_by_mode": {m: dict(collections.Counter(str(r["finish_reason"]) for r in gen if r["mode"] == m))
                               for m in ("nothink", "think")},
    "think_mode_empty_final_content": sum(1 for r in gen if r["mode"] == "think" and not (r["completion_text"] or "").strip()),
    "think_mode_empty_final_content_by_finish": dict(collections.Counter(
        str(r["finish_reason"]) for r in gen if r["mode"] == "think" and not (r["completion_text"] or "").strip())),
    "tokens_finish_stop_only": {
        "records": sum(1 for r in gen if r["finish_reason"] == "stop"),
        "prompt_tokens": sum(r["prompt_tokens"] for r in gen if r["finish_reason"] == "stop"),
        "completion_tokens": sum(r["completion_tokens"] for r in gen if r["finish_reason"] == "stop"),
        "total_tokens": sum(r["total_tokens"] for r in gen if r["finish_reason"] == "stop"),
        "by_mode": {m: {"records": sum(1 for r in gen if r["finish_reason"] == "stop" and r["mode"] == m),
                        "total_tokens": sum(r["total_tokens"] for r in gen if r["finish_reason"] == "stop" and r["mode"] == m)}
                    for m in ("nothink", "think")},
    },
    "licenses_by_source": {k: sorted(v) for k, v in lic.items()},
    "dropped": {
        "build_time_eval_and_duplicate_rules": build["drops_and_keeps_by_source"],
        "build_time_eval_sets": build["eval_sets_excluded"],
        "generation_time_prompt_gt_400_tokens": dict(drop_by),
        "pool_candidates_not_attempted": dict(not_attempted),
    },
    "prepared_cap": 4000,
    "throughput_summary": summary[-1] if summary else None,
}
with open(os.path.join(P, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print("wrote manifest.json records=", len(gen))
