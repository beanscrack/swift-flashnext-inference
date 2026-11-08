#!/usr/bin/env python3
# Verification (d) + record invariants. Re-derives every manifest count directly from gen.jsonl and compares.
# Run: python -I verify_manifest.py  (datasci)
import collections, hashlib, json

P = "/u701/j692wu/swift-opt/mtp/p2"
gen = [json.loads(l) for l in open(f"{P}/gen.jsonl", encoding="utf-8") if l.strip()]
man = json.load(open(f"{P}/manifest.json", encoding="utf-8"))
drops = [json.loads(l) for l in open(f"{P}/drops.jsonl", encoding="utf-8") if l.strip()]
sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()

problems = []
ids = [r["id"] for r in gen]
if len(ids) != len(set(ids)):
    problems.append("duplicate ids in gen.jsonl")
for r in gen:
    if r["prompt_tokens"] > 400:
        problems.append(f'{r["id"]} prompt>400')
    if len(r["prompt_token_ids"]) != r["prompt_tokens"]:
        problems.append(f'{r["id"]} prompt id length mismatch')
    if len(r["completion_token_ids"]) != r["completion_tokens"]:
        problems.append(f'{r["id"]} completion id length mismatch')
    if r["prompt_tokens"] + r["completion_tokens"] != r["total_tokens"] or r["total_tokens"] > 2048:
        problems.append(f'{r["id"]} total accounting')
    if not r["prompt_ids_match_tokenize"]:
        problems.append(f'{r["id"]} prompt ids != tokenize')
    if not r["usage_matches_ids"]:
        problems.append(f'{r["id"]} usage != ids')
    psha = sha(r["messages"][0]["content"])
    want_split = "heldout" if int(psha, 16) % 100 < 5 else "train"
    want_mode = "think" if int(sha("mode|" + psha), 16) % 5 == 0 else "nothink"
    if r["split"] != want_split:
        problems.append(f'{r["id"]} split rule')
    if r["mode"] != want_mode:
        problems.append(f'{r["id"]} mode rule')
    if r["mode"] == "nothink" and r["request"]["chat_template_kwargs"] != {"enable_thinking": False}:
        problems.append(f'{r["id"]} nothink kwargs')
    if r["mode"] == "think" and r["request"]["chat_template_kwargs"] is not None:
        problems.append(f'{r["id"]} think kwargs')

by = collections.Counter(f'{r["source"]}|{r["mode"]}|{r["split"]}' for r in gen)
tok = collections.defaultdict(collections.Counter)
for r in gen:
    k = f'{r["source"]}|{r["mode"]}|{r["split"]}'
    tok[k]["prompt_tokens"] += r["prompt_tokens"]
    tok[k]["completion_tokens"] += r["completion_tokens"]
    tok[k]["total_tokens"] += r["total_tokens"]

if dict(by) != man["records_by_source_mode_split"]:
    problems.append("records_by_source_mode_split mismatch")
for k, v in tok.items():
    if dict(v) != man["tokens_by_source_mode_split"].get(k):
        problems.append(f"token sums mismatch {k}")
tot = {
    "prompt_tokens": sum(r["prompt_tokens"] for r in gen),
    "completion_tokens": sum(r["completion_tokens"] for r in gen),
    "total_tokens": sum(r["total_tokens"] for r in gen),
}
if tot != man["tokens_total"]:
    problems.append("tokens_total mismatch")
if len(gen) != man["records"]:
    problems.append("records mismatch")
drop_counts = collections.Counter(f'{d["source"]}|{d["mode"]}' for d in drops)
if dict(drop_counts) != man["dropped"]["generation_time_prompt_gt_400_tokens"]:
    problems.append("drop counts mismatch")
prep = [r for r in gen if r["source"] in ("code_reasoning", "opencode")]
if len(prep) > man["prepared_cap"]:
    problems.append("prepared cap exceeded")
mode_share = {m: sum(1 for r in gen if r["mode"] == m) / len(gen) for m in ("nothink", "think")}
print(json.dumps({
    "records": len(gen), "tokens_total": tot, "prepared_records": len(prep),
    "mode_share": {k: round(v, 4) for k, v in mode_share.items()},
    "heldout_share": round(sum(1 for r in gen if r["split"] == "heldout") / len(gen), 4),
    "problems_found": len(problems), "problems_first10": problems[:10],
}, indent=2))
