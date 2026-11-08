#!/usr/bin/env python3
# Independent eval-leak check (verification a). Run: python -I leakcheck.py <gen.jsonl>
# Re-implements the rules from scratch (does not import build_pool.py).
#  rule1: sha256(normalized eval problem text) == sha256(normalized training problem text)
#  rule2: sha256(first 200 normalized chars) equal
#  rule3 (stricter): first 200 normalized chars of an eval problem occur inside the normalized training text
#  rule4: Codeforces problem id (contest, letter) shared with locked.jsonl (taskpack origin ids cc-N-X)
# Positive control: the same checker run on taskpack candidates.jsonl must flag the known locked overlap.
import hashlib, json, re, sys, unicodedata

TP = "/u701/j692wu/qwen/rft/taskpack"
GEN = sys.argv[1]


def nrm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return " ".join(s.split())


def h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


locked = rows(f"{TP}/locked.jsonl")
evalp = rows(f"{TP}/evalplus.jsonl")
eval_texts = [nrm(r["statement"]) for r in locked]
for r in evalp:
    body = r["prompt"].split("\n", 1)[1] if r["prompt"].startswith("Please provide") else r["prompt"]
    eval_texts.append(nrm(body.replace("```", "")))
E1 = {h(t) for t in eval_texts}
E2 = {h(t[:200]) for t in eval_texts}
E3 = [t[:200] for t in eval_texts if len(t) >= 60]
locked_cf = set()
for r in locked:
    m = re.match(r"cc-(\d+)-([A-Za-z]\d?)$", r["task_id"])
    if m:
        locked_cf.add((m.group(1), m.group(2).upper()))


def check(texts_with_cf):
    hits = {"rule1": [], "rule2": [], "rule3": [], "rule4": []}
    for idx, (text, cf) in enumerate(texts_with_cf):
        t = nrm(text)
        if h(t) in E1:
            hits["rule1"].append(idx)
        if h(t[:200]) in E2:
            hits["rule2"].append(idx)
        if any(p in t for p in E3):
            hits["rule3"].append(idx)
        if cf & locked_cf:
            hits["rule4"].append(idx)
    return {k: len(v) for k, v in hits.items()}, hits


# positive control: taskpack candidates (known to contain the 135 locked problems)
ctrl = []
for r in rows(f"{TP}/candidates.jsonl"):
    m = re.match(r"cc-(\d+)-([A-Za-z]\d?)$", r["task_id"])
    ctrl.append((r["statement"], {(m.group(1), m.group(2).upper())} if m else set()))
ctrl_counts, _ = check(ctrl)
print("POSITIVE CONTROL candidates.jsonl (expect rule1>0):", ctrl_counts, flush=True)

# gen.jsonl: check the full user message and the template-stripped problem statement
HEAD = "Solve the following competitive programming problem in Python 3.\n\n"
TAIL = "\n\nWrite a complete program that reads from standard input and writes to standard output. Put the final program in a single ```python code block."
gen = rows(GEN)
items = []
for g in gen:
    user = g["messages"][0]["content"]
    stmt = user[len(HEAD):-len(TAIL)] if user.startswith(HEAD) and user.endswith(TAIL) else user
    cf = set()
    m = re.match(r"cc-(\d+)-([A-Za-z]\d?)$", str(g["origin"]))
    if m:
        cf = {(m.group(1), m.group(2).upper())}
    items.append((stmt, cf))
    items.append((user, cf))
counts, hits = check(items)
print(f"GEN {GEN}: records={len(gen)} checked_texts={len(items)} matches={counts}", flush=True)
print("RESULT", json.dumps({"records": len(gen), "matches": counts, "control": ctrl_counts}), flush=True)
