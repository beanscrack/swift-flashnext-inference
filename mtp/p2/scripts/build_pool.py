#!/usr/bin/env python3
# P2a prompt-pool builder. Run with: python -I build_pool.py <tokenizer_dir>
# Inputs : taskpack train.jsonl + candidates.jsonl (training), locked.jsonl + evalplus.jsonl (EXCLUDED eval sets),
#          prepared-v1 train/validation.jsonl (code_reasoning + opencode user turns, decoded from token ids).
# Output : pool.jsonl (ordered candidates that pass every text filter; prompt-length filter runs at generation time),
#          build_manifest.json (drop counts per rule and source).
import collections, hashlib, json, re, sys, unicodedata
from transformers import AutoTokenizer

P = "/u701/j692wu/swift-opt/mtp/p2"
TP = "/u701/j692wu/qwen/rft/taskpack"
PREP = "/u701/j692wu/qwen/data/prepared-v1"
TOK_DIR = sys.argv[1]

HEAD = "Solve the following competitive programming problem in Python 3.\n\n"
TAIL = ("\n\nWrite a complete program that reads from standard input and writes to standard output. "
        "Put the final program in a single ```python code block.")
LIC = {
    "taskpack": "CC BY 4.0 (deepmind/code_contests, Codeforces subset; taskpack/pack-manifest.json)",
    "code_reasoning": "CC BY 4.0 (nvidia/OpenCodeReasoning rev 20a1ca19c0d050fe9057fc08339d6b370ec1c67a; row tag cc-by-4.0, dataset=code_contests; qwen/DATA_ATTRIBUTION.md)",
    "opencode": "CC BY 4.0 (nvidia/Nemotron-SFT-OpenCode-v1 rev 556d5237acff203f3e1a0be49428634c3606cda2; dataset card; qwen/DATA_ATTRIBUTION.md)",
}


def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------- eval sets (excluded) ----------
locked = load(f"{TP}/locked.jsonl")
evalplus = load(f"{TP}/evalplus.jsonl")
BOILER = re.compile(r"^Please provide a self-contained Python script that solves the following problem in a markdown code block:\s*", re.S)
eval_items = []
for r in locked:
    eval_items.append(("locked:" + r["task_id"], norm(r["statement"])))
for r in evalplus:
    core = BOILER.sub("", r["prompt"]).replace("```python", " ").replace("```", " ")
    eval_items.append(("evalplus:" + str(r["task_id"]), norm(core)))
assert all(len(n) > 0 for _, n in eval_items)
H1 = {sha(n): lab for lab, n in eval_items}                  # rule 1: sha256(normalized problem text)
H2 = {sha(n[:200]): lab for lab, n in eval_items}            # rule 2: sha256(first 200 normalized chars)
PREFIXES = [(lab, n[:200]) for lab, n in eval_items if len(n) >= 60]  # rule 3 (stricter): eval prefix contained in candidate

CF_RE = re.compile(r"^cc-(\d+)-([A-Za-z]\d?)$")
NAME_RE = re.compile(r"^(\d+)_([A-Za-z]\d?)\.")


def cf_keys(r):
    keys = set()
    m = CF_RE.match(str(r.get("task_id", "")))
    if m:
        keys.add((m.group(1), m.group(2).upper()))
    m = NAME_RE.match(str(r.get("name", "")))
    if m:
        keys.add((m.group(1), m.group(2).upper()))
    return keys


EVAL_CF = set()
for r in locked:
    EVAL_CF |= cf_keys(r)
print(f"eval items: locked={len(locked)} evalplus={len(evalplus)} CF ids={len(EVAL_CF)} prefixes={len(PREFIXES)}", flush=True)

# ---------- candidate filters ----------
stats = collections.defaultdict(collections.Counter)  # source -> rule -> count
seen = set()


def filt(cand):
    """Return None if kept, else the name of the first rule that drops the candidate."""
    N = cand["norm"]
    if sha(N) in H1:
        return "eval_exact_hash"
    if sha(N[:200]) in H2:
        return "eval_prefix200_hash"
    for _, pre in PREFIXES:
        if pre in N:
            return "eval_prefix200_contained"
    if cand["cf"] & EVAL_CF:
        return "eval_cf_id"
    if N in seen:
        return "duplicate_text"
    seen.add(N)
    return None


pool = []
# ---------- taskpack training prompts ----------
tp_total = collections.Counter()
for fname, tag in [("train.jsonl", "taskpack_train"), ("candidates.jsonl", "taskpack_candidates")]:
    for r in load(f"{TP}/{fname}"):
        stmt = r["statement"]
        user = HEAD + stmt + TAIL
        tp_total["rows"] += 1
        if r.get("prompt") == user:
            tp_total["prompt_field_matches_template"] += 1
        cand = {"norm": norm(stmt), "cf": cf_keys(r), "user": user, "origin": r["task_id"], "src": "taskpack", "tag": tag}
        stats["taskpack"]["input_rows"] += 1
        drop = filt(cand)
        if drop:
            stats["taskpack"][drop] += 1
            continue
        stats["taskpack"]["kept"] += 1
        pool.append(cand)
print("taskpack:", dict(tp_total), flush=True)

# ---------- prepared-v1 user turns (decoded from token ids) ----------
tok = AutoTokenizer.from_pretrained(TOK_DIR)
SEG = re.compile(r"<\|im_start\|>(system|user|assistant|tool)\n(.*?)<\|im_end\|>", re.S)
prep = []
for split_file, tag in [("train.jsonl", "prepared-v1/train"), ("validation.jsonl", "prepared-v1/validation")]:
    with open(f"{PREP}/{split_file}", encoding="utf-8") as f:
        for line in f:
            m = re.match(r'\{"source": "(\w+)"', line)
            if not m or m.group(1) not in ("code_reasoning", "opencode"):
                continue
            src = m.group(1)
            stats[src]["input_rows"] += 1
            r = json.loads(line)
            lab = r["labels"]
            first = next(i for i, x in enumerate(lab) if x != -100)
            ptext = tok.decode(r["input_ids"][:first], skip_special_tokens=False)
            segs = SEG.findall(ptext)
            roles = [s[0] for s in segs]
            if roles.count("user") != 1 or roles.count("assistant") or roles.count("tool"):
                stats[src]["not_single_user_turn"] += 1
                continue
            user = [c for role, c in segs if role == "user"][0].strip()
            if len(user) < 20:
                stats[src]["user_too_short"] += 1
                continue
            prep.append({"src": src, "tag": tag, "origin": r["source_id"], "user": user, "norm": norm(user), "cf": set()})
print("prepared parsed:", {k: dict(v) for k, v in stats.items() if k in ("code_reasoning", "opencode")}, flush=True)

prep.sort(key=lambda c: sha(c["norm"]))  # deterministic pseudo-random order
for cand in prep:
    drop = filt(cand)
    if drop:
        stats[cand["src"]][drop] += 1
        continue
    stats[cand["src"]]["kept"] += 1
    pool.append(cand)

# ---------- write pool ----------
out = []
for c in pool:
    psha = sha(c["user"])
    out.append({
        "id": ("tp-" if c["src"] == "taskpack" else "pp-") + psha[:16],
        "source": c["src"],
        "origin": c["origin"],
        "tag": c["tag"],
        "license": LIC[c["src"]],
        "split": "heldout" if int(psha, 16) % 100 < 5 else "train",
        "mode": "think" if int(sha("mode|" + psha), 16) % 5 == 0 else "nothink",
        "messages": [{"role": "user", "content": c["user"]}],
        "prompt_sha256": psha,
        "norm_sha256": sha(c["norm"]),
    })
ids = [o["id"] for o in out]
assert len(ids) == len(set(ids)), "id collision"
with open(f"{P}/pool.jsonl", "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")

man = {
    "eval_sets_excluded": {"locked_rows": len(locked), "evalplus_rows": len(evalplus),
                           "eval_cf_ids": len(EVAL_CF), "rule1_hashes": len(H1), "rule2_hashes": len(H2),
                           "rule3_prefixes": len(PREFIXES)},
    "drops_and_keeps_by_source": {k: dict(v) for k, v in stats.items()},
    "pool_size_before_token_filter": len(out),
    "pool_by_source": dict(collections.Counter(o["source"] for o in out)),
    "pool_by_split": dict(collections.Counter(o["split"] for o in out)),
    "pool_by_mode": dict(collections.Counter(o["mode"] for o in out)),
}
with open(f"{P}/build_manifest.json", "w", encoding="utf-8") as f:
    json.dump(man, f, indent=2, ensure_ascii=False)
print(json.dumps(man, indent=2, ensure_ascii=False), flush=True)
print("BUILD DONE", flush=True)
