"""Teacher-forced scoring of the A/B sets against a live server; records per token: logprob of the actual token, whether it is rank 1,
and the logprob gap to the rank-1 token.  Usage: score_sets.py OUT.json"""
import json, sys, urllib.request
sets = json.load(open("/u701/j692wu/swift-profile/score2/score_sets.json"))
res = {}
for name, seqs in sets.items():
    rows = []
    for c in seqs:
        body = {"model": "swift", "prompt": c["ids"], "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}
        req = urllib.request.Request("http://127.0.0.1:8000/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r: d = json.loads(r.read().decode())
        pl = d["choices"][0]["prompt_logprobs"]
        lp, rank1, gap = [], [], []
        for pos in range(c["prompt_len"], len(c["ids"])):
            ent = pl[pos]; tid = str(c["ids"][pos]); mine = ent[tid]["logprob"]
            top = max(e["logprob"] for e in ent.values())
            lp.append(mine); rank1.append(ent[tid].get("rank", 0) == 1); gap.append(top - mine)
        rows.append({"label": c["label"], "lp": lp, "rank1": rank1, "gap": gap})
    res[name] = rows
    n = sum(len(r["lp"]) for r in rows)
    print(name, "scored", n, "tokens | mean logprob %.4f | rank-1 fraction %.4f" % (sum(sum(r["lp"]) for r in rows) / n, sum(sum(r["rank1"]) for r in rows) / n), flush=True)
json.dump(res, open(sys.argv[1], "w"))
