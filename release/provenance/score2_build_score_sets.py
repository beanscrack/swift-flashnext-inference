"""Build two sets of teacher-forcing sequences from EXISTING generations: A = baseline (unpatched Marlin) outputs, B = final-stack outputs
(skinny + draft head + L2 prefetch).  Same prompts in both sets.  sequence = prompt ids + first N tokens of the generated text."""
import json, glob, os
from transformers import AutoTokenizer
N = 256
model = open("/u701/j692wu/swift-local/model-path.txt").read().strip()
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
R = "/u701/j692wu/swift-profile/runs"
def cases(path): return {c["label"]: c["prompt_ids"] for c in json.load(open(path))}
CODING = cases("/u701/j692wu/swift-local/experiments/20261021/decode-cases.json")
OOD1 = cases("/u701/j692wu/swift-profile/ood/decode-cases.json")
OOD2 = cases("/u701/j692wu/swift-profile/ood2/decode-cases.json")
SETS = {
 "A": [("m-base", "", CODING), ("m-base", "ood/", OOD1), ("p-base", "ood/", OOD2)],
 "B": [("m-final-code", "", CODING), ("m-final-multiD", "ood/", OOD1), ("p-final-multi", "ood/", OOD2)],
}
out = {}
for name, srcs in SETS.items():
    seqs = []
    for label, sub, cmap in srcs:
        for f in sorted(glob.glob(f"{R}/{label}/{sub}rep*/requests.json")):
            for r in json.load(open(f)):
                ids = cmap[r["case"]] + tok.encode(r["output"], add_special_tokens=False)[:N]
                seqs.append({"label": f"{label}/{sub}{os.path.basename(os.path.dirname(f))}/{r['case']}/{r['phase']}", "prompt_len": len(cmap[r["case"]]), "ids": ids})
    out[name] = seqs
    print(name, len(seqs), "sequences", sum(len(s["ids"]) - s["prompt_len"] for s in seqs), "scored tokens")
json.dump(out, open("/u701/j692wu/swift-profile/score2/score_sets.json", "w"))
