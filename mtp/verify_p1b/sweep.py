import json, glob, os
import torch
from safetensors import safe_open
torch.manual_seed(1)
D = "/u701/j692wu/swift-opt/runs/p1-parity-v2/captures"
recs = json.load(open("/u701/j692wu/swift-opt/mtp/verify_p1b/recs.json"))
files = sorted(glob.glob(os.path.join(D, "call_*.pt")))
calls = [torch.load(f, map_location="cpu") for f in files]
with safe_open("/u701/j692wu/swift-opt/mtp/view-identity/model-tail.safetensors", framework="pt") as f:
    lmf = f.get_tensor("lm_head.weight").float()
S = [torch.cat([c["sample_hidden"][:recs[i]["nv"]].float() for i, c in enumerate(calls)])]
H = S[0]
ref = (H @ lmf.t()).argmax(-1)
print("rows", H.shape[0])
for rel in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
    accs = []
    for r in range(5):
        n = torch.randn_like(H)
        hp = H + n * (rel * H.norm(dim=-1, keepdim=True) / n.norm(dim=-1, keepdim=True))
        accs.append(float(((hp @ lmf.t()).argmax(-1) == ref).float().mean()))
    cos = float(torch.nn.functional.cosine_similarity(hp, H, dim=-1).mean())
    print("noise rel %.2f  (cos~%.4f)  argmax agreement mean %.4f min %.4f" % (rel, cos, sum(accs)/len(accs), min(accs)))
