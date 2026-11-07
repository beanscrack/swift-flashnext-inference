import json, struct, sys, hashlib
import torch
from safetensors import safe_open
ORIG='/u701/j692wu/swift-opt/mtp/mtp_original.safetensors'
CAND='/u701/j692wu/swift-opt/mtp/p3b/ckpt/best.safetensors'
def header(p):
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        return json.loads(f.read(n))
ho=header(ORIG); hc=header(CAND)
ho.pop('__metadata__',None); hc.pop('__metadata__',None)
print('orig names',len(ho),'cand names',len(hc))
print('same name set', set(ho)==set(hc))
print('names with embed/lm_head:', [k for k in hc if 'embed' in k or 'lm_head' in k], [k for k in ho if 'embed' in k or 'lm_head' in k])
shape_ok = all(ho[k]['shape']==hc[k]['shape'] for k in ho)
dtype_ok = all(ho[k]['dtype']==hc[k]['dtype'] for k in ho)
print('shapes equal', shape_ok, 'dtypes equal', dtype_ok, set(v['dtype'] for v in ho.values()), set(v['dtype'] for v in hc.values()))
rows=[]
with safe_open(ORIG,'pt',device='cpu') as fo, safe_open(CAND,'pt',device='cpu') as fc:
    for k in sorted(ho):
        a=fo.get_tensor(k).float(); b=fc.get_tensor(k).float()
        d=(b-a).norm().item(); n=a.norm().item()
        bitwise = torch.equal(fo.get_tensor(k).view(torch.int16), fc.get_tensor(k).view(torch.int16))
        rows.append((k,tuple(a.shape),str(fo.get_tensor(k).dtype),d/max(n,1e-30),bitwise))
for r in rows: print('%-50s %-22s %-9s relchg=%.5f bitwise_identical=%s'%(r[0],str(r[1]),r[2],r[3],r[4]))
print('bitwise identical tensors:', [r[0] for r in rows if r[4]])
print('num bitwise identical:', sum(r[4] for r in rows), 'of', len(rows))
