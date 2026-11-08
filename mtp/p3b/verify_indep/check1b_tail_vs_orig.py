import sys
import torch
from safetensors import safe_open
sys.path.insert(0, '/u701/j692wu/swift-opt/mtp/p3')
from data import TAIL
ORIG = '/u701/j692wu/swift-opt/mtp/mtp_original.safetensors'
n = 0
diff = []
with safe_open(ORIG, 'pt', device='cpu') as fo, safe_open(TAIL, 'pt', device='cpu') as ft:
    mt = [k for k in fo.keys() if k.startswith('mtp.')]
    for k in mt:
        if k not in ft.keys():
            diff.append(('missing', k))
            continue
        if not torch.equal(fo.get_tensor(k).view(torch.int16), ft.get_tensor(k).view(torch.int16)):
            diff.append(('differs', k))
        n += 1
print('TAIL vs mtp_original: mtp tensors compared', n, 'of', len(mt), 'problems', diff[:5])
