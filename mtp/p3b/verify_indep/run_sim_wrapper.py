# Independent wrapper: runs UNMODIFIED p3/simulate.py main(). Only the weight loader and output paths are swapped.
#   arg 'orig' -> simulate.py's own loader (verify_ttt.load_model_and_head: MTPDraft from TAIL, mtp.* as shipped in TAIL)
#   arg 'cand' -> same TAIL-based model, then the 31 mtp.* tensors replaced by p3b/ckpt/best.safetensors
#                 via MTPDraft.load_mtp_tensors (identical to how p3b/train.py evaluates).
import sys
MODE_ARG = sys.argv[1]
OUTDIR = '/u701/j692wu/swift-opt/mtp/p3b/verify_indep'
sys.path.insert(0, '/u701/j692wu/swift-opt/mtp')
sys.path.insert(0, '/u701/j692wu/swift-opt/mtp/p3')
sys.argv = [sys.argv[0]]  # verify_ttt reads argv[1] for MODE; keep default bf16
import numpy as np
import torch
from safetensors import safe_open
import simulate
from mtp_torch import MTPDraft
from data import TAIL, CONFIG, VOCAB
CAND = '/u701/j692wu/swift-opt/mtp/p3b/ckpt/best.safetensors'
simulate.OUT = f'{OUTDIR}/sim_{MODE_ARG}.json'
simulate.PREDS = f'{OUTDIR}/preds_{MODE_ARG}.npz'
if MODE_ARG == 'cand':
    def loader():
        model = MTPDraft.from_safetensors(TAIL, CONFIG, dtype=torch.bfloat16, device='cuda').eval()
        with safe_open(CAND, framework='pt', device='cpu') as f:
            t = {k: f.get_tensor(k) for k in f.keys()}
        model.load_mtp_tensors(t)
        with safe_open(TAIL, framework='pt', device='cpu') as f:
            lm_head = f.get_tensor('lm_head.weight').to('cuda', torch.bfloat16)
        vocab = torch.from_numpy(np.load(VOCAB).astype(np.int64)).cuda()
        return model, lm_head[vocab].contiguous(), vocab
    simulate.load_model_and_head = loader
simulate.main()
