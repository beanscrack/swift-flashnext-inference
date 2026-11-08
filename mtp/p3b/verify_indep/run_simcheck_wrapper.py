# Independent wrapper: runs UNMODIFIED p3/sim_check.py main() on 20 heldout samples (sequential reference).
#   arg 'orig' -> TAIL weights as shipped; arg 'cand' -> TAIL + best.safetensors mtp.* via load_mtp_tensors.
# PREDS is pointed at the matching simulate.py run (verify_indep/preds_<arg>.npz) for the argmax agreement check.
import sys
MODE_ARG = sys.argv[1]
OUTDIR = '/u701/j692wu/swift-opt/mtp/p3b/verify_indep'
sys.path.insert(0, '/u701/j692wu/swift-opt/mtp')
sys.path.insert(0, '/u701/j692wu/swift-opt/mtp/p3')
sys.argv = [sys.argv[0]]
import torch
from safetensors import safe_open
import sim_check
from mtp_torch import MTPDraft
from data import TAIL, CONFIG
CAND = '/u701/j692wu/swift-opt/mtp/p3b/ckpt/best.safetensors'
sim_check.OUT = f'{OUTDIR}/simcheck_{MODE_ARG}.json'
sim_check.PREDS = f'{OUTDIR}/preds_{MODE_ARG}.npz'
if MODE_ARG == 'cand':
    class _Loader:
        @staticmethod
        def from_safetensors(path, cfg, dtype, device):
            m = MTPDraft.from_safetensors(path, cfg, dtype=dtype, device=device).eval()
            with safe_open(CAND, framework='pt', device='cpu') as f:
                t = {k: f.get_tensor(k) for k in f.keys()}
            m.load_mtp_tensors(t)
            return m
    sim_check.MTPDraft = _Loader
sim_check.main()
