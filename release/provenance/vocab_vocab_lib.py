"""Shared helpers: load calibration counts, build the family mixtures, select ranked subsets."""
import json, os
import numpy as np

D = '/u701/j692wu/swift-profile/vocab'
V = 248320            # lm_head rows
NTOK = 248077         # ids that exist in the tokenizer (248044 base + 33 added); rows >= NTOK are padding
SIZES = [8192, 16384, 24576, 32768, 49152, 65536]
CODE_GROUPS = ['py', 'cpp', 'js', 'ts', 'rs', 'sql', 'sh', 'misc']


def P(c):
    c = np.asarray(c, dtype=np.float64)
    s = c.sum()
    return c / s if s > 0 else c


def load_counts():
    cnt = {g: np.load(f'{D}/counts_{g}.npy') for g in CODE_GROUPS + ['docs', 'man', 'struct']}
    mc = np.load(f'{D}/model_counts.npz')
    model = {k: mc[k] for k in mc.files}      # keys like nv__swe__answer
    return cnt, model


def msum(model, pred):
    out = np.zeros(V, dtype=np.int64)
    n = 0
    for k, v in model.items():
        if pred(k):
            out += v; n += 1
    return out


def families(cnt, model):
    """per-family probability vectors (each sums to 1)"""
    code = np.mean([P(cnt[g]) for g in CODE_GROUPS], axis=0)
    prose = 0.6 * P(cnt['docs']) + 0.4 * P(cnt['man'])
    ans = (0.25 * P(model['nv__swe__answer']) + 0.25 * P(model['nv__opencode__answer'])
           + 0.15 * P(model['nv__code_reasoning__answer']) + 0.10 * P(model['nv__terminal_pivot__answer'])
           + 0.25 * P(msum(model, lambda k: k.startswith('rft__') and k.endswith('__answer'))))
    think = (0.5 * P(msum(model, lambda k: k.startswith('nv__') and k.endswith('__think')))
             + 0.5 * P(msum(model, lambda k: k.startswith('rft__') and k.endswith('__think'))))
    mod = 0.5 * ans + 0.5 * think
    struct = P(cnt['struct'])
    return dict(code=code, prose=prose, model=mod, model_answer=ans, model_think=think, struct=struct,
                docs=P(cnt['docs']), man=P(cnt['man']))


W_DEFAULT = dict(code=0.35, prose=0.20, model=0.40, struct=0.05)
# v2: answer-style outputs dominate (chosen with leave-one-prompt-out, see mix_tune.py / REPORT.md)
W_V2 = dict(code=0.15, docs=0.10, model_answer=0.72, struct=0.03)
MIXES = {'': W_DEFAULT, '_v2': W_V2}


def mixture(fam, w):
    v = np.zeros(V)
    for k, x in w.items():
        v += x * fam[k]
    return v


def guaranteed():
    g = json.load(open(f'{D}/guaranteed_ids.json'))
    ids = set(g['special_added']) | set(g['eos']) | set(g['single_byte']) | set(g['digits']) | set(g['whitespace'])
    return sorted(i for i in ids if i < NTOK), g


def select(vec, n, forced):
    """forced ids first, then highest `vec` (ties -> lower id), restricted to real tokenizer ids; sorted ascending."""
    forced = np.asarray(sorted(forced), dtype=np.int64)
    assert len(forced) <= n
    score = vec[:NTOK].copy()
    score[forced] = np.inf
    order = np.argsort(-score, kind='stable')[:n]
    ids = np.sort(order).astype(np.int32)
    assert len(ids) == n and len(np.unique(ids)) == n
    return ids


def mask_of(ids):
    m = np.zeros(V, dtype=bool)
    m[np.asarray(ids)] = True
    return m
