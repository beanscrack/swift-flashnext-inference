import json, re, sys, math, os, glob

NAME = r'^(vllm:spec_decode_num_drafts_total|vllm:spec_decode_num_accepted_tokens_total)\{[^}]*\}\s+([0-9.eE+-]+)'
POS = re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"\}\s+([0-9.eE+-]+)')
TOT = re.compile(r'^vllm:spec_decode_num_(drafts|accepted_tokens)_total\{[^}]*\}\s+([0-9.eE+-]+)')

def parse_metrics(path):
    d = {'drafts': 0.0, 'accepted': 0.0, 'pos': {}}
    with open(path) as f:
        for line in f:
            m = TOT.match(line)
            if m:
                d['drafts' if m.group(1) == 'drafts' else 'accepted'] = float(m.group(2))
                continue
            m = POS.match(line)
            if m:
                d['pos'][int(m.group(1))] = float(m.group(2))
    return d

def diff(a, b):
    return {'drafts': b['drafts'] - a['drafts'], 'accepted': b['accepted'] - a['accepted'],
            'pos': {k: b['pos'].get(k, 0) - a['pos'].get(k, 0) for k in set(a['pos']) | set(b['pos'])}}

def tpc(d):
    return d['accepted'] / d['drafts'] + 1 if d['drafts'] else float('nan')

def pos_rates(d):
    return {k: d['pos'][k] / d['drafts'] for k in sorted(d['pos'])} if d['drafts'] else {}

def mean(x): return sum(x) / len(x)
def sd(x):
    m = mean(x); return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))

# regularized incomplete beta via continued fraction (Numerical Recipes style)
def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > FPMIN else FPMIN); h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > FPMIN else FPMIN)
        c = 1 + aa / c; c = c if abs(c) > FPMIN else FPMIN; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > FPMIN else FPMIN)
        c = 1 + aa / c; c = c if abs(c) > FPMIN else FPMIN
        de = d * c; h *= de
        if abs(de - 1) < EPS: break
    return h

def betai(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * betacf(a, b, x) / a
    return 1 - bt * betacf(b, a, x) / b

def t_two_sided_p(t, df):
    return betai(df / 2, 0.5, df / (df + t * t))

def welch(x, y):
    mx, my = mean(x), mean(y); vx, vy = sd(x) ** 2, sd(y) ** 2
    nx, ny = len(x), len(y)
    se2 = vx / nx + vy / ny
    t = (my - mx) / math.sqrt(se2)  # y minus x
    df = se2 ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return t, df, t_two_sided_p(t, df)

def run_info(run):
    out = {}
    reps = []
    for r in range(1, 9):
        p = f'{run}/rep{r}.out'
        if not os.path.exists(p): continue
        lines = [l for l in open(p).read().splitlines() if l.strip()]
        last = json.loads(lines[-1])
        reps.append(last['mean_decode_tps'])
    out['rep_tps'] = reps
    # cross-check: mean of per-request decode_tps from requests.json
    xc = []
    for r in range(1, 9):
        p = f'{run}/rep{r}/requests.json'
        if os.path.exists(p):
            reqs = json.load(open(p))
            xc.append(mean([q['decode_tps'] for q in reqs]))
    out['rep_tps_from_requests'] = xc
    # coding phase = sum of rep-level metric deltas (reps 1-5 cover only decode cases)
    coding = {'drafts': 0.0, 'accepted': 0.0, 'pos': {}}
    for r in range(1, 9):
        b = f'{run}/rep{r}/metrics-before.txt'; a = f'{run}/rep{r}/metrics-after.txt'
        if not os.path.exists(b): continue
        dd = diff(parse_metrics(b), parse_metrics(a))
        coding['drafts'] += dd['drafts']; coding['accepted'] += dd['accepted']
        for k, v in dd['pos'].items(): coding['pos'][k] = coding['pos'].get(k, 0) + v
    out['coding_tpc'] = tpc(coding); out['coding_pos'] = pos_rates(coding)
    out['coding_drafts'] = coding['drafts']
    allp = parse_metrics(f'{run}/metrics-final.txt')
    out['all_tpc'] = tpc(allp); out['all_pos'] = pos_rates(allp); out['all_drafts'] = allp['drafts']
    sc = open(f'{run}/extra.out').read().strip()
    m = re.search(r'mean logprob (-?[0-9.]+); top1 fraction ([0-9.]+)', sc)
    out['logprob'] = float(m.group(1)); out['top1'] = float(m.group(2))
    srv = open(f'{run}/server.log', errors='replace').read()
    out['tracebacks'] = srv.count('Traceback')
    mm = re.search(r'non-default args: \{.*?\'model\': \'([^\']+)\'', srv)
    out['model'] = mm.group(1) if mm else 'NOT FOUND'
    return out

if __name__ == '__main__':
    runs = sys.argv[1:]
    res = {r: run_info(r) for r in runs}
    for r, o in res.items():
        print('==', r)
        print('  rep tok/s:', ['%.2f' % v for v in o['rep_tps']])
        print('  rep tok/s (mean of requests.json decode_tps):', ['%.2f' % v for v in o['rep_tps_from_requests']])
        print('  mean %.3f sd %.3f min %.2f max %.2f' % (mean(o['rep_tps']), sd(o['rep_tps']), min(o['rep_tps']), max(o['rep_tps'])))
        print('  coding tokens/cycle %.4f  drafts %d' % (o['coding_tpc'], o['coding_drafts']))
        print('  coding pos acc:', {k: round(v, 4) for k, v in o['coding_pos'].items()})
        print('  all-phase tokens/cycle %.4f  drafts %d' % (o['all_tpc'], o['all_drafts']))
        print('  all-phase pos acc:', {k: round(v, 4) for k, v in o['all_pos'].items()})
        print('  logprob %.4f top1 %.4f%%' % (o['logprob'], o['top1'] * 100))
        print('  tracebacks', o['tracebacks'], ' model', o['model'])
    json.dump(res, open('/u701/j692wu/swift-opt/mtp/p5/raw_ab_' + os.environ.get('TAG', 'out') + '.json', 'w'), indent=1)
