import json, sys
sys.path.insert(0, '/u701/j692wu/swift-opt/mtp/p5')
import raw_ab as R
runs = {'p4-identity': '/u701/j692wu/swift-opt/runs/p4-identity', 'p4-mtp-v1': '/u701/j692wu/swift-opt/runs/p4-mtp-v1',
        'p5-mtp-v1': '/u701/j692wu/swift-opt/runs/p5-mtp-v1', 'p5-identity': '/u701/j692wu/swift-opt/runs/p5-identity'}
info = {k: R.run_info(v) for k, v in runs.items()}
print('| run | n | mean | min | max | sd | coding tok/cycle | all-phase tok/cycle | logprob | top1 % | tracebacks | model |')
for k, o in info.items():
    t = o['rep_tps']
    print('| %s | %d | %.2f | %.2f | %.2f | %.2f | %.4f | %.4f | %.4f | %.2f | %d | %s |' % (k, len(t), R.mean(t), min(t), max(t), R.sd(t), o['coding_tpc'], o['all_tpc'], o['logprob'], o['top1']*100, o['tracebacks'], o['model']))
print('coding per-pos acceptance:')
for k, o in info.items(): print(' ', k, {p: round(v, 4) for p, v in o['coding_pos'].items()})
p5m = info['p5-mtp-v1']['rep_tps']; p5i = info['p5-identity']['rep_tps']
p4m = info['p4-mtp-v1']['rep_tps']; p4i = info['p4-identity']['rep_tps']
t, df, p = R.welch(p5i, p5m)
print('Welch p5 alone: mtp-v1 %.2f vs identity %.2f diff %.2f t=%.3f df=%.2f p=%.4g' % (R.mean(p5m), R.mean(p5i), R.mean(p5m)-R.mean(p5i), t, df, p))
t2, df2, p2 = R.welch(p4i + p5i, p4m + p5m)
print('Welch pooled p4+p5: mtp-v1 n=%d mean %.2f vs identity n=%d mean %.2f diff %.2f t=%.3f df=%.2f p=%.4g' % (len(p4m+p5m), R.mean(p4m+p5m), len(p4i+p5i), R.mean(p4i+p5i), R.mean(p4m+p5m)-R.mean(p4i+p5i), t2, df2, p2))
print('tokens/cycle ratio mtp-v1/identity (coding): p4 %.4f  p5 %.4f' % (info['p4-mtp-v1']['coding_tpc']/info['p4-identity']['coding_tpc'], info['p5-mtp-v1']['coding_tpc']/info['p5-identity']['coding_tpc']))
print('tokens/cycle ratio mtp-v1/identity (all-phase): p4 %.4f  p5 %.4f' % (info['p4-mtp-v1']['all_tpc']/info['p4-identity']['all_tpc'], info['p5-mtp-v1']['all_tpc']/info['p5-identity']['all_tpc']))
for k, o in info.items():
    lp_ok = -0.2144 <= o['logprob'] <= -0.2104
    t1_ok = 96.9 <= o['top1']*100 <= 98.1
    print('window %s: logprob %.4f in [-0.2144,-0.2104]=%s ; top1 %.2f%% in [96.9,98.1]=%s' % (k, o['logprob'], lp_ok, o['top1']*100, t1_ok))
