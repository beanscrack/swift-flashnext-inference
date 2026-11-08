import json
ev=[json.loads(l) for l in open('/u701/j692wu/swift-opt/mtp/p3b/logs/evals.jsonl')]
print('step epoch  acc0    acc1    acc2    acc3    tpc     n_rounds')
for e in ev:
    print('%4d  %d  '%(e['step'],e['epoch']) + ' '.join('%.4f'%a for a in e['acc']) + '  %.4f  %d'%(e['tokens_per_cycle'], e['n_rounds']))
steps=[e['step'] for e in ev]; print('eval steps in order', steps)
acc3=[e['acc'][3] for e in ev]; tpc=[e['tokens_per_cycle'] for e in ev]
print('acc3 monotone nondecreasing over eval order:', all(b>=a for a,b in zip(acc3,acc3[1:])))
print('tpc monotone nondecreasing over eval order:', all(b>=a for a,b in zip(tpc,tpc[1:])))
best=max(ev,key=lambda e:e['tokens_per_cycle'])
print('argmax tpc step', best['step'], 'is last eval', best['step']==ev[-1]['step'], 'is last trained step 996:', best['step']==996)
print('num evals (heldout selections)', len(ev))
print('step-996 acc', [round(a,4) for a in ev[-1]['acc']], 'tpc', round(ev[-1]['tokens_per_cycle'],4))
print('2nd best tpc', sorted(((e['tokens_per_cycle'],e['step']) for e in ev), reverse=True)[:3])
