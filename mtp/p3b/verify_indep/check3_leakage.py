import json, sys, collections
sys.path.insert(0, '/u701/j692wu/swift-opt/mtp/p3')
GEN='/u701/j692wu/swift-opt/mtp/p2/gen.jsonl'
# independent parse of gen.jsonl (same fields data.py reads)
rows=[json.loads(l) for l in open(GEN)]
print('gen.jsonl rows', len(rows), 'split counts', collections.Counter(r['split'] for r in rows))
held_ids={r['id'] for r in rows if r['split']=='heldout'}
train_ids={r['id'] for r in rows if r['split']=='train'}
print('heldout ids', len(held_ids), 'train ids', len(train_ids), 'overlap', len(held_ids & train_ids), 'dup ids in file', len(rows)-len({r['id'] for r in rows}))
print('splits other than train/heldout:', set(r['split'] for r in rows)-{'train','heldout'})
# cross-check against data.load_gen (the loader the eval and train both use)
import data
dh={r['id'] for r in data.load_gen('heldout')}; dt={r['id'] for r in data.load_gen('train')}
print('data.load_gen heldout == independent heldout:', dh==held_ids, len(dh))
print('data.load_gen train == independent train:', dt==train_ids, len(dt))
# ids the simulators evaluated (sim_results_bf16 from p3 and the verify_indep sim runs if present)
import os
for p in ['/u701/j692wu/swift-opt/mtp/p3/sim_results_bf16.json','/u701/j692wu/swift-opt/mtp/p3b/verify_indep/sim_orig.json','/u701/j692wu/swift-opt/mtp/p3b/verify_indep/sim_cand.json']:
    if os.path.exists(p):
        r=json.load(open(p)); ids={s['id'] for s in r['per_sample']}
        print(os.path.basename(p), 'n_per_sample', len(ids), 'equals heldout set', ids==held_ids, 'n_samples', r['n_samples'])
# training: sample ids in summary, train_steps counts
s=json.load(open('/u701/j692wu/swift-opt/mtp/p3b/logs/summary.json'))
print('summary: n_train_used', s['n_train_used'], 'n_heldout', s['n_heldout'], 'heldout_ids_in_gradient', s['heldout_ids_in_gradient'], 'train_samples_in_gradient_total', s['train_samples_in_gradient_total'], 'frozen_identical', s['frozen_identical'], 'stop_reason', s['stop_reason'], 'steps_done', s['steps_done'], 'total', s['total_steps'])
print('skipped_no_valid_label', s['skipped_no_valid_label'])
print('train used == train split size minus C<5:', s['n_train_used'] == sum(1 for r in rows if r['split']=='train' and r['C']>=5) if False else sum(1 for r in rows if r['split']=='train' and len(r['completion_token_ids'])>=5))
