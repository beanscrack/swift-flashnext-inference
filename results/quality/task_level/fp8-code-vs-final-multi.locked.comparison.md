# candidate vs base (paired, task-level)

Tasks scored for both models: **135**. Samples per task: base 6.0, candidate 6.0.

| | base | candidate |
|---|---:|---:|
| mean pass rate | 76.8% | 73.8% |
| pass@1 | 76.8% | 73.8% |
| pass@2 | 85.1% | 83.6% |
| pass@4 | 90.1% | 88.6% |
| pass rate among attempts that finished | 89.9% | 87.8% |
| truncated (hit token budget) | 14.6% | 15.9% |
| finished without code | 0.6% | 1.7% |
| mean completion tokens | 1,997 | 2,086 |
| mean tokens on passes | 1,620 | 1,695 |

**Paired difference (candidate - base): -2.96 pp**, 95% cluster-bootstrap CI [-6.05, +0.12] pp, sign-flip permutation p = 0.077.
Tasks improved / regressed / tied: 21 / 39 / 75. McNemar (majority outcome per task): candidate-only 5, base-only 7, exact p = 0.774.

| rating bin | tasks | mean delta (pp) |
|---|---:|---:|
| 1200-1499 | 30 | -9.44 |
| 1500-1799 | 41 | +2.03 |
| 1800-2099 | 25 | -4.67 |
| 2100-2399 | 6 | -8.33 |
| 900-1199 | 33 | -1.01 |

CI/p-values treat tasks as the sampling unit; with this many tasks only effects of several points are detectable.
