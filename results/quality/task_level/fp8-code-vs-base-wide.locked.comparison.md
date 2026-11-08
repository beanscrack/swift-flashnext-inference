# candidate vs base (paired, task-level)

Tasks scored for both models: **135**. Samples per task: base 6.0, candidate 6.0.

| | base | candidate |
|---|---:|---:|
| mean pass rate | 75.9% | 73.8% |
| pass@1 | 75.9% | 73.8% |
| pass@2 | 84.9% | 83.6% |
| pass@4 | 89.1% | 88.6% |
| pass rate among attempts that finished | 89.1% | 87.8% |
| truncated (hit token budget) | 14.8% | 15.9% |
| finished without code | 0.6% | 1.7% |
| mean completion tokens | 2,006 | 2,086 |
| mean tokens on passes | 1,607 | 1,695 |

**Paired difference (candidate - base): -2.10 pp**, 95% cluster-bootstrap CI [-5.06, +0.99] pp, sign-flip permutation p = 0.210.
Tasks improved / regressed / tied: 25 / 33 / 77. McNemar (majority outcome per task): candidate-only 7, base-only 8, exact p = 1.000.

| rating bin | tasks | mean delta (pp) |
|---|---:|---:|
| 1200-1499 | 30 | -11.11 |
| 1500-1799 | 41 | +4.47 |
| 1800-2099 | 25 | +1.33 |
| 2100-2399 | 6 | -16.67 |
| 900-1199 | 33 | -2.02 |

CI/p-values treat tasks as the sampling unit; with this many tasks only effects of several points are detectable.
