# candidate vs base (paired, task-level)

Tasks scored for both models: **135**. Samples per task: base 6.0, candidate 6.0.

| | base | candidate |
|---|---:|---:|
| mean pass rate | 75.9% | 76.8% |
| pass@1 | 75.9% | 76.8% |
| pass@2 | 84.9% | 85.1% |
| pass@4 | 89.1% | 90.1% |
| pass rate among attempts that finished | 89.1% | 89.9% |
| truncated (hit token budget) | 14.8% | 14.6% |
| finished without code | 0.6% | 0.6% |
| mean completion tokens | 2,006 | 1,997 |
| mean tokens on passes | 1,607 | 1,620 |

**Paired difference (candidate - base): +0.86 pp**, 95% cluster-bootstrap CI [-2.10, +3.83] pp, sign-flip permutation p = 0.623.
Tasks improved / regressed / tied: 30 / 24 / 81. McNemar (majority outcome per task): candidate-only 7, base-only 6, exact p = 1.000.

| rating bin | tasks | mean delta (pp) |
|---|---:|---:|
| 1200-1499 | 30 | -1.67 |
| 1500-1799 | 41 | +2.44 |
| 1800-2099 | 25 | +6.00 |
| 2100-2399 | 6 | -8.33 |
| 900-1199 | 33 | -1.01 |

CI/p-values treat tasks as the sampling unit; with this many tasks only effects of several points are detectable.
