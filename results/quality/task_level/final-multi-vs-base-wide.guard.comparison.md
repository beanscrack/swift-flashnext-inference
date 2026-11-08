# candidate vs base (paired, task-level)

Tasks scored for both models: **542**. Samples per task: base 1.0, candidate 1.0.

| | base | candidate |
|---|---:|---:|
| mean pass rate | 84.1% | 83.4% |
| pass@1 | 84.1% | 83.4% |
| pass@2 | nan% | nan% |
| pass@4 | nan% | nan% |
| pass rate among attempts that finished | 84.3% | 83.7% |
| truncated (hit token budget) | 0.2% | 0.4% |
| finished without code | 0.0% | 0.0% |
| mean completion tokens | 198 | 204 |
| mean tokens on passes | 185 | 185 |

**Paired difference (candidate - base): -0.74 pp**, 95% cluster-bootstrap CI [-1.66, +0.18] pp, sign-flip permutation p = 0.222.
Tasks improved / regressed / tied: 1 / 5 / 536. McNemar (majority outcome per task): candidate-only 1, base-only 5, exact p = 0.219.

| rating bin | tasks | mean delta (pp) |
|---|---:|---:|
| 0-299 | 542 | -0.74 |

CI/p-values treat tasks as the sampling unit; with this many tasks only effects of several points are detectable.
