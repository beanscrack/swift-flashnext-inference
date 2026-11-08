# candidate vs base (paired, task-level)

Tasks scored for both models: **542**. Samples per task: base 1.0, candidate 1.0.

| | base | candidate |
|---|---:|---:|
| mean pass rate | 84.1% | 84.9% |
| pass@1 | 84.1% | 84.9% |
| pass@2 | nan% | nan% |
| pass@4 | nan% | nan% |
| pass rate among attempts that finished | 84.3% | 85.0% |
| truncated (hit token budget) | 0.2% | 0.2% |
| finished without code | 0.0% | 0.0% |
| mean completion tokens | 198 | 198 |
| mean tokens on passes | 185 | 187 |

**Paired difference (candidate - base): +0.74 pp**, 95% cluster-bootstrap CI [-0.18, +1.85] pp, sign-flip permutation p = 0.291.
Tasks improved / regressed / tied: 6 / 2 / 534. McNemar (majority outcome per task): candidate-only 6, base-only 2, exact p = 0.289.

| rating bin | tasks | mean delta (pp) |
|---|---:|---:|
| 0-299 | 542 | +0.74 |

CI/p-values treat tasks as the sampling unit; with this many tasks only effects of several points are detectable.
