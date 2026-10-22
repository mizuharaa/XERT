# Official ACT reference comparison (MuJoCo only)

| Model | Training seed | Successes / trials | Wilson 95% interval | Ledger row |
|---|---|---|---|---|
| XERT ACT-style | 0 | 30 / 30 | [0.8865, 1.0000] | d35d3a3e-8f9b-4a9e-a6e4-d5283d807197 |
| LeRobot ACT | 0 | 30 / 30 | [0.8865, 1.0000] | 15103d29-afdb-4853-8e63-717e448723ba |
| XERT ACT-style | 1 | 30 / 30 | [0.8865, 1.0000] | 8ab93c8e-e880-49c2-a7f6-2229615d92e2 |
| LeRobot ACT | 1 | 18 / 30 | [0.4232, 0.7541] | 93e321d8-0b9c-49d0-9692-b6ecb8af6909 |
| XERT ACT-style | 2 | 30 / 30 | [0.8865, 1.0000] | c98903f8-a108-4e12-8b59-b1091ae48467 |
| LeRobot ACT | 2 | 30 / 30 | [0.8865, 1.0000] | 18dee197-8c4f-43ae-8324-0a1b996e2dbf |

Predeclared CPU reference using official LeRobot ACT. Same data, split indices, training seeds, final epoch budget and held-out reset states as P1.3; chunk50 fixed from its validation sweep. Manual train-only observation normalization; actions retain normalized target units. Full observation partition: q+qd12 and environment14. Official oldest-favored ensemble differs from P1.3 newest-favored. Architecture and dependency versions also differ; no causal ablation or claim of convergence.

The same evaluation states recur across models; do not pool trials. This comparison does not establish optimal training or isolate one architectural change.
