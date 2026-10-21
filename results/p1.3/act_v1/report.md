# P1.3 held-out MuJoCo evaluation

Independent per-training-seed Wilson intervals; the same evaluation seeds are not pooled across models.

| Training seed | Successes / trials | Success rate | Wilson 95% interval | Ledger row |
|---|---|---|---|---|
| 0 | 30 / 30 | 1.0000 | [0.8865, 1.0000] | d35d3a3e-8f9b-4a9e-a6e4-d5283d807197 |
| 1 | 30 / 30 | 1.0000 | [0.8865, 1.0000] | 8ab93c8e-e880-49c2-a7f6-2229615d92e2 |
| 2 | 30 / 30 | 1.0000 | [0.8865, 1.0000] | c98903f8-a108-4e12-8b59-b1091ae48467 |

Scripted noisy demonstrations, privileged simulator state, pinned XLeRobot left arm. This is not SO-101 hardware, visual perception, or a human-data quality-gate experiment.

The budget and chunk selection rule were fixed before held-out evaluation; these results do not establish convergence or optimal hyperparameters.

## Interpretation notes

The committed configuration used 3 sweep epochs for each chunk length 10/20/50.
Validation action MAE selected 50 before any held-out evaluation. Each final
training seed (0/1/2) received 20 epochs and was evaluated on 10000..10029.
The selection objective scores all future chunk positions, not ensembled rollout
performance; changing chunk length changes the forecasting task. Training seeds
also change the episode-level train/validation split.

Exponential ensemble weights favor newer predictions, unlike the older-favored
weighting in [original ACT](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py).
Recovery data, training budget, chunk selection, and execution changed together:
there is no controlled ablation attributing gains to ensembling alone. Different
evaluation seeds prevent a direct comparison to historical runs.

Training and measurement source remained frozen at commit
`3baa44ba69056af555c1de7633b6a9942a21af6b`. The initial configuration records
`git_dirty=false`; evaluation rows retain `git_dirty=true` because generated
evidence and the explainer draft existed by then. Completed model, dataset, and
rollout artifacts retain their checksums.

Evidence: [ledger](../../ledger.jsonl), [configuration](config.json),
[validation selection](selection.json), and [completed outcomes](outcomes.json).
