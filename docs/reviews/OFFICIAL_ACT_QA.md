# Official ACT reference review

Implementation `0c26610` and completed evidence commit `3988e41` for the
predeclared P1.4 experiment are accepted. Results below describe MuJoCo simulation with scripted recovery
demonstrations and privileged simulator state.

## Review boundary

The actual Claude Code CLI reviewed fixed source `106225c` with repository-scoped
Read, Glob and Grep only and reported model `claude-opus-5-5`. Codex independently
reviewed the installed LeRobot implementation, executed the optional package tests
and preflight checks, and reviewed the provenance/disclosure follow-up `0c26610`.
That follow-up did not change training behavior. Raw review outputs remain local
under the QA worktree's ignored `build/qa` directory.

## Checks completed

- Every legacy observation feature is retained in the state/environment partition.
  Real LeRobot arrays and corrective-label fingerprints match the source episodes.
- Training-only normalization and each original episode split match P1.3 exactly;
  whole-export LeRobot statistics are not used.
- Chunks stop at episode boundaries. The installed official loss and variational
  encoder use the padding mask. Prior validation enters evaluation mode and does
  not receive future action labels.
- The installed official policy queries its temporal ensemble each control tick
  and clears it on episode reset. Saved checkpoints load through the official API.
  Held-out rollouts use the learned policy without an expert fallback.
- Actual initial-state hashes match P1.3 across dependency stacks. A bounded replay
  of a saved P1.3 action trajectory also matched every recorded state, reward,
  termination flag and final outcome in the optional environment.
- The fixed configuration selects checkpoints on validation error before held-out
  evaluation, runs training seeds sequentially, retains per-model intervals and
  refuses an existing experiment output directory.

The native binary used by QA was built by the learning lane for the optional
interpreter; QA did not independently rebuild it. These checks establish software
and data compatibility, not a new policy-performance measurement.

## Findings and limits

Claude flagged assumptions about older ACT padding and evaluation behavior. Reading
the installed LeRobot source resolved them: its loss divides by valid elements,
prior prediction switches to evaluation mode, and inference disables gradients.
No algorithm change was needed.

The follow-up adds hashes of the installed official model and configuration source
alongside versions and the runner hash. Documentation now distinguishes official
linear action predictions from the prior model's bounded tanh predictions. Raw
predictions determine validation error; execution clips commands after ensembling.
The models also differ in architecture and ensemble weighting, so this comparison
cannot isolate the cause of a performance difference or establish convergence.

## Completed measurement review

Codex independently verified every success-row artifact hash, all raw rollout
arrays and outcomes, matching initial-state hashes, original per-seed training
splits and normalization, and checkpoint selection by minimum validation error.
Representative saved-action trajectories exactly reproduced every state, reward,
termination flag and outcome, including a failed rollout. This bounded replay
checks saved evidence; it is not an additional policy-performance measurement.

| Training seed | Official ACT successes / trials | Ledger row |
|---|---|---|
| 0 | 30 / 30 | `15103d29-afdb-4853-8e63-717e448723ba` |
| 1 | 18 / 30 | `93e321d8-0b9c-49d0-9692-b6ecb8af6909` |
| 2 | 30 / 30 | `18dee197-8c4f-43ae-8324-0a1b996e2dbf` |

Independent Wilson calculations match each row. The ledger-generated
`results/p1.4/official_act_v1/comparison.md` matches both official and earlier
XERT rows, including the weaker middle-seed result. Failed trials remain in the
dataset; the frozen plan was not retuned after evaluation. Source, configuration
and dependency declarations remain unchanged from the clean source commit, and
the previous ledger bytes remain an exact prefix.

The experiment completion flag is true. Shared evaluation states must not be
pooled as independent extra trials. The comparison does not show a general model
ranking, training convergence, quality-gate benefit, or hardware performance.
