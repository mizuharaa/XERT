# Official ACT reference review

Implementation `0c26610` is accepted for the predeclared P1.4 experiment.
Measurement acceptance is pending completed training, held-out evaluations and
artifact verification. This review does not assert a policy success rate.

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

Final evidence review must verify model/source/artifact hashes, training-only
statistics, raw rollout outcomes and initial-state identity, per-model Wilson
intervals, and the ledger-generated comparison. Shared evaluation states must not
be pooled as independent extra trials. All results remain simulation-only.
