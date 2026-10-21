# Runtime recorder review

P2.2 source `d8757d4` and evidence `d0bba9f` are accepted for scripted
MuJoCo recording. This establishes data preservation through the actual
LeRobot reader; it does not establish live-camera latency, human demonstration
quality, task success, or hardware behavior.

## Review boundary

The actual Claude Code CLI reviewed fixed source `c35fd85` using repository-scoped
Read, Glob and Grep only and reported model `claude-opus-5-5`. Codex independently
ran the real LeRobot recorder tests, inspected connected runtime behavior, and
verified the follow-ups `7bb06de` and `d8757d4`. Raw Claude output and independent
probes remain local under the QA worktree's ignored `build/qa` directory.

The constructor could replace a primary setup error with a secondary cleanup
error. An independent double-failure probe reproduced that behavior. Fix
`7bb06de` preserves the primary error and attaches cleanup and staging notes;
the same probe passed afterward. The provenance follow-up enumerates and hashes
published files without changing recording behavior.

## Evidence checked

The measurement used clean source `d8757d4`. Each dataset contains twelve
scripted transitions, with measured maximum reconstruction error zero. The
append-only ledger records are:

| Embodiment | Ledger row |
|---|---|
| SO-101 | `d8305772-fdea-4517-97f3-f82fae77e446` |
| G1 upper body | `f1ed95f7-a25d-4f35-86e5-4e9fb705e409` |
| LEAP hand | `a0f12489-5961-4b2b-b7b3-ee2ba9938b93` |

For every published file under `results/p2.2/scripted_runtime_v1`, Codex checked
the complete inventory and SHA-256 against its ledger row. For every journal
entry, the actual reader matched pre-action state, applied targets, successor
state, joint velocities, exact float64 stage times, missing-value masks,
clamp counts/events, phase and source fields. State continuity and frame indices
also matched. The ordinary LeRobot reader preserved state/action arrays; the
feature-aware helper preserved declared float64 timing fields.

Manifests retain the embodiment specification hash and explicitly identify
scripted input and a scripted clock. Missing tracking confidence remains missing,
and task outcome remains unspecified. Transaction completion does not imply task
success. Failure tests cover journal retention and refusal to replace existing
outputs. These software checks are separate from the ledger-backed measurement.

The optional-interpreter native binary was supplied by the implementation lane;
QA did not independently rebuild it. No camera or real motors were accessed.
