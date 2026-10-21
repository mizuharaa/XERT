# Simulation MVP review

Status: P0.1–P0.3 retain their previous acceptance; P1.1–P1.3 and
P3.1–P3.4 are accepted. P1.3's completed experiment passed independent artifact
and provenance checks. This is a slice checklist,
not a claim that the remaining product work has equal effort.

## Independent review method

The separate QA worktree used the actual Claude Code CLI. Its returned model
identity was `claude-opus-5-5`. Claude had only repository-scoped Read, Glob and
Grep tools; it could neither run commands nor change implementation. Reviews
used fixed commits. Raw JSON reports remain local under `build/qa` in the QA
worktree and are not committed.

The Codex QA lane independently inspected findings, executed focused checks,
and checked fixes. Claude's static conclusions are not runtime test results.
Only the P3.1 fix received an additional Claude re-review; later fixes were
closed through targeted source review and executable reproductions.

| Slice | Reviewed implementation/evidence | Findings and resolution |
|---|---|---|
| P1.1 | `94493ae` | Real learned-action replay distinguished feedback failure from prediction error. `01ee216` softened the causal claim and clarified truncated replays; append-only corrections preserve the original rows. Saved diagnostic source hashes were independently verified. |
| P1.2 | `511a4f8` | Executed actions retain exact replay semantics; separately hashed corrective labels feed training. The loader hashes and decodes one bounded byte snapshot. Every integrated collection payload and label alignment was checked against its manifest, including held-out seed exclusion. |
| P3.1 | `73db8d7` → `2ff7616` | Fixed gripper target bounds exceeding the actuator range and added a generated-scene drift check. Added missing model-name and numeric-type checks. Claude re-reviewed the fix; focused spec tests passed. |
| P3.2 | `4d59ea9`, `6e2ae51` → `9b90061`, `35fda53` | Prevented evidence overwrite and added an analytical conflicting-target test for simultaneous IK weights. Focused IK tests passed. Legacy expert code remains separate. |
| P3.3 | `60c78ae` → `37ea936` | Reproduced and fixed a lost emergency-stop latch on invalid input, stale frames rewinding return-to-rest, and stationary-target overshoot. Regression tests and randomized command-limit probes passed after the fixes. |
| P3.4 | `c6de222`, `045e8d9` → `4628786`, `0ae776a`, `de588da` | Refreshed derived tool poses after integration, latched simulation faults closed, and compared motion-driven commands with a zero-offset control. Corrected evidence was appended without replacing old artifacts. Focused backend tests passed. |
| P1.3 | `3baa44b` | Ensemble indexing, weights, resets and old-checkpoint defaults passed review and focused tests. A legacy-style metadata probe retained open-loop execution. Every completed model and rollout passed the evidence checks described below. |

The integration portability fix `9d70654` explicitly compares MuJoCo enum
values as integers; its source review found no behavior change beyond portable
comparison.

## Evidence and limits

The QA lane used the middleware-built native extension for its focused checks;
it did not independently rebuild the native core. Full Windows build, pytest,
CTest, exact legacy-equivalence and hosted Ubuntu checks belong to the
integration verification logs. No full-suite result is inferred from Claude's
reviews.

The integration lane reported passing Windows pytest and CTest at source merge
`f3a7d16`. Its [Ubuntu run](https://github.com/mizuharaa/XERT/actions/runs/39652493894)
also passed the build, pytest, CTest and exact legacy-equivalence checks.
Final evidence and documentation commits receive their own integration checks.

For the completed [P1.3 experiment](../../results/p1.3/act_v1/outcomes.json),
QA checked each model's weights, metadata and trainer hashes; every corrective
training payload and fingerprint; normalization recomputed from training episodes
only; and every raw evaluation payload, terminal flag, outcome and artifact link.
Initial-state hashes computed from the arrays were distinct from demonstration
states. Per-model Wilson intervals agreed with the ledger. The ledger row IDs are
`d35d3a3e-8f9b-4a9e-a6e4-d5283d807197`,
`8ab93c8e-e880-49c2-a7f6-2229615d92e2` and
`c98903f8-a108-4e12-8b59-b1091ae48467`.
These models share the held-out initial states; their rollouts must not be pooled
as independent initial-state samples. QA validated recorded evidence and did not
retrain these models or independently rerun their full learned-policy evaluation.

P3.2 evidence covers local reachable position targets, not global convergence,
collision avoidance or dynamic tracking. P3.4 proves shared scripted simulation
execution; it does not measure human retargeting quality. Safety limits apply
to scheduled joint commands, not measured motor motion or certified protection.
The current safety event history remains in memory; a long-running runtime must
persist or bound it.

P1.3 uses a newer-prediction-favored ensemble, unlike the older-prediction-favored
weights in the [upstream ACT implementation](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py#L229-L238).
Its predeclared chunk selection compares validation errors over different future
horizons, not executed ensemble success. Its result is bounded by the declared
training budget and seed-dependent training/validation split. Evaluation rows
retain artifact-related dirty flags; implementation files were unchanged during
measurement against `3baa44b`.
These limits do not justify changing a running experiment after seeing outcomes.

Human demonstration collection, the quality-gate experiment, webcam latency,
LeRobot round trips and hardware operation are outside this acceptance.
