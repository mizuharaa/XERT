# Simulation MVP acceptance tracker

The current work order permits parallel lanes in separate worktrees. Within each
lane, dependent slices still wait for their prerequisites. Each implementation
slice keeps its own branch, tests, learning note and independent review.

The halfway checkpoint counts the required simulation slices in Phases 0–5:
twenty slices, excluding optional cameras (P2.3), hardware (Phase 6) and final
packaging (Phase 7). Ten must satisfy their acceptance criteria. This is a work
checklist, not a robotics performance metric; partial implementations do not count.

| Slice | Acceptance status | Evidence |
| --- | --- | --- |
| P0.1 | Complete | `STATE_OF_REPO.md`, `learn/P0.1.md` |
| P0.2 | Complete | `learn/P0.2.md`, append-only results ledger |
| P0.3 | Complete | `learn/P0.3.md`, linked Ubuntu CI and Windows checks |
| P1.1 | In progress | Single-episode diagnosis and evidence plots |
| P1.2 | Pending | Corrective demonstration collection and provenance |
| P1.3 | Pending | ACT sweep, training seeds and held-out evaluations |
| P1.4 | Pending | Official LeRobot ACT comparison |
| P2.1 | Pending | LeRobot export and load round trip |
| P2.2 | Pending | Direct recording with timestamps and safety events |
| P3.1 | In progress | Validated embodiment specifications |
| P3.2 | Pending | Generalized IK and measured reachable-target errors |
| P3.3 | Pending | Limits, watchdog and emergency-stop tests |
| P3.4 | Pending | Three embodiments driven through shared code |
| P3.5 | Pending | Webcam input, runtime and measured latency |
| P3.6 | Pending | Retarget errors on a recorded human clip |
| P4.1 | Pending | Teleoperation collection UI |
| P4.2 | Requires human demonstration capture | Scripted data cannot substitute |
| P5.1 | Pending | Quality distributions on collected demonstrations |
| P5.2 | Pending | Thresholds committed before gated training |
| P5.3 | Pending | All/gated/random-control experiment and report |

Planned halfway checkpoint: P0.1–P0.3, P1.1–P1.3 and P3.1–P3.4.
The remaining slices retain their original acceptance criteria. No hardware
commands or human-data claims are authorized by this simulation checkpoint.
