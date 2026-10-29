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
| P1.1 | Complete | [Diagnosis](learn/P1.1.md), saved predictions and joint-error plots |
| P1.2 | Complete | [Recovery collection](learn/P1.2.md), immutable executed actions and corrective labels |
| P1.3 | Complete | [ACT experiment](learn/P1.3.md), [ledger-generated held-out results](../results/p1.3/act_v1/report.md) |
| P1.4 | Complete | [Official ACT comparison](learn/P1.4.md), [ledger-generated results](../results/p1.4/official_act_v1/comparison.md), [independent QA](reviews/OFFICIAL_ACT_QA.md) |
| P2.1 | Complete | [Lossless LeRobot export](learn/P2.1.md), real reader round trip and ledger-backed dataset |
| P2.2 | Complete | [Direct LeRobot recording](learn/P2.2.md), timestamp/mask preservation and [scripted round-trip QA](reviews/RECORDER_QA.md) |
| P3.1 | Complete | [Validated embodiment specifications](learn/P3.1.md) |
| P3.2 | Complete | [Generalized IK](learn/P3.2.md), reachable-target measurements in the ledger |
| P3.3 | Complete | [Command limits, watchdog and emergency stop](learn/P3.3.md), regression checks |
| P3.4 | Complete | [Shared simulation backend](learn/P3.4.md), three scripted embodiments and configuration-only extension test |
| P3.5 | Partial | [Reviewed webcam/runtime software](learn/P3.5.md); live calibration and measured webcam latency still required |
| P3.6 | Partial | [Reviewed clip recorder and retarget measurement](learn/P3.6.md); a recorded human clip and its ledger rows still required |
| P4.1 | Partial | [Reviewed bench teleop collection](learn/P4.1.md); live webcam check on the `xlerobot_left` bench still required |
| P4.2 | Requires human demonstration capture | Scripted data cannot substitute |
| P5.1 | Partial | [Reviewed per-episode gate metrics](learn/P5.1.md); distributions on collected demonstrations still required |
| P5.2 | Pending | Thresholds committed before gated training |
| P5.3 | Pending | All/gated/random-control experiment and report |

Accepted slices: P0.1–P0.3, P1.1–P1.4, P2.1–P2.2 and P3.1–P3.4.
These are thirteen of the twenty required simulation slices. This counts accepted
slices, not equal engineering effort or a finished end-to-end MVP.

The learned-policy result applies to the existing XLeRobot-arm MuJoCo task.
SO101, fixed G1 upper body and LEAP hand share the new scripted middleware path;
the learning experiment does not establish learned cross-robot transfer.
See the [interview notes](learn/INDEX.md), [original simulation QA](reviews/SIM_MVP_QA.md),
[export/runtime QA](reviews/DATA_RUNTIME_QA.md), [recorder QA](reviews/RECORDER_QA.md),
and [official ACT QA](reviews/OFFICIAL_ACT_QA.md).
The integrated source at `4a64bb4` passed both the
[Linux build and legacy equivalence checks](https://github.com/mizuharaa/XERT/actions/runs/39671141509)
and [real optional-stack checks](https://github.com/mizuharaa/XERT/actions/runs/39671141712).
After merging the measured artifacts, Windows checks also passed for the base
environment, isolated LeRobot/ACT/recorder environment, and isolated vision environment.
The historical ledger prefix and every recorded artifact checksum were preserved.

The remaining slices retain their original acceptance criteria. P3.5 needs a real
neutral-pose calibration and live webcam measurements for each embodiment. P3.6
needs recorded human motion, P4.1 needs a live webcam session on the bench, and
P4.2 needs actual human demonstrations before the quality-gate experiment can be
accepted. P3.6, P4.1 and P5.1 software passed independent review; none of them
has produced a measurement yet. Scripted recorder checks satisfy
neither requirement. No hardware commands or human-data claims are authorized
by this simulation checkpoint.
