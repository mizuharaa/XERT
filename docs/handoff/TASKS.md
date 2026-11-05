# XERT task tracker (SO-101 track)

Plan: `PLAN.md` (local, not committed). Codex approved it on 2026-11-02 after 3 rounds; findings R1–R8 were all accepted.
Owner rules: "Daniel's portfolio rules" in `~/AGENTS.md` and `~/.claude/CLAUDE.md`. CORE logic is left as `TODO(daniel):`.
Each slice has its own handoff in this folder: what changed, why, how to verify, 3 interview questions, and its TODO(daniel) list.

| Slice | What | Branch | Status | Handoff |
| --- | --- | --- | --- | --- |
| R1 | Ubuntu 24.04 CMake + CI (CTest, pytest, equivalence) | done in P0.3 | Already done: `linux.yml` on every push | [P0.3 note](../learn/P0.3.md) |
| E | ROS 2 Jazzy `xert_ros` scaffold, URDF, RViz, launch, QoS README | `feat/e-xert-ros` | Merged; CI builds and tests on Jazzy | [E-xert-ros.md](E-xert-ros.md) |
| C | `xert export-lerobot` / `import-lerobot` / `to-store`, round trips, camera keys | `feat/c-lerobot-io` | Merged | [C-lerobot-io.md](C-lerobot-io.md), [CAMERA_KEYS](../CAMERA_KEYS.md) |
| B | SO-101 as a second bench robot, `--robot` CLI, timing contract | `feat/b-so101-bench` | Merged | [B-so101-bench.md](B-so101-bench.md) |
| D | Real-vs-sim replay: per-joint plots + RMSE (sysid step 1) | `feat/d-replay-real` | Merged | [D-replay-real.md](D-replay-real.md) |
| B2 | SO-101 export schema + cameras (wrist = Menagerie `wrist_cam`) | `feat/b2-so101-export` | Merged | [B2-so101-export.md](B2-so101-export.md) |
| F1 | `xert map-from-calibration`: fills the SO-101 joint map from a LeRobot calibration + ≥2 still-pose recordings (files only) | `feat/f1-map-from-calibration` | Merged | [F1-map-from-calibration.md](F1-map-from-calibration.md) |
| env | Real-arm environment: LeRobot `[feetech]` extra (STS3215 SDK, pyserial, deepdiff) in the teleop locks; `python -m scripts.hw_env_check` readiness report | on integration | Done 2026-11-04; teleop CI job green with the new lock | README "Python environments" |
| G1 | Image-ACT train/evaluate scaffold on rendered `scene`/`wrist` (settings, loss and selection are TODO(daniel)) | `feat/g1-image-act` | Merged | [G1-image-act.md](G1-image-act.md) |
| H | Block-stacking task bench: second task on both arms, Python-scored success with release of both blocks, shared `TaskBackend`, `task_id` through data and ledger (stacking expert is TODO(daniel)) | `feat/h-stack-bench` | Merged (Codex plan review 3 rounds; QA accepted; one intermediate commit, a51ce11, was red and fixed next commit; per-commit sweep unverified past 93fb8e7) | [H-stack-bench.md](H-stack-bench.md) |

## Daniel's TODO(daniel) queue
These fill in as slices land. Each entry gives file:line and the test that defines "done".
- **E, ROS control loop:**
  - `ros/xert_ros/include/xert_ros/xert_node.hpp`: `control_step()`, `on_command()`, `read_state()`. Their contracts are in the header comments; the line numbers are in [E-xert-ros.md](E-xert-ros.md).
  - Done = the two `DISABLED_` gtests in `ros/xert_ros/test/test_xert_node.cpp` are enabled and pass in `ros.yml`.
- **B, SO-101 scripted expert** (`xle_workbench/so101_expert.py`):
  - `so101_expert_targets()` at :11 (waypoints) and `so101_expert_action()` at :30 (IK targets, gains, rate limits).
  - Done = the three strict xfails in `tests/test_so101_expert.py` (:43 pre-grasp within 5 mm, :54 success on seeds 0..8, :62 held-out success) pass with their markers removed.
  - The 5 mm pre-grasp test is weak: a naive hover passes it. The success tests are the real bar.
- **H, stacking expert** (`xert/tasks/stack_expert.py`): `stack_expert_targets()` at :10 and `stack_expert_action()` at :29 (IK waypoints, gains, rate limits), for both arms.
  - Done = the strict xfails in `tests/test_stack_expert.py` (:28 seeds 0..8, :36 held-out 10000..10008) pass with their markers removed.
  - Success needs both blocks released and resting on the base/table for 12 ticks; a held base or a held block never counts. `info["object_in_robot_contact"]` reports `block` only (native meaning), by default.
- **G1, image-ACT training choices** (`xert/eval/image_act.py`): `image_act_settings()`, `make_optimizer()`, `training_step()`, `select_epoch()`.
  - Done = the two strict xfails in `tests/test_image_act.py` pass with their markers removed: valid settings, then a checkpoint that evaluates (1 seed in CI; the real ≥30-seed evaluation is a manual run).
  - Ledger rows need human training data, ≥30 held-out seeds, a clean tree and an unused experiment tag. Only the selected epoch is saved.
- **D, actuator parameter fitting** (system identification): `fit_actuator_params()` at `xert/eval/replay_real.py:269`.
  - Done = the strict xfail at `tests/test_replay_real.py:221` passes with its marker removed: on a kp×0.5 synthetic recording, held-out RMSE < 0.1 × baseline and fitted kp within 10% of 499.11.
  - Results and any fit are tied to the replay dt (QA measured 0.0016–0.0118 rad RMSE between runs that differ only in timestep).
- **Notes from B:**
  - The base placement (`SO101_BASE`) was chosen for position reach only; a top-down grasp approach was not measured. Move it if your expert needs to.
  - Migration: core edits change `implementation_sha256`, so P1.2–P2.1 stored episodes and models refuse replay/load on the new core and reproduce at their recorded `git_commit` (see the B handoff).

## Owner decisions (2026-11-03: Daniel accepted every recommended default)
| # | Decision | Applied as | Where |
| --- | --- | --- | --- |
| 1 | Project licence | Apache-2.0 (same as the vendored SO-101 and Menagerie assets) | `LICENSE`, `ros/xert_ros/package.xml` |
| 2 | Real camera names | `scene` and `wrist`, identical to sim | `docs/CAMERA_KEYS.md` |
| 3 | Block/tray state in `to-store` | Appended to observations when the import has `environment_state` | slice D fix round |
| 4 | SO-101 contact solver settings | Keep Menagerie's (10/20/10), tuned with that model | B scene provenance |
| 5 | Trust for `replay-real --ledger` | Record the LeRobot calibration file sha256 and robot id at import; ledger rows require it | slice D fix round |
| 6 | Replay start velocity | Assume rest, but flag episodes whose first frames are already moving | slice D fix round |
| - | Joint-limit tolerance in replay | 1e-6 rad, the same as import | slice D fix round |

Already defaulted earlier and unchanged:
- P5.3 pooled non-overlap rule and C redrawn per seed
- teleop Esc saves the episode, then quits
- lossless PNG images
- `--spec` selects the teleop arm
- earlier agent-written CORE code is kept as tutoring material
- `--motor-units degrees` asserts LeRobot's `use_degrees=True`

## Real-arm first steps (F1 merged)
1. `lerobot-calibrate` the follower, then the leader (`--teleop.*`). At the "middle of range" prompt, put `wrist_roll` at sim roll 0.
2. Record ≥2 short `lerobot-record` episodes holding the arm still in known poses (the bench reset pose plus one with every joint moved ≥ its minimum spread; table in the F1 handoff), cameras named `scene`/`wrist`.
3. `python -m xert map-from-calibration --calibration <robots/so_follower/<id>.json> --poses poses.json --output <fresh map>`; read it, then copy it into `configs/hardware/so101_joint_map.yaml` by hand. `max_speed`/`max_accel` stay yours to choose.
4. Then `import-lerobot --motor-units degrees --calibration …` and `replay-real` accept real recordings.

## Waiting on Daniel (hands-on)
P3.5 live webcam runs, P3.6 clip, P4.1 live check, P4.2 demos, P5.2 thresholds, P5.3 run, demo video, SO-101 bring-up (P6.1/P6.2 per `docs/learn/P6.md`).
