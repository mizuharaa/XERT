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
| K | Drawer-opening bench: a third task on an articulated object (slide joint + handle), Python-scored success with release, same `TaskBackend`/`task_id` pattern as H (drawer expert is TODO(daniel)) | `feat/k-drawer-bench` | Merged at 36f349e (Codex plan approved round 2; QA fix-first: done-test mutation guard, slide solref (0.004, 1), handoff numbers; all five gates green on the merge) | [K-drawer-bench.md](K-drawer-bench.md) |
| M | P6.2 blockers before any motor moves: (a) e-stop reaches the hardware backend (`hold`), (b) `SafetyController.observe` tracks the applied command on the hardware path, (c) one read-only start pose via `xert hw-read` + `hw-dry-run --seed-from`; braking law untouched | `feat/m-p6-blockers` | Building (plan approved by Codex on round 3 after M-01..M-06) | `M-p6-blockers.md` |
| L | `xert add-pose`: Cartesian pose columns (`observation.state.cartesian`, `action.cartesian`, 7-D DROID/Praxis layout) for simulation exports and recordings, FK of the grasp site in the fixed base frame; hardware data deferred to L2 | `feat/l-add-pose` | Merged at 5191416 (plan approved by Codex on round 2 after L-01..L-06; simulation-only, order export -> add-pose -> render-cameras; gripper 0 = open and xlerobot z-up frame applied as owner defaults, 15 Hz and pose point still open) | [L-cartesian-pose.md](L-cartesian-pose.md) |
| M | P6.2 software blockers: e-stop `hold` in the hardware backend (a), `SafetyController.observe` of the applied command (b), read-only `xert hw-read` + `hw-dry-run --seed-from` start pose incl. the mapper's model (c, c'); no CORE change, fake bus only | `feat/m-p6-blockers` | Built; QA fix round applied; awaiting merge (plan approved by Codex on round 3 after M-01..M-06; QA FIX-FIRST on 8d82bd3 folded in: sim no-observe pin, saturation strictly beyond, LeRobot 0.6.1 gate, port closed after a failed handshake; 301 passed / 74 skipped / 14 xfailed base env, ctest 2/2, equivalence 0.0, LeRobot-env `test_hw_read.py` 18 passed) | [M-p6-blockers.md](M-p6-blockers.md) |
| J | DAgger scaffolding on the pick-and-place bench: rollout with expert relabelling in the native two-file Store format, atomic per-iteration aggregation, held-out evaluation (mixing rule, stopping rule and hyperparameters are TODO(daniel)) | `feat/j-dagger` | Merged (Codex plan approved round 3; QA accepted; 3 low findings fixed pre-merge) | [J-dagger.md](J-dagger.md) |
| I | ROS 2 bridge: `xert_bridge` publishes safety-checked `/xert/joint_commands` (stamp = generation time); `xert_sim_follower` drives MuJoCo and publishes `/joint_states` (not a controller; `control_step` stays Daniel's) | `feat/i-ros-bridge` | Merged; Jazzy CI green at 1ea035d after 6 runs (launch test end to end); QA accepted | [I-ros-bridge.md](I-ros-bridge.md) |
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
- **J, the DAgger loop** (`xert/eval/dagger.py`): `dagger_action()` at :43 (mixing rule), `should_stop()` at :58, `dagger_settings()` at :71 (P1.3-style training config + rollouts per iteration).
  - Done = the strict xfails at `tests/test_dagger.py:351` (rules exist, valid values) and `:364` (a 2-iteration run grows the aggregate by the rollout count) pass with markers removed. Performance targets are yours to pre-register; none is typed anywhere.
  - Signature note: `dagger_action` gets no RNG or episode seed, so randomized beta-mixing needs a signature change; `iterate` probes the rule once on zero actions, so it must be deterministic and stateless at that call.
  - Seed data must be recollected on the current core (`scripts/collect_recovery.py`); `init` refuses the checked-in P1.2 collection and any non-default bench config.
  - QA's extra interview questions: why train on `corrective.npz` while replay uses `episode.npz`; what the manifest looks like after two racing `iterate` processes; what the zero-action probe constrains.
- **I, owner decision on your ROS node:** `/xert/joint_commands` stamps are now the command's generation time; the safety layer upstream handles camera loss (hold → return → rest). The sim follower only HOLDS on transport loss (bounded braking, then constant target) and never returns to rest by itself. Your `control_step` stub's comment still says "return to rest 0.7 s after the last fresh command". Decide which your node does; both are documented in [I-ros-bridge.md](I-ros-bridge.md). To run the bridge against your node: swap the follower `Node` in `ros/xert_ros_bridge/launch/bridge.launch.py` for `xert_node` (one line, shown in the handoff).
- **H, stacking expert** (`xert/tasks/stack_expert.py`): `stack_expert_targets()` at :10 and `stack_expert_action()` at :29 (IK waypoints, gains, rate limits), for both arms.
  - Done = the strict xfails in `tests/test_stack_expert.py` (:28 seeds 0..8, :36 held-out 10000..10008) pass with their markers removed.
  - Success needs both blocks released and resting on the base/table for 12 ticks; a held base or a held block never counts. `info["object_in_robot_contact"]` reports `block` only (native meaning), by default.
- **G1, image-ACT training choices** (`xert/eval/image_act.py`): `image_act_settings()`, `make_optimizer()`, `training_step()`, `select_epoch()`.
  - Done = the two strict xfails in `tests/test_image_act.py` pass with their markers removed: valid settings, then a checkpoint that evaluates (1 seed in CI; the real ≥30-seed evaluation is a manual run).
  - Ledger rows need human training data, ≥30 held-out seeds, a clean tree and an unused experiment tag. Only the selected epoch is saved.
- **K, drawer expert** (`xert/tasks/drawer_expert.py`): `drawer_expert_targets()` at :10 and `drawer_expert_action()` at :31, for both arms.
  - Done = the strict xfails in `tests/test_drawer_expert.py` (:28 seeds 0..8, :36 held-out) pass with their markers removed. The rollout raises RuntimeError if the expert writes `backend.data` (the done tests cannot be passed by setting the slide joint).
  - Success needs drawer.pos >= 0.08 m, the handle released, slide speed < 0.01 m/s for 12 ticks. QA's grip probe: the XLeRobot gripper (kp 5, forcerange +/-0.3 N m) creeps ~17 mm over 4 s under a 2 N pull at any handle radius; so101 creeps 1-2 mm. Closed grip vs hook-pull is your call; details in the K handoff.
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
0. After calibrating (step 1), with your explicit OK to open the bus: `python -m xert hw-read --port <COMx>
   --calibration <robots/so_follower/<id>.json> --output <read.json>` (read-only: connect, one
   `sync_read("Present_Position")`, `disconnect(disable_torque=False)`). Compare `read.json` with the arm by hand;
   once the map exists (step 3), rerun with `--joint-map` and use `hw-dry-run --seed-from <read.json>`.
1. `lerobot-calibrate` the follower, then the leader (`--teleop.*`). At the "middle of range" prompt, put `wrist_roll` at sim roll 0.
2. Record ≥2 short `lerobot-record` episodes holding the arm still in known poses (the bench reset pose plus one with every joint moved ≥ its minimum spread; table in the F1 handoff), cameras named `scene`/`wrist`.
3. `python -m xert map-from-calibration --calibration <robots/so_follower/<id>.json> --poses poses.json --output <fresh map>`; read it, then copy it into `configs/hardware/so101_joint_map.yaml` by hand. `max_speed`/`max_accel` stay yours to choose.
4. Then `import-lerobot --motor-units degrees --calibration …` and `replay-real` accept real recordings.

## Stop point 2026-11-16: everything left needs the real SO-101 or Daniel
Agent-buildable work is exhausted at slice L. Scoped but NOT started, because its shape depends on real recordings:
- **L2, hardware pose path:** `add-pose` for `import-lerobot --motor-units degrees` outputs (radians via the P6 joint map, which is all TBD until P6.1 measures the arm). The hard part is keeping the real camera videos while appending the Cartesian columns: LeRobot v3 stores per-feature stats in `meta/episodes/*.parquet` and `meta/stats.json`, so appending features means Parquet + stats surgery, or re-encoding the videos. Decide after the first real recording whether (a) Cartesian columns are appended offline (L2), or (b) FK is logged at capture time by a LeRobot robot wrapper that records `observation.state.cartesian` directly, which needs the arm.
- **Owner decisions open:** 25 Hz -> 15 Hz for Praxis (resample vs re-record), pose point (XERT `grasp` site vs Praxis's Franka flange), `xert hw-read` needs an explicit OK to open the servo bus.

## Waiting on Daniel (hands-on)
P3.5 live webcam runs, P3.6 clip, P4.1 live check, P4.2 demos, P5.2 thresholds, P5.3 run, demo video, SO-101 bring-up (P6.1/P6.2 per `docs/learn/P6.md`).
