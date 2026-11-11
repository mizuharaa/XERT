# Slice K: drawer opening, the third bench task and the first articulated object

Branch `feat/k-drawer-bench`, from `integration/sim-mvp` @ 9c62e9b. Simulation only. The work is environment code,
plumbing, tests and docs. **No drawer result and no number exists**: the scripted drawer expert is a
`TODO(daniel)` stub, no policy was trained, and no ledger row was written. No hardware, no camera device, no new
dependency. The plan is `PLAN.md` section K (Codex review findings K-01, K-02).

## What changed and why

The goal is a task on an articulated object, so the pipeline is shown to work beyond free blocks. Everything
follows slice H: the same recorder, cameras, quality gate, P5.3 harness and image-ACT plumbing, reached through
`TaskBackend` and `task_id`.

- **Scene** (`xert/tasks/drawer.py`, `drawer_spec` / `drawer_scene`).
  - The base is the robot's pick-and-place bench MjSpec (`cameras._bench_spec`, as `stack_spec`). The free `block`
    and the static `tray` are deleted in memory: they are not part of this task, and a free block would need its
    pose recorded.
  - A `cabinet` body welded to the world on the table top: five box geoms (bottom, top, two sides, back), outer
    size 0.16 deep × 0.14 wide × 0.09 high, 6 mm walls.
  - A `drawer` body (child of the cabinet) on the slide joint `drawer_slide`, range 0 to `DRAWER_TRAVEL` = 0.12 m
    (0 = closed). Five box panels, 4 mm clearance to the shell on every side, and a cabinet–drawer contact exclude,
    so only the joint guides it. 0.14 m deep, so it stays 2 cm inside the shell at full travel.
  - Slide dynamics are MuJoCo's own defaults, written out: damping 0, frictionloss 0, armature 0. The limit's
    solref is tightened to (0.004, 1) (see "Solver settings"). Density is MuJoCo's default 1000 kg/m³: the drawer
    body's `body_subtreemass` is 0.2617 kg in total (panels, handle and posts; measured). None of this is a measured
    cabinet.
  - `handle`: a horizontal cylinder, radius 10 mm, half-length 35 mm, its axis along the drawer front, 35 mm proud
    of the face, tied to the face by two 8 mm posts (drawer geoms, so touching them counts as holding).
  - Sliding friction 1.0 on every geom, as the bench sets.
  - Built from primitives: no asset, no licence. `sim.py`, `cpp/` and `assets/` are untouched, so
    `implementation_sha256`, the native contract hash and the vendored checksums do not change.
  - Fingerprint: sha256 of {bench fingerprint, removed bodies, placement, shell, clearance, drawer depth, handle,
    travel, slide dynamics, solver options, provenance notes}.
- **Placement, fixed per robot (review K-01)** (`PLACEMENT`; cabinet frame on the table top, +x out of the front).
  - `xlerobot_left`: (0.35, −0.03, 0.08), yaw π. The front faces the arm (−x); the drawer is pulled toward it.
  - `so101`: (0.26, 0.12, 0.08), yaw −π/2. The front faces −y; the drawer is pulled sideways across the arm's
    front. Facing the SO-101 (+x) would put the cabinet's back off the table (x < 0.12 m) or the open handle over
    the arm base (x = 0.42 m).
  - The cabinet pose is in the fingerprint, so the whole scene is rebuilt from the arm joints plus `drawer.pos`.
    Per-episode cabinet placement (a mocap-attached cabinet and a recorded transform) is a later slice.
- **Seeded variation** (`initial_opening(seed)`): `drawer.pos` uniform in [0, 0.02] m at reset. Every start is
  below `OPEN_M`, so no episode begins already successful (tested on 200 seeds).
- **State names** (`xert/tasks/__init__.py`): `environment_names("drawer")` = `drawer.pos`, `handle.x/y/z`,
  `grasp.x/y/z`, `time_fraction`. One function, generated from `JOINTS` (scalar joint-valued object state),
  `BODIES` (free bodies) and `POINTS` (tracked points), so pick_place and stack keep their exact lists (tested).
- **Success, in Python** (`DrawerTask`, scored on `MjData` after every 25 Hz tick). A tick counts when all hold:
  - (a) `drawer.pos ≥ OPEN_M` = 0.08 m, two thirds of the travel (chosen here);
  - (b) released: no robot contact on the `drawer` body, whose geoms include the handle and its posts. "Robot"
    means any body other than the world (table, floor) and the cabinet, as `cpp/src/env.cpp` `Env::step` counts
    `held` and the stack rule's (b) mirrors (`env.cpp:118` `!held`);
  - (c) still: |slide speed| < `SLIDE_SPEED` = 0.01 m/s (`xert/tasks/stack.py` `LINEAR_SPEED`).

  Success is `SETTLE_TICKS` = 12 consecutive such ticks (`env.cpp:120` `stable_steps >= 12`). Any other tick,
  including a dip below `OPEN_M` or a touch, resets the count. The only failure is the `HORIZON` = 600 ticks
  (`xle_workbench/sim.py:145`), reported as `timeout`. `lifted` is always False (nothing is lifted). Every
  DrawerBackend `metadata()` carries the constants and their sources under `success_rule`.
- **Backend** (`xert/backends/drawer.py`, `DrawerBackend`): `MuJoCoBackend` on the drawer scene with the bench
  reset pose, float32 observations, `send_targets(q, None)` stamped with sim time. `info` has H's keys plus
  `drawer_open_fraction` (`object_position` is the handle centre). `task_backend(spec, seed, "drawer")`.
- **Consumers.**
  - `xert teleop --task drawer` (`runtime/cli.py`, choices from `TASKS`); `TASK_TEXT` "Pull the drawer open by
    its handle, then let go"; success provenance per task. The `--viewer` path copies the whole `qpos`, so the
    slide joint is shown with no change.
  - Ledger and image-ACT check `task_id` against `TASKS`.
  - `xle_workbench collect|evaluate --task drawer` refuses with "drawer expert not tuned (TODO(daniel))" before
    creating anything.
  - `xert dagger` stays pick-and-place only (review K-02): no task option, and `init` refuses a collection whose
    episodes name another `task_id` (tested). See "What a task-aware DAgger would need".
  - quality and `resume_numbers` already key on `task_id`; nothing changed there (quality on a drawer recording
    is tested).
- **Cameras** (`xert/data/cameras.py`).
  - `TASK_CAMERAS["drawer"]` overrides only the scene camera per robot, chosen by eye in simulation to see the
    cabinet front: xlerobot_left pos (0.02, −0.4, 0.45) → target (0.25, −0.02, 0.17); so101 pos (0.1, −0.3, 0.33)
    → target (0.27, 0.05, 0.12); both fovy 45. Wrist cameras stay the robot's. `cameras(robot, task_id)` is the one
    lookup; `CAMERAS` and the XLeRobot pin are unchanged.
  - `CameraRenderer.render(joints, objects)` also takes scalar joint states by name (`{"drawer.pos": ...}`).
    `_columns`, `row_state` and `scene_objects` take them the same way, so offline `render-cameras` and live
    `image_act.live_frame` share one path. Reconstruction checks for the drawer: `handle`, `grasp`, `drawer.pos`,
    tolerance 1e-5.

## Handle and jaw sizing, reach (measured this session)

`.\.venv\Scripts\python.exe scripts\drawer_reach.py` (exit 0):

- **Reach.** Position-only `xert.retarget.ik.solve` from the bench reset pose to the handle centre at 0, 50 and 100 %
  travel, then FK. Approach orientation is not measured: a pose that reaches the handle centre may hold the jaws at
  an angle that cannot close on it.

  | robot | handle at 0 / 50 / 100 % (m) | worst FK error | arm–shell contacts at the IK poses |
  |---|---|---|---|
  | xlerobot_left | (0.315, −0.03, 0.125) / (0.255, …) / (0.195, …) | 1.0e-16 m | 0 |
  | so101 | (0.26, 0.085, 0.125) / (0.26, 0.025, …) / (0.26, −0.035, …) | 1.4e-16 m | 0 |

  Tolerance 1 mm. A first XLeRobot placement 5 cm closer to the arm failed at 100 % travel (0.375 m FK error); it
  was moved before this measurement.
- **Jaws.** A probe cylinder of the handle's size, its axis on the gripper hinge axis, on a 2 mm grid (±4 cm) in
  the closing plane through the grasp site, arm at the reset pose. A position counts when the probe touches
  neither jaw at the gripper's open actuator limit, and the moving jaw overlaps it at the closed limit while the
  fixed jaw does not. Static geometry, no forces.

  | robot | open / closed (rad) | graspable positions | nearest to grasp site | open clearance fixed / moving | closed overlap |
  |---|---|---|---|---|---|
  | xlerobot_left | 1.4 / −0.2 | 146 | 8.9 mm | 0.88 / 48.8 mm | 1.46 mm |
  | so101 | 1.745 / −0.175 | 227 | 10.8 mm | 0.02 / 64.9 mm | 6.55 mm |

  With a 6 mm radius, an earlier version of this check (probe resting on the fixed jaw next to the grasp site) left
  the closed XLeRobot jaw 6.4 mm short, so the radius was raised to 10 mm; the grid check was not run at 6 mm. The
  margin that matters is the closed overlap: 1.46 mm on XLeRobot is small.

## Solver settings (per bench, unchanged)

| robot | timestep | solver | iterations | cone | impratio | integrator |
|---|---|---|---|---|---|---|
| xlerobot_left | 0.002 s | Newton (2) | 100 | elliptic (1) | 1.0 | implicitfast (3) |
| so101 | 0.002 s | Newton (2) | 10 | elliptic (1) | 10.0 | implicitfast (3) |

Recorded in every scene's composition (`solver`). The slide limit is a soft constraint with solref (0.004, 1),
2× the timestep, set in `SLIDE` (so it is in the composition and the fingerprint). Under a constant 2 N pull from
closed, arm held at reset, 4 s (`tests/test_drawer.py::test_a_2n_pull_from_closed_overshoots_the_stop_by_at_most_4mm`):

| slide limit solref | peak `drawer.pos` | final under load |
|---|---|---|
| (0.02, 1), MuJoCo default | 0.13002 m (+10.02 mm), 1.36 m/s into the stop | 0.12030 m |
| (0.01, 1) | 0.12501 m (+5.01 mm) | 0.12008 m |
| (0.004, 1), used | 0.12244 m (+2.44 mm) | 0.12001 m |

Same numbers on both arms (my probe, matching QA's). `info["drawer_open_fraction"]` is `drawer.pos / 0.12`,
not clamped: it can leave 0..1. Against the stop under the 2 N pull it reads 1.0001 (1.0025 with the old default)
and peaks at 1.020 during the impact.

## How to verify (commands run on this branch; numbers below are mine, from this session)

```
.\.venv\Scripts\python.exe -m pytest -q
ctest --test-dir build -C Release
.\.venv\Scripts\python.exe scripts\equivalence_check.py
.\.venv\Scripts\python.exe scripts\drawer_reach.py
C:\Users\Khang\tmp\reach-public\build\venv-teleop\Scripts\python.exe -m pytest -q tests/test_cameras.py tests/test_so101_cameras.py tests/test_teleop.py tests/test_recorder.py tests/test_quality.py tests/test_stack.py tests/test_drawer.py tests/test_drawer_expert.py
C:\Users\Khang\tmp\reach-public\build\venv-teleop\Scripts\python.exe -m pytest -q tests/test_gate_experiment.py tests/test_image_act.py tests/test_stack_pipeline.py tests/test_drawer_pipeline.py tests/test_dagger.py
```

- Base pytest: 263 passed, 72 skipped, 15 xfailed, 8 subtests (147 s). Baseline at 9c62e9b: 240 / 69 / 11 / 8.
- ctest: 2/2. Equivalence: 16 seeds identical, max observation difference 0.0.
- Combined venv-teleop, run in two chunks to keep one process under ~5 minutes: 75 passed, 1 skipped, 4 xfailed
  (116 s) and 60 passed, 4 xfailed (315 s); together 135 / 1 / 8. Baseline for the 10 listed files: 109 / 1 / 4.
  The XLeRobot camera pin and every stack test are in it and green.
- `tests/test_drawer.py`: scene composition, vendored checksums, fingerprint per placement, slide direction, reach
  and jaw sizing on both arms, every check on hand-built states, the negative tests, seeding, names, the backend.
- `tests/test_drawer_pipeline.py`: the K-01 reconstruction (below); the gate harness over a full 600-tick episode;
  ledger and teleop CLI accept `drawer`; a 5-frame scripted-hand teleop recording with `next.success` and the
  8-name `environment_state`; render-cameras, export and quality on it; image-ACT `describe` of the rendered
  dataset and `evaluate` with a constant stub over 600 ticks of live 64 px renders.
- **K-01 result**, both arms, seeds 0 and 1 (different initial openings): offline and `live_frame` rebuild handle,
  grasp and `drawer.pos` within 1e-5 of each backend's scene, the cabinet position equals the backend's, and the
  scene and wrist pixels equal a direct render of that backend's `MjData` (MSAA off for the comparison, as in H).
  The two seeds give different handle positions and different scene images.
- Negative tests (both arms): gripper closed and pressed into the handle with the drawer open at 0.1 m for 20 ticks,
  no success; arm back at reset, success on clean tick 12. Grasp site on the front panel 2.5 cm below the handle
  (contact on panel geoms, not the handle), no success for 20 ticks; clear, success on tick 12. A dip to 1 mm below
  `OPEN_M` at tick 12 resets the count.

## Three interview questions

1. *Why require release for a drawer?* Without it, the gripper holding the handle at 8 cm counts as "open", but a
   drawer that only stays open while held has not been opened; the test holds it open for 20 ticks with no success,
   then releases and gets success on clean tick 12.
2. *Why fix the cabinet per robot instead of randomising it?* A fixed cabinet makes the slide value the whole object
   state, so render-cameras, image-ACT's live frames and the viewer rebuild the scene without per-episode model
   changes; the seed varies the initial opening instead.
3. *How do you know the arm can do the task without an expert?* I don't: the IK reaches the handle centre at 0, 50
   and 100 % travel on both arms with 1e-16 m FK error, and a 20 mm handle fits the open jaws and is overlapped
   by the closed jaw, but orientation and contact forces are not measured.

## TODO(daniel) (CORE)

- `xert/tasks/drawer_expert.py:10` `drawer_expert_targets(step, handle, pull_axis)`: approach, grasp, pull and
  release schedule.
- `xert/tasks/drawer_expert.py:31` `drawer_expert_action(backend)`: joint targets for one tick, approach
  orientation and rate limit included.
- Done means these pass, after which you delete the markers: `tests/test_drawer_expert.py:28`
  `test_scripted_drawer_success_on_seeds_0_to_8` and `:36` `test_scripted_drawer_success_on_held_out_seeds`
  (10000..10008), both for xlerobot_left and so101.
- The done tests' rollout copies `backend.data.qpos`/`qvel` before each expert call and fails with
  `RuntimeError("expert mutated backend.data")` if either changed. RuntimeError is not an expected failure, so an
  expert that writes `backend.data` fails even while the xfail markers are on. Same guard in the stack done tests.

## Owner decisions applied

- Handle radius stays 10 mm. QA's grip probe (4 s under a 2 N pull): at r = 10, 12 and 13 mm the XLeRobot jaw
  creeps about the same, ~16.8 mm in the jaw-opening direction and 12.5 mm along the axis. The limit is the XLeRobot
  gripper actuator (kp 5, forcerange ±0.3 N·m) against so101's (kp 998, ±2.94 N·m); so101 creeps 1.2–1.9 mm. A
  frictionless 0.262 kg slide needs far less than 2 N.
- Grasp strategy (closed grip or hook-pull) stays with Daniel's expert.
- Slide limit solref tightened to (0.004, 1); table under "Solver settings".
- A shared StackBackend/DrawerBackend base is deferred until a third Python-scored task exists.

## What is NOT shown

- No drawer success rate, scripted, teleop or learned. No expert exists.
- The drawer's damping, friction loss, armature and mass are MuJoCo defaults, not a measured cabinet.
- Reach is position-only; approach orientation and arm–cabinet collisions along a path are not measured. The jaw
  check is static geometry, not a simulated grasp.
- `scripts/drawer_reach.py` counts cabinet-shell contacts at the 3 IK poses only. QA's probe, 13 samples along the
  travel: xlerobot_left has 0 shell and 0 non-handle drawer contacts, its hinge axis 5.5–8.8° off the handle axis;
  so101's fixed jaw (`fixed_jaw_box2`) is inside `drawer_front` at 92 % and 100 % travel and its hinge axis is
  62–90° off the handle axis (position-only IK crosses the jaws over the handle instead of wrapping them).
- No human data. Every recording in the tests is labelled `scripted`.

## What a task-aware DAgger would need

`xert dagger` is bound to pick-and-place on XLeRobot by its contracts (`dagger.py` `_bench`, `init`, and
`pipeline.rollout` driving the native `reset/step/expert_action`). For the drawer it would need: a rollout over
`TaskBackend` (`task_backend(spec, seed, task_id)`), an expert callable per task (`drawer_expert_action`, once it
exists), an observation contract built from `environment_names(task_id)` instead of the native vector, a
contract hash for Python-scored tasks (scene fingerprint + rule constants + task module source) in place of
`implementation_sha256`, `task_id` in the run manifest, the Store metadata and its validation, and the P2.1
export/import keeping `task_id` and the two action streams.

## Found along the way

- `teleop.run_session` prints the handle position under the label `block=` (it prints `info["object_position"]`
  for every task). Cosmetic; not changed.
- `success_rule` provenance is only written by the webcam `execute` path, not by `run_session`, so scripted-hand
  recordings never carry it (true for stack too).
- `DrawerBackend` and `StackBackend` share most of their code. A common base is a small refactor of slice H's
  code; not done here.

## For integration (not edited here)

- README SO-101 track: one line, "`xert teleop --task drawer` records the drawer-opening task (cabinet fixed per
  robot); the scripted drawer expert is TODO(daniel)".
- TASKS.md: add slice K, branch `feat/k-drawer-bench`, owner Claude (builder), status built, awaiting QA.
- MVP_PROGRESS: third task (articulated) scaffolded; no result.
