# Slice H: block stacking, the second bench task

Branch `feat/h-stack-bench`, from `integration/sim-mvp` @ 237fa48. Simulation only. The work is environment code,
plumbing, tests and docs. **No stacking result and no number exists**: the scripted stacking expert is a
`TODO(daniel)` stub, no policy was trained, and no ledger row was written. No hardware, no camera device, no new
dependency.

## What changed and why

The goal is a task harder than pick-and-place, run by the recorder, quality gate, P5.3 harness, cameras and
image-ACT scaffold without rewriting them. The plan is `PLAN.md` section H (Codex review findings H-01..H-07).

- **Task identity.** `xert/tasks/__init__.py` defines `TASKS = ("pick_place", "stack")` and
  `environment_names(task_id)`.
  - `environment_names` is the one names list. Teleop records it as `observation.environment_state`, and
    render-cameras and image-ACT pose scenes from it.
  - For `pick_place` it equals the native bench's `observation_names[12:]` (tested).
  - The field is `task_id`, never `task`: `task` stays the LeRobot natural-language description.
- **Scene** (`xert/tasks/stack.py`, `stack_spec`/`stack_scene`).
  - The base is the robot's pick-and-place bench MjSpec, the same rebuild render-cameras uses (`cameras._bench_spec`).
    It keeps the table, the tray (static), the block and, for SO-101, `so101_scene()`'s base placement.
  - On top of that, one free `base_block` (blue) that copies the bench block's geom size, mass, condim and friction.
  - Composition happens in memory. `sim.py`, `so101_expert.py`, `cpp/` and `assets/` are byte-identical to 237fa48,
    so `implementation_sha256`, the native contract hash and the vendored checksums do not change.
  - Model fingerprint: sha256 of {pick-and-place bench fingerprint, base_block definition, spawn rectangles,
    rest z}.
  - Seeded spawns (`spawn(seed)`):
    - `block` in [[.225, -.045], [.255, -.015]], the bench's own `object_position_bounds`;
    - `base_block` in [[.225, .015], [.255, .04]];
    - the strips are 3 cm apart in y, so the 16 mm footprints never overlap (tested on 200 seeds).
- **Reach** (`scripts/stack_reach.py`, the slice B method).
  - Position-only `xert.retarget.ik.solve` from the bench reset pose, then FK, at the 4 corners of both rectangles.
  - Heights: block centre and +6 / +7.5 cm; stacked centre (base top + half a block, z = .128 m) and +6 / +7.5 cm.
  - Measured worst FK error over 24 samples: **2.2e-16 m (xlerobot_left)** and **1.4e-16 m (so101)**, against a
    1 mm tolerance.
- **Success, in Python** (`StackTask`, scored on `MjData` after every 25 Hz tick). A tick is *stacked* when all of
  these hold:
  - (a) contact between `block` and `base_block`, with contact geoms resolved to bodies;
  - (b) no robot contact on `block`. "Robot" means any body other than the world (table, floor), the tray and
    base_block, the same definition as env.cpp's `held`. A held tick resets the count, so the robot touches the
    block at none of the window's ticks;
  - (c) |block bottom − base_block top| ≤ 5 mm, from the box corners, so a tilted block is measured correctly.
    The window is two-sided because MuJoCo's resting contact penetrates about 0.1 mm;
  - (d) xy centre offset ≤ half a block width (0.008 m, the geom's x half-size);
  - (e) block linear speed < 0.01 m/s and angular speed < 0.1 rad/s.

  Success is 12 consecutive stacked ticks. Failure is either block's centre below z = .04 m (`fell_off_table`), or
  the 600-tick horizon (`timeout_after_lift` / `timeout_no_lift`, lifted = block z > .145 m). Constants copied from
  the native bench: 12 ticks, .04 and .145 (`cpp/src/env.cpp` `Env::step`), .096 rest z (`Env::reset_state`),
  600 (`xle_workbench/sim.py:145`). Every StackBackend `metadata()` carries them under `success_rule`.
- **Shared backend contract** (review H-02).
  - `xert/backends/base.py` holds `TaskBackend` and `task_backend(spec, seed, task_id)`: one protocol, two
    implementations.
  - `BenchBackend` (pick_place, native C++ rule) now also exposes `task_id`, `config`, `limits`, `metadata()` and
    `observation()`. Its numbers are unchanged; `observation()` is still the native float32 vector.
  - `StackBackend` (`xert/backends/stack.py`) is `MuJoCoBackend` (new `model=` argument) on the stack scene with the
    bench reset pose. It returns float32 observations, like the native env. `send_targets(q, None)` stamps the
    command with the current sim time, which is finite and strictly increasing.
- **Callers through the protocol.**
  - `teleop.run_session(..., task_id=)` and `xert teleop --task {pick_place,stack}` (`teleop.py`, `runtime/cli.py`):
    - the backend comes from `task_backend`;
    - `source.bench = backend.metadata()` and the horizon comes from `backend.config`;
    - `task` is `TASK_TEXT[task_id]` and `task_id` goes to the manifest;
    - `record["task_id"]` goes to every journal line (per-frame provenance);
    - `success_rule` provenance is set per task, and `--viewer` is reused.
  - `gate_experiment`:
    - `bench_observation(backend, names)` reads `backend.observation()`;
    - `rollout(..., task_id)`;
    - `run()` takes the task from the recordings (one task_id; None or mixed is refused), and so does the
      pre-training probe;
    - `config.json` and the rows carry `task_id`.
  - `image_act`:
    - `describe()` resolves one task_id, which goes into checkpoint metadata, the report and the rows;
    - `evaluate()` refuses metadata without task_id;
    - `rollout()` uses `task_backend` and `CameraRenderer(size, robot, task_id)`.
- **Two-block rendering** (review H-03). `CameraRenderer.render(joints, objects)` takes a `{"block.x": ...}` dict
  covering every free body of the task. Offline (`row_state` + `scene_objects` in `_render_one`) and live
  (`image_act.live_frame`, from `backend.observation()`) both go through it. The reconstruction checks gain
  `base_block_position` / `base_block_orientation`.
- **task_id in data and the ledger** (H-04..H-07).
  - Manifests and rows:
    - recorder `task_id=` kwarg (manifest);
    - `xert_cameras.json`;
    - P2.1 `xert_export.json` (`pick_place`: the Store only holds native-bench episodes);
    - import.json path (b);
    - quality `metrics.json`, plus its ledger rows when the task is known;
    - gate rows and image-act rows.
  - `quality` refuses a session that mixes task_ids.
  - Ledger: `task_id` is optional. `validate_row` accepts rows without it (all 64 historical rows; the latency and
    IK writers) and refuses any other value when present. `resume_numbers` already keys on every non-measurement
    field, so pick_place and stack rows never merge (tested).
  - Legacy data without `task_id`, `cameras._recorded_task`: the manifest's task_id if present. Otherwise the
    fingerprints from `_recorded_models`, covering both layouts (runtime `source.bench.model_sha256`, export
    `source_episodes[*].simulation.model_sha256`):
    - if they all equal the robot's pick-and-place bench fingerprint (`bench_fingerprint`, cached), the task is
      `pick_place`;
    - other fingerprints are refused ("not guessed");
    - no bench at all (record-motion data) gives None, which quality accepts and render-cameras and the
      harnesses refuse.
  - replay-real stays without task_id: it replays hardware imports on the bare spec model, with no bench task
    (owner decision).
- **Scripted expert = CORE.** `xert/tasks/stack_expert.py` holds the stubs. `xle_workbench collect|evaluate --task
  stack` refuses with "stacking expert not tuned (TODO(daniel))" before creating anything. Teleop works without the
  expert.

## How to verify (commands run on this branch; numbers below are mine, from this session)

```
.\.venv\Scripts\python.exe -m pytest -q
ctest --test-dir build -C Release
.\.venv\Scripts\python.exe scripts\equivalence_check.py
.\.venv\Scripts\python.exe scripts\stack_reach.py
C:\Users\Khang\tmp\reach-public\build\venv-teleop\Scripts\python.exe -m pytest -q tests/test_cameras.py tests/test_so101_cameras.py tests/test_teleop.py tests/test_recorder.py tests/test_quality.py tests/test_gate_experiment.py tests/test_image_act.py tests/test_ledger.py tests/test_resume_numbers.py tests/test_lerobot_io.py tests/test_stack.py tests/test_stack_expert.py tests/test_stack_pipeline.py
```

- Base pytest: 214 passed, 67 skipped, 9 xfailed, 8 subtests. Baseline at 237fa48 was 196 / 61 / 5 / 8.
- ctest: 2/2.
- Equivalence: 16 seeds identical, max observation difference 0.0.
- Combined venv-teleop list (13 files): 118 passed, 1 skipped, 6 xfailed (375 s). Baseline (10 files) was 94 / 1 / 2. The XLeRobot camera pin test (`test_cameras.py::test_xlerobot_camera_poses_stay_the_p2_3_poses`) is in it and green.
- `tests/test_stack_pipeline.py` covers:
  - both evaluation harnesses with a constant-action stub: a full 600-tick stack episode, t_cmd=None, every step
    +0.04 s within 1e-9, info and timeout reported, policy input including base_block. image-ACT evaluate does
    this with live 64 px renders;
  - live-vs-offline exact pixel equality with base_block displaced, plus checks that the displacement is rendered;
  - a stack teleop recording through `next.success`, `environment_state` (21 names) and the journal task_id, then
    render-cameras and export;
  - a legacy recording and a legacy P2.1 `xert_export.json` without task_id rendering as pick_place with their
    `task` text intact, and a stripped stack recording refused;
  - mixed-task quality refused;
  - the real 64-row ledger read, a stack row and a latency row (no task_id) appended to a copy, with the real
    file's sha256 unchanged;
  - two rows differing only in task_id both eligible in `resume_numbers`.
- Measured in a probe script, not a test:
  - One StackBackend tick including StackTask takes 0.77 ms mean / 1.22 ms max (xlerobot_left) and 0.69 / 1.03 ms
    (so101) over 50 ticks. The extra contact query does not threaten the 25 Hz budget (40 ms).
  - A block placed on the base and left alone scores success at tick 13 (placed) or 14 (dropped from 3 mm) on both
    arms, so MuJoCo holds a two-block stack with the bench friction and solver settings.

## The TaskBackend contract

`task_id`, `spec`, `model`, `data`, `limits` (6×2 radians = spec position limits), `config` (has `horizon`), `info`
(after a step: `success`, `failure_reason`, `lifted`, `steps`, `object_position`, `object_in_robot_contact`; stack
adds `stable_steps`), `finished`, `object_names`, `metadata()` (recorded as `source.bench`; has `model_sha256`,
`observation_names`), `object_state()`, `observation()` ({name: value}), `read_state()`,
`send_targets(q, t_cmd | None)`, `close()`.

## TODO(daniel) (CORE)

- `xert/tasks/stack_expert.py:10` `stack_expert_targets(step, block, base_block)`: waypoint schedule.
- `xert/tasks/stack_expert.py:29` `stack_expert_action(backend)`: joint targets for one tick.
- Done means these pass, after which you delete the markers: `tests/test_stack_expert.py:24`
  `test_scripted_stack_success_on_seeds_0_to_8` and `:31` `test_scripted_stack_success_on_held_out_seeds`
  (10000..10008), both for xlerobot_left and so101.

## What is NOT shown

- No stacking success rate, scripted, teleop or learned. No expert exists.
- No real-time claim: StackTask timing is sim-time plus the probe's wall-clock numbers above.
- No human data. Every recording in the tests is labelled `scripted`.

## Three interview questions

1. *Why require release from the gripper?* Without it, a block held perfectly above the base would count as
   stacked. The test holds a perfect stack for 20 ticks and gets no success, then releases and gets success on clean
   tick 12.
2. *Why score in Python instead of extending the C++ Env?* Any C++ change alters the native contract hash and risks
   the 0.0-difference equivalence of pick-and-place. A Python rule on MjData costs under 1.3 ms per tick, and the
   C++ is not touched.
3. *How is legacy data without task_id classified?* Only by fingerprint. If every recorded model hash equals that
   robot's pick-and-place bench model, the task is pick_place. Anything else is refused, never guessed.

## Found along the way

- With MuJoCo's default 4 offscreen MSAA samples, re-rendering one identical state on this Windows GPU differed in
  3 of 40 frames for pick_place and 4 of 40 for stack (0 of 40 with `offsamples = 0`). This is pre-existing
  renderer noise, not a path difference. The pixel test turns MSAA off for its comparison. CI uses osmesa.
- Commit a51ce11 has that pixel test flaky in the base venv (1 value off by 1). 2e74d76 fixes it.

## For integration (not edited here)

- README SO-101 track: one line, "`xert teleop --task stack` records the block-stacking task; scripted stacking is
  TODO(daniel)".
- TASKS.md: add slice H, branch `feat/h-stack-bench`, owner Claude (builder), status built, awaiting QA.
- MVP_PROGRESS: second manipulation task scaffolded; no result.
