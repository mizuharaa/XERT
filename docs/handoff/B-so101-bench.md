# Slice B: SO-101 as a second, CLI-selectable bench robot

Branch `feat/b-so101-bench`, from `integration/sim-mvp` @ 5cd6853. Simulation only; no camera, no
hardware, no ledger rows written by the builder.

## What changed and why

- **Native core is robot-driven** (`cpp/include/reach/env.hpp`, `cpp/src/env.cpp`, `cpp/src/bindings.cpp`).
  `reach::Robot` holds the six joint names (actuator i drives joint i), the grasp site, the reset pose,
  `control_hz` and whether the native scripted expert is tuned for that arm. `Env` used to hard-code
  the XLeRobot names (`env.cpp:20`), the reset pose and 20 substeps (`env.cpp:87`).
  - Substeps are now derived: `round(1/(control_hz*timestep))`. The core refuses the model unless that
    value is an exact integer (relative 1e-9). XLeRobot: 1/(25*0.002) = 20.
  - `xlerobot_left()` is the default robot. The standalone `reach_trials`, `reach_tests` and the old
    four-argument Python call are unchanged.
  - `Env::expert_action` refuses an arm whose `scripted_expert` is false. The XLeRobot expert in
    `ik.cpp` is untouched, and so is the generalized IK.
- **Python bench** (`xle_workbench/sim.py`). `ROBOTS` holds both arms. `ManipulationEnv` takes config key
  `robot` (default `xlerobot_left`) and passes that robot's config to the core. Observation and action
  names follow the arm.
- **SO-101 scene** (`so101_scene()` in `sim.py`). It loads the vendored `so101_xert.xml` with `mujoco.MjSpec`,
  then attaches the XLeRobot bench's own light, floor, table, tray and block. These are taken from
  `model_xml()` with the XLeRobot arm and its mount removed, so block and tray are identical bodies at
  the same world poses. No vendored file is written; the menagerie checksums are verified at load.
- **Backends and CLI.** `BenchBackend` picks the arm from the spec name, so `xert calibrate --bench` and
  `xert teleop` select the SO-101 bench with `--spec xert/spec/so101.yaml`. `xle_workbench`
  collect/evaluate/replay take `--robot {xlerobot_left,so101}`.
- **Teleop profile** `configs/teleop/so101_bench.json`: position-only, gains `[0.4, 0.4, 0.3]`.
- **SO-101 expert** `xle_workbench/so101_expert.py`: TODO(daniel) stubs. SO-101 collect/evaluate refuse with
  `NotImplementedError: SO-101 scripted expert not tuned (TODO(daniel)) ...` before anything is written.
  Teleop does not need the expert.

## Scene provenance (measured, also stored in every SO-101 episode's `simulation.source.composition`)

| Item | Value | Evidence |
|---|---|---|
| Base pose | pos (0.42, 0.04, 0.08) on the table top, quat (0,0,0,1) = yaw pi, so the arm faces -x towards the block area and tray | `scripts/so101_bench_reach.py` |
| Reach | 14 samples, position-only `xert.retarget.ik.solve` (200 iterations) from the bench reset pose, then FK. Worst FK error **1.84e-16 m** (gate: < 1 mm, the P4.1 tolerance). Samples: the 4 corners of the block-spawn bounds at block centre z 0.096 and +0.06 / +0.075, and the tray at +0.04 / +0.10. | `test_reach_measurement_covers_block_region_and_tray` |
| Placement choice | Selection probe (scratch, not committed; y 0.03, zero start pose, same samples): base x 0.38 worst 6.8e-3 m (fails); x 0.40 / 0.42 / 0.45 worst <= 3.1e-16 m; x 0.40 with yaw 0 worst 0.178 m. Picked 0.42, between the passing values, and y 0.04, between the block area and the tray. | probe output |
| Timestep | Menagerie 0.005 s, bench **0.002 s** (same as the XLeRobot bench), so 20 substeps = 0.04 s = one 25 Hz period. 0.005 s would also be exact (1/(25*0.005) = 8.0); the plan pins 0.002 s. | `test_so101_bench_step_advances_one_25hz_period` |
| Actuator ctrlrange | Clamped to the joint range, so actions normalized by ctrlrange stay inside `so101.yaml` limits: shoulder_lift ±1.74533 to ±1.7453293, wrist_roll [-2.74385, 2.84121] to [-2.7438473, 2.8412063], gripper upper 1.74533 to 1.7453292 | `test_limits_come_from_the_menagerie_model_and_spec` |
| Reset pose | [0, -1.5, 1.0, 1.4, 0, 1.4] rad (gripper at the spec's open value). Grasp at (0.251, 0.040, 0.170). No arm-block contact, and joints within 1e-3 rad after 25 held ticks (measured drift 3.9e-4 rad after 50) | `test_reset_holds_still_without_touching_the_block` |
| Teleop reach | Bench profile, P4.1 method. Image dx <= 0.212, dy <= 0.184, palm scale 0.99 to 1.09, IK error <= 2.0e-14 m, for the seed-0 block, the 4 spawn corners, 6 cm above the block and the tray +4 cm | `test_bench_profile_reaches_block_and_tray_from_modest_hand_motion` |

Joint names = LeRobot 0.6.1 `so101_follower` (`so_follower.py:54-59`). The test compares them with the
installed source when `lerobot` imports (combined env: "installed lerobot"), else with a pinned list
(base env). LeRobot defines no radian limits; per-arm calibration supplies them. So the limits are checked
against the Menagerie model and `so101.yaml`.

## How to verify (from the checkout root)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_core.ps1
.\.venv\Scripts\python.exe -m pytest -q
ctest --test-dir build -C Release
.\.venv\Scripts\python.exe scripts\equivalence_check.py
.\.venv\Scripts\python.exe scripts\so101_bench_reach.py
# Combined env: needs a cp313 _reach_core built from THIS C++ (the old cp313 module lacks the new arguments).
C:\Users\Khang\tmp\reach-public\build\venv-teleop\Scripts\python.exe -m pytest -q tests/test_teleop.py tests/test_recorder.py tests/test_so101_bench.py tests/test_so101_expert.py
```

CLI surface:

```powershell
.\.venv\Scripts\python.exe -m xle_workbench.cli collect --robot so101 --episodes 1       # refuses: TODO(daniel)
.\.venv\Scripts\python.exe -m xle_workbench.cli evaluate --robot so101                  # refuses: TODO(daniel)
.\.venv\Scripts\python.exe -m xle_workbench.cli replay EPISODE_ID --robot so101         # refuses another arm's episode
.\.venv\Scripts\python.exe -m xert calibrate --bench --spec xert/spec/so101.yaml --profile configs/teleop/so101_bench.json --output build/so101_neutral.json
.\.venv\Scripts\python.exe -m xert teleop --spec xert/spec/so101.yaml --profile configs/teleop/so101_bench.json --calibration build/so101_neutral.json --output build/so101_session
```

## TODO(daniel) and the tests that define done

| Stub | Done when (strict xfail, `raises=NotImplementedError`) |
|---|---|
| `xle_workbench/so101_expert.py:11` `so101_expert_targets(step, initial_block, tray) -> (target, grip)`: the waypoint schedule | `tests/test_so101_expert.py:43` `test_expert_reaches_pre_grasp_within_5mm_on_seeds_0_8[so101]` |
| `xle_workbench/so101_expert.py:30` `so101_expert_action(env) -> (6,) in [-1,1]`: IK targets, gains, rate limits | `tests/test_so101_expert.py:54` `test_expert_lifts_through_contact_and_succeeds_on_ctest_seeds_0_8[so101]`, `tests/test_so101_expert.py:62` `test_expert_succeeds_on_held_out_seeds_10000_10009[so101]` |

The criteria are the XLeRobot expert's own, and each test also runs, and passes, on `xlerobot_left`:
- seed-0 lift through contact plus release (`test_sim.py`, `test_core.cpp`);
- the CTest export seeds 0..8;
- the evaluate default seeds from 10000;
- 5 mm pre-grasp (PLAN.md).

Measured XLeRobot reference: hover error worst 4.4 mm on seeds 0..8 at step 65; 9/9 on 0..8; 10/10 on
10000..10009. Once the stubs exist, CI turns red. A missed criterion shows as a failure; all three
passing show as XPASS(strict). Then delete the marker.

## Interview questions

1. *How did you make the C++ env robot-agnostic without changing XLeRobot results?* Every hard-coded value
   moved into a `Robot` struct whose default holds the same literals. The equivalence check still matches
   the untouched Python reference bit for bit (16 seeds, max observation difference 0.0).
2. *Why change the SO-101 timestep from 0.005 to 0.002 s?* Substeps per 25 Hz tick are derived as
   1/(hz·dt) and must be an exact integer. 0.002 s keeps the same 20 × 2 ms integration as the XLeRobot
   bench, and the recorder checks that every tick advances sim time by exactly 0.04 s.
3. *How do you know the SO-101 can reach the block and tray?* The existing generalized IK solves 14
   sample targets from the reset pose, FK confirms each one to 1.84e-16 m worst, and the result is
   recorded in the scene provenance. A base 4 cm closer failed by 6.8 mm.

## README lines for integration to add

- Under "xlerobot_left stands in for SO-101 on the bench" (README.md:149): "Slice B adds an SO-101 bench
  (`--robot so101`, `--spec xert/spec/so101.yaml` with `configs/teleop/so101_bench.json`). Teleop
  works; scripted collect/evaluate wait for the TODO(daniel) SO-101 expert."
- In the original-workbench command block (README.md:271-275): `collect/evaluate/replay ... --robot {xlerobot_left,so101}` (default xlerobot_left).
- In the native-core section: "Substeps = 1/(control_hz·timestep), refused unless an exact integer (20 on both benches)."

## Open questions

- Top-down grasp orientation reach at this placement was **not measured**. Position-only reach is the
  gate. Daniel's expert may want the base moved, which changes `SO101_BASE` and the provenance.
- The reset pose is a scaffolding choice, not a tuned value. Calibration neutral = this pose.
- B2 owns cameras and the exporter: `CameraRenderer` and `export-lerobot` still accept XLeRobot only.
- The XLeRobot `config` now stores `"robot": "xlerobot_left"`, and `implementation_sha256` changed as any
  core edit does, so episodes recorded before this branch fail the replay contract check (as before
  for any core change).
