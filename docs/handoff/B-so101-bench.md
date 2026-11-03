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
- **Deviation from PLAN.md B:** the plan asked for `--robot` on `xert teleop` / `calibrate --bench`. Instead the arm comes
  from `--spec` (`BenchBackend` uses `spec.name`, `xert/backends/bench.py:16`). Those commands already require
  `--spec`, and a second selector could disagree with it.
- **Teleop profile** `configs/teleop/so101_bench.json`: position-only, gains `[0.4, 0.4, 0.3]`.
- **SO-101 expert** `xle_workbench/so101_expert.py`: TODO(daniel) stubs. SO-101 collect/evaluate refuse with
  `NotImplementedError: SO-101 scripted expert not tuned (TODO(daniel)) ...` before anything is written.
  Teleop does not need the expert.

## Scene provenance (measured, also stored in every SO-101 episode's `simulation.source.composition`)

| Item | Value | Evidence |
|---|---|---|
| Base pose | pos (0.42, 0.04, 0.08) on the table top, quat (0,0,0,1) = yaw pi, so the arm faces -x towards the block area and tray | `scripts/so101_bench_reach.py` |
| Reach | 14 samples, position-only `xert.retarget.ik.solve` (200 iterations) from the bench reset pose, then FK. Worst FK error **1.84e-16 m** (gate: < 1 mm, the P4.1 tolerance). Samples: the 4 corners of the block-spawn bounds at block centre z 0.096 and +0.06 / +0.075, and the tray at +0.04 / +0.10. | `test_reach_measurement_covers_block_region_and_tray` |
| Placement margin | `scripts/so101_bench_reach.py --base X 0.04 0.08` (same yaw, reset-pose start). Worst FK error: x 0.36 **2.54e-2 m (fails)**, x 0.38 1.82e-4 m, x 0.40 1.33e-16 m, x 0.45 1.62e-16 m, x 0.50 1.95e-16 m. y 0.04 sits between the block area and the tray. | that command |
| Timestep | Menagerie 0.005 s, bench **0.002 s** (same as the XLeRobot bench), so 20 substeps = 0.04 s = one 25 Hz period. 0.005 s would also be exact (1/(25*0.005) = 8.0); the plan pins 0.002 s. | `test_so101_bench_step_advances_one_25hz_period` |
| Contact solver | Kept from Menagerie `so101_xert.xml:2`: iterations 10, ls_iterations 20, impratio 10. The XLeRobot bench uses 100 / 50 / 1. **Contact solver settings differ between the benches.** Recorded as `composition.option`. | `test_scene_composition_leaves_vendored_files_and_records_provenance` |
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

Note for Daniel: QA found that a naive hover using the existing IK already passes the 5 mm pre-grasp
test. The two success tests carry the real bar.

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
   recorded in the scene provenance. `--base 0.36 0.04 0.08` misses by 25 mm, so the margin is measured.

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
- Pin the SO-101 contact solver to the XLeRobot bench's iterations 100 / ls_iterations 50 / impratio 1, or keep
  Menagerie's 10 / 20 / 10? Not changed; Daniel decides.

## Migration

- `implementation_sha256` is 473fff47 at 5cd6853, b986c806 at b0abc9b (QA), and 07f21e5d at HEAD (comments only since).
  The change is source-only. QA replayed 18 episodes across the hash change (p1.2 recovery_integrated,
  p1.3 act_v1, p0.2 scripted) with max difference 0.0, and the 16-seed equivalence is 0.0.
- The contract hash includes `implementation_sha256`. So episodes and models from p1.2/recovery_integrated,
  p1.3, p1.4 and p2.1 now fail `replay` / `load_verified_policy` on this core. Those results reproduce at
  their recorded `git_commit`.
- Training on old episodes and the P5.3 gate experiment are unaffected.
- XLeRobot configs now store `"robot": "xlerobot_left"`.
