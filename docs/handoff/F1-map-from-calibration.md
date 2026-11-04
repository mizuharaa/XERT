# Slice F1: `xert map-from-calibration`

Branch `feat/f1-map-from-calibration`, from `integration/sim-mvp` @ cd51dff. This slice is data tooling, tests
and docs. It has no CORE logic. The builder used no hardware, wrote no ledger rows and added no dependency.
Every calibration, pose and recording in the tests is a made-up fixture.

## What changed
- `xert/data/map_from_calibration.py` adds
  `xert map-from-calibration --calibration <id>.json --poses <poses.json> --output <fresh map.yaml>`.
  - **Files only.** It reads the LeRobot calibration file and the pose recordings. It never imports
    `lerobot.robots`, `lerobot.motors` or `serial`, and never opens a bus. A subprocess test checks
    `sys.modules` after a real CLI run.
  - **Output.** A fresh path only. The real `configs/hardware/so101_joint_map.yaml` exists, so the tool refuses
    to write it. You copy the result in by hand after you read it.
  - **Template.** Only `joint`, `motor`, `id` and `units` come from the real map. Every other field starts as
    TBD. The tool checks that the template units are LeRobot's so_follower norm modes (`so_follower.py:50-60`,
    `use_degrees=True`: degrees for the body joints, range_0_100 for the gripper). It also checks that each
    calibration motor id matches the map id.
- **Per pose.** It reads `observation.state` (`<motor>.pos`) of one episode and takes the median.
  - It refuses the pose if any joint's max − min over the episode is above 2 encoder steps (the slice D rule).
    One step is 360/4095 deg for a body joint and 100/(range_max − range_min) for the gripper.
  - It refuses a `sim_q` outside the so101 spec position limits, and a dataset that is not a so101 follower
    recording with the six `<joint>.pos` names.
- **Per joint, `min`/`max`.** These come from the calibration range through LeRobot 0.6.1's normalization
  (`motors_bus.py:854-911`):
  - degrees: `(range_min − mid)·360/4095` and `(range_max − mid)·360/4095`, with `mid = (range_min + range_max)/2`.
    The degree zero is the calibrated range midpoint, not the sim zero.
  - range_0_100: 0 and 100.
- **Per joint, `sign`/`units_per_rad`/`zero_offset`.**
  - **2 or more distinct `sim_q` values for the joint:** least squares `motor = a·sim_q + b` (`np.polyfit`,
    degree 1). `sign = sign(a)`, `units_per_rad = |a|`, `zero_offset = b`.
    - It refuses a max |residual| above 1.0 motor unit (1 deg, or 1% of gripper travel). This is a tooling
      choice, not a measurement.
    - With exactly 2 poses the residual is 0 by construction. The map says so in `fit.<joint>.note`. A third
      pose is the real check.
  - **1 distinct value, body joint:** offset only. `sign = +1` and `units_per_rad = 180/π` (direct drive) are
    ASSUMED and listed in `provenance.assumed`.
  - **1 distinct value, gripper:** sign, scale and offset stay TBD, because a 0–100 scale cannot be assumed.
  - **Sim limits check:** it maps the spec position limits through the fitted line and refuses the map if
    either end falls outside the calibrated `[min, max]`.
  - `max_speed`/`max_accel` always stay `TBD (not measured)`. They are your safety choice (P6 QA question 5).
- **`provenance` block in the map:**
  - the calibration path, sha256, robot id and raw values;
  - the tick→unit formulas;
  - the still rule;
  - each pose's label, dataset path, per-file dataset sha256, episode, frame count, `sim_q`, median and
    max − min;
  - the fit method, pose count, residual and tolerance per joint;
  - the sim limits in motor units;
  - `assumed`.
- **`xert/backends/so101_hw_backend.py` `load_joint_map(..., require="motion")`.**
  - `TIERS = {"kinematic": sign, units_per_rad, zero_offset, min, max; "motion": all seven}`.
  - Outside dry-run, a field that is TBD or listed in `provenance.assumed` is refused when it is in the
    required tier.
  - `import-lerobot` (path c) now asks for `require="kinematic"`. replay-real reads the import, so it follows.
    The default stays `motion`.
  - Dry-run is unchanged: any TBD or assumed field leaves the joint unmapped (`None`).
- `xert/__main__.py`: registers the command.
- `tests/test_map_from_calibration.py` (4 tests, LeRobot-dependent). `tests/test_so101_hw_backend.py` gets
  2 tier tests. `optional.yml`: the new file is added to the lerobot and teleop jobs.

## Why
The joint map is all TBD, so `import-lerobot --motor-units degrees`, `replay-real` and `hw-dry-run` refuse
real data. The calibration file already pins the range, and a few still poses pin the line from sim radians to
LeRobot's units. Speed and acceleration limits are a safety decision, so the tool never fills them. The
kinematic tier lets conversion run without them, while motion still needs all seven.

## Measured (2026-11-03, Windows, this worktree)
- Base `.\.venv\Scripts\python.exe -m pytest -q`: before (cd51dff) 190 passed, 44 skipped, 5 xfailed,
  8 subtests. After: 192 passed, 48 skipped, 5 xfailed, 8 subtests (+2 tier tests; the 4 new LeRobot tests
  skip without LeRobot).
- Combined `reach-public\build\venv-teleop` on `test_lerobot_io.py test_so101_hw_backend.py test_replay_real.py`:
  before 41 passed, 2 xfailed. With `test_map_from_calibration.py` added: 47 passed, 2 xfailed.
- On the fixture, the 2-pose fit recovers sign·scale and offset to 1e-9, including negative signs and a
  drive_mode 1 gripper. The fixture's import-lerobot then gives back `sim_q` to 1e-9 rad.

No real calibration or pose has been run through the tool. No real number exists yet.

## How to verify
```powershell
# Tests
.\.venv\Scripts\python.exe -m pytest -q
C:\Users\Khang\tmp\reach-public\build\venv-teleop\Scripts\python.exe -m pytest -q tests/test_map_from_calibration.py tests/test_so101_hw_backend.py tests/test_lerobot_io.py tests/test_replay_real.py

# 1. Calibrate the follower (you, owner-supervised; this writes the servos).
#    Writes %USERPROFILE%\.cache\huggingface\lerobot\calibration\robots\so_follower\<id>.json
#    (or $HF_LEROBOT_CALIBRATION\robots\so_follower\<id>.json).
lerobot-calibrate --robot.type=so101_follower --robot.port=<COM?> --robot.id=<id>

# 2. Record the still poses: one episode per pose, in one dataset. lerobot-record needs a teleop, so you hold
#    the leader still with the follower in the pose. Hold for the whole episode; reset between episodes.
lerobot-record --robot.type=so101_follower --robot.port=<COM?> --robot.id=<id> --teleop.type=so101_leader --teleop.port=<COM?> --teleop.id=<id> --dataset.repo_id=<user>/so101-poses --dataset.root=<dir>\so101-poses --dataset.fps=30 --dataset.num_episodes=3 --dataset.episode_time_s=3 --dataset.reset_time_s=20 --dataset.single_task="hold still" --dataset.push_to_hub=false

# 3. Write poses.json next to the dataset (schema below), then generate a FRESH map and read it.
.\.venv\Scripts\python.exe -m xert map-from-calibration --calibration $env:USERPROFILE\.cache\huggingface\lerobot\calibration\robots\so_follower\<id>.json --poses <dir>\poses.json --output <dir>\so101_joint_map.generated.yaml

# 4. Use it before you copy it in: import + replay through the kinematic tier.
.\.venv\Scripts\python.exe -m xert import-lerobot --dataset <recording> --output <dir>\import --motor-units degrees --joint-map <dir>\so101_joint_map.generated.yaml --calibration <id>.json
# 5. After reading provenance (fit residuals, assumed, sim_limits_in_motor_units), copy it by hand into
#    configs\hardware\so101_joint_map.yaml. Fill max_speed/max_accel yourself before any motion.
```
To set a pose, match the arm to the sim, for example
`python -m mujoco.viewer --mjcf=assets/menagerie/robotstudio_so101/so101_xert.xml` with the joint sliders.
Pick poses you can set precisely, such as links square to each other or to the table. Spread each joint far
apart: the scale error is about the pose error divided by the spread. Check that every pose is reachable and
collision-free on the bench before you record it.

## poses.json
A list with one entry per pose. Each entry has exactly these keys:
- `label`: text;
- `dataset`: a LeRobot v3 so_follower dataset path, relative to poses.json or absolute;
- `episode`: the episode index, an int;
- `sim_q`: 6 radians in `so101.yaml` order (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
  gripper). This is the pose you assert the arm was in.

Each joint needs at least 2 distinct `sim_q` values across the poses to be fitted. A joint with 3 or more
distinct values gets a residual check that means something. Example: the bench reset pose
(`xle_workbench/sim.py` ROBOTS["so101"]), the sim zero with the gripper closed, and the reset pose with the
pan and roll turned. Shoulder_pan and wrist_roll are 0 in the reset pose, so the third pose is what fits them.
```json
[
  {"label": "bench reset", "dataset": "so101-poses", "episode": 0, "sim_q": [0.0, -1.5, 1.0, 1.4, 0.0, 1.4]},
  {"label": "sim zero, gripper closed", "dataset": "so101-poses", "episode": 1, "sim_q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
  {"label": "reset, pan +1.0, roll +1.5", "dataset": "so101-poses", "episode": 2, "sim_q": [1.0, -1.5, 1.0, 1.4, 1.5, 1.4]}
]
```
Only the reset pose comes from the code. The other two are examples: check them in the viewer and on the
bench first. If sim zero is not reachable, use another pose.

## Interview questions
1. *Why a fit and not just the calibration file?* The calibration fixes LeRobot's zero at the range midpoint
   and the tick scale, but not where the sim's zero is or which way the joint turns relative to the sim, and
   only labelled poses tie the two frames together.
2. *Why refuse assumed values instead of using 180/π?* Direct drive is likely for the body joints, but
   unmeasured, so it is recorded as assumed and refused like TBD outside dry-run, and a wrong sign would
   mirror every joint command.
3. *Why two completeness tiers?* Converting recorded frames needs only the line from radians to motor units,
   but commanding motion also needs speed and acceleration limits, which are a safety choice and not
   something a still pose can measure.

## Stays TBD
- `max_speed`, `max_accel` for every joint: your choice (P6 QA question 5).
- Gripper sign, scale and offset until you record 2 or more distinct gripper poses.
- Any body joint recorded at only one `sim_q` value: the offset is filled, sign and scale are assumed, and
  the loader refuses them.

## README / TASKS / MVP_PROGRESS lines integration may add
- README, SO-101 track: "`xert map-from-calibration` fills the joint map's kinematic fields from a LeRobot
  calibration file plus labelled still poses; speed and acceleration limits stay TBD (owner choice)."
- TASKS: "F1 map-from-calibration: tool + tests merged; owner runs lerobot-calibrate, records ≥3 still poses,
  reviews the generated map, copies it in, chooses max_speed/max_accel."
- MVP_PROGRESS: "Joint map: tooling to fill kinematic fields done (F1); real values pending owner bring-up."

## Open questions
1. Residual tolerance 1.0 motor unit is a guess. Poses set by eye against the viewer may scatter more than
   1 deg. Tighten or loosen it once real poses show their scatter.
2. "The joint moves" means 2 or more distinct `sim_q` values, with no minimum spread. A tiny spread gives a
   poorly conditioned slope. A minimum (for example 0.5 rad) is a one-line add if you want it.
3. LeRobot 0.6.1 `so_follower.calibrate` always writes `drive_mode=0` (`so_follower.py:146-153`). The drive_mode
   1 gripper exists only in the fixture. With degrees the sign comes from the fit either way.
4. The sim-limit check refuses a map when the sim range is wider than the calibrated one. On a real arm it may
   trip for wrist_roll (calibrated 0..4095 = ±180 deg, while the spec allows −157..+163 deg plus the offset).
   If it trips, it means the sim can command a pose the arm cannot reach. Decide whether that should be a
   refusal or a warning.
