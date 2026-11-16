# Slice L: Cartesian pose columns for VLA fine-tuning

Branch `feat/l-add-pose`, from dbb039b. Simulation only. A format conversion: forward kinematics, a dataset
rewrite, tests and docs. **No result and no ledger row**: no policy was trained and nothing ran on hardware. No
CORE code (no IK, no controller, no training logic) and no new dependency. The plan is `PLAN.md` section L
(Codex review findings L-01..L-06). A QA fix round followed (test gaps, two owner defaults applied, handoff facts).

## What changed and why

End-effector-space VLAs such as Runway's Praxis-1 take the gripper pose, not joint angles. XERT exports joint
positions and normalized joint targets. `add-pose` adds the pose next to them without touching the recorder, the
Store or the existing export.

- **Command.** `python -m xert add-pose --dataset <P2.1 export | P4.1 recording | session of episode_*> --output
  <fresh dir>` (`xert/data/pose.py`, registered in `xert/__main__.py`). The `render-cameras` shape: never
  overwrites, one bench robot and one `task_id` per run, a refused run writes nothing.
- **Two new features**, appended. Every existing column is copied byte-equal; the provenance sidecars are copied.
  - `observation.state.cartesian`: float32, shape (7,), names `[x, y, z, roll, pitch, yaw, gripper]`, from the
    recorded joint positions (`joint_position.<joint>` in exports, `<joint>.pos` in recordings).
  - `action.cartesian`: the same layout, from the dataset's `action` feature (see "Action semantics").
- **Pose.** The spec's `grasp` site (both specs declare exactly `sites: ["grasp"]`), by `mujoco.mj_kinematics` on
  the bench model the data was recorded on (`camera_model(robot, task_id)`: pick_place, stack and drawer all
  carry the same arm chain). The site's orientation is the frame of the body it sits on (no site rotation).
- **Base frame (L-01, plus owner default 2).** The root body's constant reference placement in the world,
  `model.body_pos[root]` / `model.body_quat[root]`, never its live `xpos`/`xquat`, then a fixed per-robot
  correction (`ROOTS` in `pose.py`) so both robots report z up and joint 1 turns about base -z. Both roots are
  worldbody children (asserted). A fixed frame, not the world, so the bench placement cancels and sim and future
  real poses are comparable.
  - so101: root `base`, correction identity. Base position (0.42, 0.04, 0.08), quat (0, 0, 0, 1) (yaw pi).
  - xlerobot_left: root `Rotation_Pitch` (it carries `Rotation_L`), correction (sqrt(.5), -sqrt(.5), 0, 0),
    -90 degrees about the root's x: base x = root x, base y = root -z, base z = root y. Resulting base quat
    (0.707105, 1.4e-6, 1.4e-6, 0.707108), about 90 degrees about world z.
  - Measured: world up in the base frame is (0, 0, 1) for so101 and (8e-12, 4e-6, 1.0) for xlerobot_left. The
    4e-6 is in the vendored root quat (0.499998, 0.5, 0.500002, 0.5), not the correction; the test bound is 1e-5.
  - `xert_pose.json` records the root quat, the correction (`base_frame_correction`) and the resulting base quat.
- **Rotation (L-02).** Extrinsic x-y-z Euler radians: `R = Rz(yaw) Ry(pitch) Rx(roll)` (the DROID convention).
  MuJoCo's oracle is `mju_euler2Quat(quat, euler, "XYZ")`. Measured on mujoco 3.14.0 with roll 0.3, pitch -0.7,
  yaw 1.1: `"XYZ"` equals the matrix `Rz Ry Rx` to 2.2e-16 (max absolute element difference), and differs from
  `Rx Ry Rz` by 0.64 by the same metric. `"XYZ"` and `"xyz"` of the same angles differ by 0.354 as quaternions
  (max absolute component, sign-aligned), a 0.80 rad rotation. So uppercase is extrinsic, as the plan read from the
  MuJoCo source.
- **Gripper (L-06, plus owner default 1).** `g = (open - q) / (open - closed)` from the spec's `gripper` block
  (so101 open 1.4 / closed 0; xlerobot_left 1.4 / -0.1), clipped to [0, 1]: **0 = spec open, 1 = spec closed**,
  DROID's direction. The so101 joint limits give -0.247 (upper, 1.7453 rad) and 1.125 (lower, -0.1745 rad)
  unclipped. The raw joint columns stay in the dataset, so clipping loses nothing. `xert_pose.json` records the
  clip count and the largest excursion beyond [0, 1] per column.
- **Action semantics (L-04).** `action.cartesian` is FK of the dataset's `action`, whatever it holds.
  - P4.1 recordings: `<joint>.pos` radians after safety clamps; FK applies directly.
  - P2.1 exports: `action` is normalized to [-1, 1]. It is decoded with `q = low + (a + 1) / 2 * (high - low)`,
    using `action_joint_ranges_radians` from `xert_export.json`. Episodes with different ranges are refused.
  - In exports, `action` is the training-label stream: verified corrective labels when present, executed targets
    otherwise. So `action.cartesian` is "FK of the training labels", not "the commanded pose".
    `xert.executed_action` gets no Cartesian column in this slice (a one-line extension if a consumer needs it).
    The sidecar copies the export's `action_semantics` string.
- **Built-in check.** Bench data records `grasp.x/y/z` in the world frame. `add-pose` recomputes the world grasp
  position from the same FK and refuses if `max |FK - recorded| > 1e-5 m` (`TOLERANCE`, the render-cameras bound).
  This checks the FK and the joint-column wiring. The base-frame conversion is tested separately
  (`tests/test_pose_fk.py`), because world FK does not depend on the reporting frame.
- **Sidecar `xert_pose.json`** (one per dataset): site, root body, root quat, frame correction and base transform,
  Euler convention, `max_abs_pitch_rad` per column, gripper definition, spec values and clip counts,
  `action_semantics` and the decode, `model_sha256`, `spec_sha256`, `mujoco_version`, `lerobot_version`, source
  hashes, the measured FK maximum error, and the statement "kinematic model pose of the simulated arm".
- **Refusals.** Existing output; a dataset already carrying `*.cartesian`; image-bearing input ("run add-pose
  before render-cameras"); hardware imports; missing joint or `grasp.*` columns; FK disagreeing with the recorded
  grasp; a model fingerprint that is not the current bench model; two robots or two tasks in one session; mixed
  action ranges. Each one is tested, and each writes nothing.
- **Simulation data only (L-03).** Admission needs the recorded bench model fingerprint, which only simulator
  recordings and exports carry. A raw `so101_follower` dataset has no XERT manifest. `to-store` writes
  `derived_frame_pairs` episodes with no `simulation` object, and `lerobot_export._schema` needs
  `hardware=False` and `simulation_only=True`, so the real-data chain does not reach the exporter today. Nothing
  here weakens `_schema` or `to_store`.
- **Order (L-05).** `export-lerobot` -> `add-pose` -> `render-cameras`. `render-cameras` converts non-image
  features as NumPy arrays, so it carries the two float32 (7,) columns through byte-equal (tested). Existing
  `--cameras` exports must be made again in this order.

## Where the build differed from the plan

1. **xlerobot_left base frame: resolved by owner default 2.** As planned (root placement only), the xlerobot
   frame had y up and `Rotation_L` was the base (0, -1, 0) axis, so joint 1 moved base x and z, not the plan's
   `[x, y, yaw]`. Applied default: the fixed frame correction above, recorded in the sidecar as
   `base_frame_correction`. Now on both robots +0.4 rad of joint 1 alone changes x, y and yaw (by exactly -0.4,
   1e-12) and leaves z, roll and pitch unchanged (1e-12); the test asserts this for both.
2. `ROOTS` lives in `xert/data/pose.py`, not next to `PROFILES` in `lerobot_export.py`, to keep the slice in one
   new module.

## How to verify

Run from the worktree, one pytest at a time.

```
.venv\Scripts\python.exe -m pytest -q tests/test_pose_fk.py
.venv\Scripts\python.exe -m pytest -q
ctest --test-dir build -C Release --output-on-failure
.venv\Scripts\python.exe scripts/equivalence_check.py
set HF_HUB_OFFLINE=1
<lerobot env>\Scripts\python.exe -m pytest -q tests/test_add_pose.py tests/test_lerobot_io.py
<lerobot env>\Scripts\python.exe -m pytest -q tests/test_lerobot_export.py tests/test_cameras.py
```

Measured on this branch at 959e70f, after the QA fix round (Windows, mujoco 3.14.0, LeRobot 0.6.1, 2026-11-16),
every command exit code 0:

| Gate | Result |
| --- | --- |
| `tests/test_pose_fk.py` (base env) | 9 passed |
| Base env, whole suite | 274 passed, 73 skipped, 15 xfailed, 8 subtests passed, 122 s (`test_add_pose.py` skips there: no lerobot) |
| `ctest` | 2/2 passed |
| `equivalence_check.py` | 16 seeds, identical, max observation difference 0.0 |
| `tests/test_add_pose.py` (lerobot env) | 13 passed |
| LeRobot gate, four files in two runs | 27 + 14 = 41 passed (52 s + 29 s) |

From the `xert_pose.json` files that `tests/test_add_pose.py` wrote in that run (`--basetemp` kept them), 3-frame
scripted fixtures (not demonstrations). The FK errors and clip counts are the same as before the fix round:

| Data | max \|FK - recorded grasp\| (m) | Gripper clips (state / action) | max \|pitch\| state / action (rad) |
| --- | --- | --- | --- |
| xlerobot_left pick_place export (first 3 steps of a P1.2 episode) | 2.26e-8 | 0 / 0 | 1.439 / 1.559 |
| so101 pick_place export (3-step hold at the reset pose) | 7.79e-9 | 0 / 0 | 0.670 / 0.670 |
| xlerobot_left pick_place teleop recording | 1.66e-8 | 0 / 0 | 1.196 / 1.185 |
| xlerobot_left stack teleop recording | 1.85e-8 | 0 / 0 | 1.196 / 1.185 |
| xlerobot_left drawer teleop recording | 1.85e-8 | 0 / 0 | 1.196 / 1.185 |

The FK errors are float32 storage of the joint angles, far below the 1e-5 m bound. No real fixture clips the
gripper; a dataset-path test narrows the spec's `open` to 1.0 so the so101 hold (1.4 rad) clips on all 3 frames
of both columns (excursion 0.4).

**Gimbal note.** Roll and yaw are not unique at pitch = +-pi/2; the matrix is exact, Euler is only a
representation, and the tests skip a 0.05 rad band there. Pitch near +-pi/2 means the grasp site's x axis is near
base vertical. Measured (base env, mujoco 3.14.0, single-joint steps of +-0.02 rad):
- so101 all-zero `rest`: pitch -1.5221, 0.049 rad from -pi/2; the steps move roll or yaw by up to 0.39 rad.
- xlerobot_left all zeros: pitch 1.570790, 6e-6 rad from pi/2 (not a bench pose, but on the singularity).
- Both bench reset poses are well conditioned: so101 pitch -0.670 (0.90 rad from -pi/2), roll/yaw move at most
  0.026 rad; xlerobot_left pitch 1.171 (0.40 rad from pi/2), at most 0.051 rad.
- The fixtures above come closer: the xlerobot P1.2 labels reach |pitch| 1.559, 0.012 rad from pi/2.

A consumer must not read roll and yaw near the band as smooth. `xert_pose.json` records `max_abs_pitch_rad` per
Cartesian column, so each dataset shows how close it comes. A fixed tool-frame rotation (like the base
correction) would move the singularity away from the xlerobot's working poses; that is an owner decision, not
made here.

What the tests cover:
- Base frame: constant, equal to the root placement times the correction in every task scene, z up on both
  robots. The xlerobot root's live frame turns with joint 1, so it must not be used.
- Joint 1 alone, on both robots. A hand-built frame conversion.
- Euler: random reachable and random rotations through the `"XYZ"` oracle within 1e-9; a mixed-axis case against
  `Rz Ry Rx`; an intrinsic reading that fails.
- Gripper: endpoints (open -> 0, closed -> 1), midpoint, counted clipping on hand values and in a dataset.
- Both robots' exports: every source column byte-equal; literal feature names, dtype float32 in `info.json` and in
  the Parquet; `LeRobotDataset` loads; `task` kept; world grasp rebuilt from the stored base-frame xyz within
  1e-5; both Cartesian columns equal to FK of the joints and of the decoded action.
- `action.cartesian` follows the recorded ranges, not the spec limits (both benches record ranges equal to their
  spec limits, so a test changes them), and it is FK of the decoded labels, not of `xert.executed_action`, on a
  fixture whose streams differ.
- pick_place, stack and drawer recordings (xlerobot_left; SO-101 teleop fixtures do not exist) and a session.
- Every refusal above, writing nothing. The `add-pose` -> `render-cameras` chain (pose columns byte-equal, images
  identical to rendering the plain export, MSAA off in the test only). The CLI.

## Model input tables

Praxis-1, as read from huggingface/lerobot `docs/source/praxis.mdx` on 2026-11-11; the flange pose, the fine-tune
recipe and the GPU figures were read from the same guide in the QA fix round:

| Praxis-1 input | Praxis-1 expects | XERT after `add-pose` (+ `render-cameras`) |
| --- | --- | --- |
| `observation.state` | 7-D `[x, y, z, roll, pitch, yaw, gripper]`: the Franka flange pose in the base frame, metres, extrinsic xyz radians, gripper [0, 1] | `observation.state.cartesian`: the `grasp` site between the jaws, not a flange (see below); `observation.state` stays joint radians |
| `action` | 7-D absolute Cartesian targets, same layout | `action.cartesian` (FK of the training labels); `action` stays joint targets |
| `observation.images.exterior` | external camera, any size (tiled to 224x416) | `observation.images.scene` (renamed at training) |
| `observation.images.wrist` | wrist camera | `observation.images.wrist` |
| `task` | instruction string | `task` |
| rate | 15 Hz, 24-step chunks | 25 Hz (`control_hz`) |

Pose point: Praxis's DROID data is the Franka flange pose; XERT reports the `grasp` site between the jaws. That
is a convention difference and an owner decision (keep `grasp`). On a rigid gripper the offset is a fixed
transform, so either can be derived from the other.

SmolVLA in LeRobot 0.6.1 (`policies/smolvla/configuration_smolvla.py`): any state and action up to 32-D
(`max_state_dim`, `max_action_dim`), images padded to 512x512, 50-step chunks, MEAN_STD state/action. It needs
nothing from this slice:

| SmolVLA input | SmolVLA expects | XERT |
| --- | --- | --- |
| `observation.state` | <= 32-D | 6 joint radians (recordings) or 26-D bench state (exports); or the 7-D Cartesian column |
| `action` | <= 32-D | 6 joint targets |
| `observation.images.*` | one or more cameras | `scene`, `wrist` |
| `task` | instruction string | `task` |

Key mapping at training time. The Praxis guide's own fine-tune recipe uses this slice's columns through
`--rename_map`, and states that a renamed key replaces an existing one:

```
RENAME='{"observation.state.cartesian":"observation.state","action.cartesian":"action","observation.images.scene":"observation.images.exterior"}'
```

- With the LeRobot that ships Praxis (0.7+), the recipe's map does the key swap: the joint `observation.state`
  and `action` are replaced by the Cartesian columns. So no keys-swapped copy of the dataset is needed.
- With the LeRobot pinned here, 0.6.1, that map does not work and is not needed: 0.6.1 ships no Praxis policy
  (`lerobot/policies` has no `praxis` directory), its `RenameObservationsProcessorStep` renames observation keys
  only, and `rename_map` requires a pretrained checkpoint (`configs/train.py:247`). All three checked in the
  installed 0.6.1 on 2026-11-16.

## Not in this slice

1. Hardware data (slice L2). It needs: hardware provenance kept end to end (calibration sha256, robot id, the
   joint-map identity from slice F1); an explicit, recorded choice of FK model (the Menagerie kinematic model is
   not a measurement of Daniel's arm); and no fabricated simulation provenance.
2. 25 Hz -> 15 Hz for Praxis: an owner decision between resampling (an interpolation rule for actions and a frame
   choice for images) and recording new data at 15 Hz (`control_hz` in the specs; changes `contract_sha256`).
3. IK execution of Cartesian commands on the arm (CORE).
4. Any Praxis or SmolVLA training run. That needs real demonstrations. Per the guide, Praxis inference needs one
   80 GB GPU; its LoRA runs used 4x H100 at about 30 GB per rank, or one 96 GB-class GPU (RTX PRO 6000 or B200)
   at about 62 GB; a full fine-tune needs at least 4x 80 GB. The laptop has 12 GB, so Praxis never runs locally.

float32 storage of metres loses about 1e-7 m, below the tolerance.

## Owner decisions

Two were applied as recommended defaults (standing instruction). Both are reversible; the tests that pin them
(`tests/test_pose_fk.py`) must be flipped in the same change.

- **Gripper direction: 0 = open, 1 = closed (DROID's).** Neither the Praxis guide nor the DROID schema page states
  the direction. DROID's robot code settles it (evidence found by QA): `droid/franka/robot.py` reads
  `1 - width / max_width` and commands `width = max_width * (1 - command)`, so 0 = open, 1 = closed. To reverse:
  the formula line in `gripper_fraction` in `xert/data/pose.py` (now `(open - q) / (open - closed)`), plus the
  `gripper.definition` string written to `xert_pose.json`.
- **xlerobot_left z-up base frame.** QA measured world up in the xlerobot root (`Rotation_Pitch`) frame as
  (0, 1, -4e-6) and the `Rotation_L` axis as (0, -1, 0); re-measured here: (8e-12, 1, -4e-6) and (0, -1, 0). The
  guide names no axis convention for a single-arm base frame. The default adds a fixed -90 degree correction about
  the root's x so z is up and joint 1 turns about base -z, as on the SO-101, recorded in the sidecar as
  `base_frame_correction`. To reverse: the `xlerobot_left` entry of `ROOTS` in `xert/data/pose.py` (set the
  correction to identity, (1, 0, 0, 0)).

Still open:
- 15 Hz: resample or re-record (above).
- Pose point: `grasp` site (kept) versus a flange-like point.
- xlerobot tool-frame orientation near the pitch singularity (gimbal note above).

TODO(daniel): none new.

## Questions an interviewer could ask

1. Why a fixed base transform and not the root body's live frame? (The xlerobot root carries joint 1, so its live
   frame turns with the arm; a pose in it would hide joint 1. The fixed placement also cancels where the bench put
   the arm, so sim and real poses compare.)
2. Why is `action.cartesian` FK of the training labels, and when does that differ from the executed command?
   (Exports put corrective labels in `action` when present; in DAgger data the arm executes `dagger_action`'s
   output while the label is the expert's, so the two streams differ.)
3. What breaks if the Euler convention is intrinsic? (Intrinsic x-y-z is `Rx Ry Rz`: the same three numbers name a
   different rotation, 0.80 rad away in the test case, so a model trained on DROID-convention data would be fed
   wrong orientations.)
