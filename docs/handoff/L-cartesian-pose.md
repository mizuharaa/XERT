# Slice L: Cartesian pose columns for VLA fine-tuning

Branch `feat/l-add-pose`, from dbb039b. Simulation only. A format conversion: forward kinematics, a dataset
rewrite, tests and docs. **No result and no ledger row**: no policy was trained and nothing ran on hardware. No
CORE code (no IK, no controller, no training logic) and no new dependency. The plan is `PLAN.md` section L
(Codex review findings L-01..L-06).

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
  carry the same arm chain).
- **Base frame (L-01).** The root body's constant reference placement in the world, `model.body_pos[root]` /
  `model.body_quat[root]`, never its live `xpos`/`xquat`. Roots (`ROOTS` in `pose.py`): so101 `base`,
  xlerobot_left `Rotation_Pitch`. Both are worldbody children (asserted). A fixed frame, not the world, so the
  bench placement (`SO101_BASE`) cancels and sim and future real poses are comparable.
  - so101 base: position (0.42, 0.04, 0.08), quat (0, 0, 0, 1) (yaw pi). z is up.
  - xlerobot_left base: position (0, 0, 0.20), quat (0.499998, 0.5, 0.500002, 0.5). **y is up** in this frame
    (see "Disagreements with the plan", 1).
- **Rotation (L-02).** Extrinsic x-y-z Euler radians: `R = Rz(yaw) Ry(pitch) Rx(roll)` (the DROID convention).
  MuJoCo's oracle is `mju_euler2Quat(quat, euler, "XYZ")`. Measured on mujoco 3.14.0 with roll 0.3, pitch -0.7,
  yaw 1.1: `"XYZ"` equals `Rz Ry Rx` to 2.2e-16 and differs from `Rx Ry Rz` by 0.64; lowercase `"xyz"` is the
  reverse. So uppercase is extrinsic, as the plan read from the MuJoCo source.
- **Gripper (L-06).** `g = (q - closed) / (open - closed)` from the spec's `gripper` block (so101 closed 0 / open
  1.4; xlerobot_left -0.1 / 1.4), clipped to [0, 1]. The so101 joint limit 1.7453 rad gives 1.247 unclipped.
  The raw joint columns stay in the dataset, so clipping loses nothing. `xert_pose.json` records the clip count
  and the largest excursion beyond [0, 1] per column. 0 = spec closed, 1 = spec open.
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
- **Sidecar `xert_pose.json`** (one per dataset): site, root body and base transform, Euler convention, gripper
  definition, spec values and clip counts, `action_semantics` and the decode, `model_sha256`, `spec_sha256`,
  `mujoco_version`, `lerobot_version`, source hashes, the measured FK maximum error, and the statement "kinematic
  model pose of the simulated arm".
- **Refusals.** Existing output; a dataset already carrying `*.cartesian`; image-bearing input ("run add-pose
  before render-cameras"); hardware imports; missing joint or `grasp.*` columns; a model fingerprint that is not
  the current bench model; two robots or two tasks in one session; mixed action ranges.
- **Simulation data only (L-03).** Admission needs the recorded bench model fingerprint, which only simulator
  recordings and exports carry. A raw `so101_follower` dataset has no XERT manifest. `to-store` writes
  `derived_frame_pairs` episodes with no `simulation` object, and `lerobot_export._schema` needs
  `hardware=False` and `simulation_only=True`, so the real-data chain does not reach the exporter today. Nothing
  here weakens `_schema` or `to_store`.
- **Order (L-05).** `export-lerobot` -> `add-pose` -> `render-cameras`. `render-cameras` converts non-image
  features as NumPy arrays, so it carries the two float32 (7,) columns through byte-equal (tested). Existing
  `--cameras` exports must be made again in this order.

## Disagreements with the plan (reported, not redesigned)

1. **The xlerobot_left base frame is not z-up.** Plan L-01 expects "rotating `Rotation_L` (resp. `shoulder_pan`)
   alone changes the reported `[x, y, yaw]`". That holds for so101 only. Measured: `Rotation_Pitch`'s placement
   turns its local y to world up, so in its frame `Rotation_L` is the axis (0, -1, 0) (within 1e-5), and joint 1
   moves base x and z, not y; yaw is not "turn about vertical" there. For so101, `shoulder_pan` is base (0, 0, -1)
   and +0.4 rad changes yaw by exactly -0.4 (1e-12) with z, roll and pitch unchanged. The test asserts both
   measured behaviours and the frame-independent ones (height along the axis and distance from it are kept).
   **Owner decision:** keep the plan's frame (the root body's placement), or add a fixed rotation for
   xlerobot_left so that z is up as DROID-style consumers expect. Not decided here.
2. **`--rename_map` cannot select the Cartesian columns.** In LeRobot 0.6.1, `RenameObservationsProcessorStep`
   renames observation keys only (`processor/rename_processor.py`), and `rename_map` needs a pretrained
   checkpoint (`configs/train.py:247`). So `action.cartesian` cannot become the `action` key that training reads,
   and mapping `observation.state.cartesian` onto `observation.state` would collide with the joint
   `observation.state` that is still there. The camera rename below works. A Praxis fine-tune needs a dataset
   whose `observation.state` and `action` are the 7-D columns: a small follow-up (a copy with the keys swapped),
   not built here.
3. `ROOTS` lives in `xert/data/pose.py`, not next to `PROFILES` in `lerobot_export.py`, to keep the slice in one
   new module. Same table, one place.

## How to verify

Run from the worktree, one pytest at a time.

```
.venv\Scripts\python.exe -m pytest -q tests/test_pose_fk.py
.venv\Scripts\python.exe -m pytest -q
ctest --test-dir build -C Release --output-on-failure
.venv\Scripts\python.exe scripts/equivalence_check.py
set HF_HUB_OFFLINE=1
<lerobot env>\Scripts\python.exe -m pytest -q tests/test_add_pose.py tests/test_lerobot_io.py tests/test_lerobot_export.py tests/test_cameras.py
```

Measured on this branch (Windows, 2026-11-16):

| Gate | Result |
| --- | --- |
| `tests/test_pose_fk.py` (base env) | 9 passed |
| Base env, whole suite | 274 passed, 73 skipped, 15 xfailed (`test_add_pose.py` skips there: no lerobot) |
| `ctest` | 2/2 passed |
| `equivalence_check.py` | 16 seeds, identical, max observation difference 0.0 |
| `tests/test_add_pose.py` (lerobot env) | 11 passed |
| LeRobot gate (four files above) | 39 passed |

`max |FK - recorded grasp|` (world, metres) from `xert_pose.json`, 3-frame fixtures:

| Data | Max error (m) | Gripper clips (state / action) |
| --- | --- | --- |
| xlerobot_left pick_place export (first 3 steps of a P1.2 episode) | 2.26e-8 | 0 / 0 |
| so101 pick_place export (3-step hold at the reset pose) | 7.79e-9 | 0 / 0 |
| xlerobot_left pick_place teleop recording | 1.66e-8 | 0 / 0 |
| xlerobot_left stack teleop recording | 1.85e-8 | 0 / 0 |
| xlerobot_left drawer teleop recording | 1.85e-8 | 0 / 0 |

No fixture clips the gripper; clipping and its count are tested on hand values in `tests/test_pose_fk.py` (the
so101 joint limit 1.7453 -> 1.247 -> 1.0, counted). The errors are float32 storage of the joint angles, far
below the 1e-5 m bound. They are scripted fixtures, not demonstrations.

What the tests cover: base frame constant and equal to the root placement in every task scene, and the xlerobot
root's live frame turning with joint 1 (so it must not be used); joint 1 alone; a hand-built frame conversion;
random reachable and random rotations through the `"XYZ"` oracle within 1e-9 (outside a 0.05 rad band around
pitch +-pi/2); a mixed-axis case against `Rz Ry Rx`, and an intrinsic reading that fails; gripper endpoints,
midpoint and counted clipping; both robots' exports (every source column byte-equal, `LeRobotDataset` loads,
float32 (7,), `task` kept, world grasp rebuilt from the stored base-frame xyz within 1e-5); `action.cartesian`
equal to FK of the decoded labels and not of `xert.executed_action` on a fixture whose streams differ, with the
decode ranges equal to the Store's and the bench `actuator_ctrlrange`; pick_place, stack and drawer recordings
(xlerobot_left; SO-101 teleop fixtures do not exist) and a session directory; every refusal above, writing
nothing; the `add-pose` -> `render-cameras` chain (pose columns byte-equal, images identical to rendering the
plain export, MSAA off in the test only); the CLI.

## Model input tables

Praxis-1, as read from huggingface/lerobot `docs/source/praxis.mdx` on 2026-11-11 (not re-read for this slice):

| Praxis-1 input | Praxis-1 expects | XERT after `add-pose` (+ `render-cameras`) |
| --- | --- | --- |
| `observation.state` | 7-D `[x, y, z, roll, pitch, yaw, gripper]`, base frame, metres, extrinsic xyz radians, gripper [0, 1] | `observation.state.cartesian` (same layout); `observation.state` stays joint radians |
| `action` | 7-D absolute Cartesian targets, same layout | `action.cartesian` (FK of the training labels); `action` stays joint targets |
| `observation.images.exterior` | external camera, any size (tiled to 224x416) | `observation.images.scene` (rename at training) |
| `observation.images.wrist` | wrist camera | `observation.images.wrist` |
| `task` | instruction string | `task` |
| rate | 15 Hz, 24-step chunks | 25 Hz (`control_hz`) |

SmolVLA in LeRobot 0.6.1 (`policies/smolvla/configuration_smolvla.py`): any state and action up to 32-D
(`max_state_dim`, `max_action_dim`), images padded to 512x512, 50-step chunks, MEAN_STD state/action. It needs
nothing from this slice:

| SmolVLA input | SmolVLA expects | XERT |
| --- | --- | --- |
| `observation.state` | <= 32-D | 6 joint radians (recordings) or 26-D bench state (exports); or the 7-D Cartesian column |
| `action` | <= 32-D | 6 joint targets |
| `observation.images.*` | one or more cameras | `scene`, `wrist` |
| `task` | instruction string | `task` |

Camera rename at training time (LeRobot `--rename_map`, which needs `--policy.path` to a pretrained checkpoint):

```
--rename_map='{"observation.images.scene": "observation.images.exterior"}'
```

## Not in this slice

1. Hardware data (slice L2). It needs: hardware provenance kept end to end (calibration sha256, robot id, the
   joint-map identity from slice F1); an explicit, recorded choice of FK model (the Menagerie kinematic model is
   not a measurement of Daniel's arm); and no fabricated simulation provenance.
2. 25 Hz -> 15 Hz for Praxis: an owner decision between resampling (an interpolation rule for actions and a frame
   choice for images) and recording new data at 15 Hz (`control_hz` in the specs; changes `contract_sha256`).
3. A dataset whose `observation.state`/`action` are the Cartesian columns (Disagreement 2).
4. IK execution of Cartesian commands on the arm (CORE).
5. Any Praxis or SmolVLA training run. That needs real demonstrations and, for Praxis, one 80 GB GPU for LoRA
   (per the guide's measurements). The laptop has 12 GB.

Gimbal lock: at pitch = +-pi/2, roll and yaw are not unique. The matrix is exact; Euler is only a
representation. float32 storage of metres loses about 1e-7 m, below the tolerance.

## Owner decisions (recorded, not made)

- Gripper direction: 0 = spec closed, 1 = spec open. Praxis's guide does not state DROID's direction; check
  before a fine-tune and flip in a copy if needed.
- 15 Hz: resample or re-record (above).
- xlerobot_left base orientation (Disagreement 1).

TODO(daniel): none new.

## Questions an interviewer could ask

1. Why a fixed base transform and not the root body's live frame? (The xlerobot root carries joint 1, so its live
   frame turns with the arm; a pose in it would hide joint 1. The fixed placement also cancels where the bench put
   the arm, so sim and real poses compare.)
2. Why is `action.cartesian` FK of the training labels, and when does that differ from the executed command?
   (Exports put corrective labels in `action` when present; in DAgger data the arm executes `dagger_action`'s
   output while the label is the expert's, so the two streams differ.)
3. What breaks if the Euler convention is intrinsic? (Intrinsic x-y-z is `Rx Ry Rz`: the same three numbers name a
   different rotation, so a model trained on DROID-convention data would be fed wrong orientations; the test's
   intrinsic reading misses by more than 0.1.)
