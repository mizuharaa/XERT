# Simulation method and limits

## What is actually modeled

This is a stationary, single-left-arm XLeRobot bench task in MuJoCo. The rest of
the mobile robot is not simulated. The arm's joint transforms, meshes and body
inertias come from [Vector-Wangel/XLeRobot](https://github.com/Vector-Wangel/XLeRobot),
revision `749abc837d5d771f26aeff961009c290e574b024`, Apache-2.0. The upstream license
is preserved in `assets/xlerobot/LICENSE`; `SOURCE.json` records SHA-256 hashes
of the source XML and every downloaded mesh. `scripts/fetch_robot_assets.py`
fetches this revision, not the latest branch.

The benchmark extracts that arm into a new scene with a 20 cm mount, a table,
a tray, and a 16 x 16 x 32 mm, 15 g block. Mount height, servo parameters and
contact friction are **bench approximations**, not calibrated WowRobo hardware
parameters. The model is useful for implementing and testing the pipeline; it
does not establish hardware transfer or safety.

## From a target to contact

`expert_action()` is a scripted teacher with seven phases: approach, descend,
close, lift, transfer, lower, release/retract. It uses the initial simulator
object position and damped least-squares inverse kinematics to obtain joint
targets. Its wrist orientation keeps the gripper approach axis vertical.
Target changes are rate-limited before reaching position servos.

The block is a free rigid body throughout the episode. The teacher only writes
arm actuator targets: there are no attachment equality constraints, object
teleports, pose overwrite during stepping, or fabricated success transitions.
Grasping is produced by finger/block contact and friction.

The initial version pushed the block aside instead of lifting it. The fix was
to inspect the actual fixed/moving jaw mesh extents, place the grasp site between
the contact surfaces, raise the arm mount to make the vertical approach reachable,
and reduce jaw gain/force. The successful configuration is a 16 mm-wide object,
grasp site `(-0.003, -0.10, 0)` in the fixed-jaw frame, and jaw servo `kp=5`,
`kv=0.1`, force range `[-0.3, 0.3]`. Other arm servos use `kp=60`, `kv=2`,
force range `[-3, 3]`. These are explicitly tunable simulation parameters,
not measurements of motor performance.

## Data contract

- Physics timestep: 0.002 s; 20 physics steps per command, or 25 Hz control.
- Six actions: absolute joint position targets normalized to `[-1, 1]` using
  the model's actuator ranges. Joint order is Rotation, Pitch, Elbow,
  Wrist Pitch, Wrist Roll, Jaw, all for the left arm.
- Actions are quantized to float32 at the physics input boundary because the
  recorder persists float32. Without this, tiny rounding differences between
  live float64 commands and saved commands change contact dynamics during replay.
- Twenty-six observations: six joint positions, six joint velocities, block
  position/quaternion, grasp position, tray position, and elapsed horizon fraction.
- Coordinates: world-frame metres; quaternions in `wxyz` order; radians and rad/s.
- Observations are privileged simulator state. This is **not** a camera-trained
  manipulation policy. `render()` supplies an RGB inspection view separately.
- `delay_steps` delays actuator commands using a FIFO. Friction changes the
  sliding-friction coefficient of scene geoms. Seeds set initial object position.
- `reset()` returns the first observation; `step()` returns observation, binary
  success reward, episode-finished flag, and a structured outcome.

Every environment verifies the downloaded asset bytes against `SOURCE.json`
before compilation. The model identity hashes the generated XML with its mesh
directory normalized, plus source provenance and mesh hashes. Moving a checkout
therefore preserves identity; changed geometry changes identity. The manifest is
provenance/integrity bookkeeping, not a signed supply-chain authenticity proof.
The metadata also records the simulator implementation's SHA-256 and MuJoCo
version. This conservatively invalidates artifacts after simulator code changes,
including comment-only edits, so unchanged XML cannot hide changed task semantics.

## What counts as success

The block must previously have risen above 0.145 m. It must then remain within
0.029 m of the tray center on each horizontal axis, between 0.095 and 0.115 m
height, moving below 0.025 m/s, and **out of robot contact** for 12 consecutive
control steps (0.48 s). Merely pushing a block into the tray or holding it over
the tray is insufficient. A fall below 0.04 m terminates with `fell_off_table`;
other unsuccessful runs end as `timeout_no_lift` or `timeout_after_lift`.
This task-specific geometric criterion is not a collision/safety certificate.

## Checks executed

`python -m unittest discover -s tests -p test_sim.py -v` passed four tests:
physical contact pick/release, deterministic replay with delayed commands,
input validation, and asset integrity/portable identity (5.430 s in this run).

The scripted teacher was also run at default settings on seeds
`0, 1, 2, 3, 4, 17, 42`. All seven reached success at step 426, after a physical
lift and release, in a combined 26.548 s of wall time after float32 canonicalization.
Raw outcomes and contract hashes are preserved in `docs/SIMULATION_CHECK.json`.
The different seeds sample
the small configured initial-position rectangle. This is a local regression
check, **not** a statistical robustness claim or a learned-policy result.

Rendering was checked sequentially on two separate worker threads, each returning
`uint8` RGB of shape `(480, 640, 3)`. Each render creates and closes its own GL
context on that thread; a context is not cached and handed between HTTP workers.
This costs context startup per frame. Introduce a dedicated render thread only
if measured rendering throughput requires it.

## Rebuild it by hand

1. Load the pinned arm; print joint names, ranges and mesh bounds before control.
2. Implement normalized-target decoding and test a single joint without a block.
3. Implement seeded reset and observation order; verify identical replay twice.
4. Add the block, tray and contact-based success evaluator before writing a teacher.
5. Implement position Jacobian IK, then the orientation term and damping.
6. Tune approach/close/lift independently. Inspect contacts and local coordinates;
   do not compensate for a failed grasp by attaching the object programmatically.
7. Add delay and friction perturbations only after the default grasp works.
8. Save observations/actions and train a baseline; report its rollout outcomes
   separately from this scripted teacher's performance.

No physical robot or motor interface was used. Future real-robot execution needs
measured joint calibration, camera/robot frame calibration, control timing,
actuator limits, a stop mechanism and supervised low-speed validation.
