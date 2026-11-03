# Slice D: real-vs-sim replay (sysid step 1)

Branch `feat/d-replay-real`, from `integration/sim-mvp` @ b0f695c. The QA fix round merged
`integration/sim-mvp` @ 582f6bb (slice B, LICENSE, camera keys). This slice is tooling, tests, plots and docs.
Parameter fitting is CORE and is left to Daniel as a stub. The builder wrote no ledger rows, used no hardware
or camera, and added no dependency. Every test uses synthetic episodes simulated by the same model.

## What changed
- `xert/eval/replay_real.py` adds `xert replay-real --dataset <import dir> --output <fresh> [--ledger]`.
  - **Input contract (R3).** The command takes only slice C's frame-pair import (`frames.npz` + `import.json`)
    from hardware path (c). It refuses these inputs:
    - a directory without `import.json` that has `meta/info.json`, which is a direct LeRobot dataset ("run
      `xert import-lerobot --motor-units degrees` first");
    - `derived: true`, or any `source_kind` other than `so101_follower`, which covers path (b) sim recordings
      and path (a) Stores;
    - `units` other than `radians`, such as `raw_lerobot` or a missing value;
    - no 64-hex `joint_map.sha256`;
    - a `spec_sha256` that differs from the current `so101.yaml`, or state/action columns that are not the six
      `<joint>.pos` names;
    - timestamps that differ from `frame_index / fps` by more than 1e-4 s. LeRobot writes
      timestamp = frame_index/fps, so this check catches edited or resampled frames. It cannot detect
      control-loop overruns;
    - an action more than 1e-6 rad past a spec limit (owner decision: the same slack as `to-store`). An action
      inside that slack is clipped to the limit, and the largest excess is recorded as
      `action_max_limit_excess_rad`. `MuJoCoBackend` keeps its own 1e-10 check for every other caller.
  - **Timing.** k = ceil(1/(fps·asset_dt)) physics steps per frame, and dt = 1/(fps·k).
    - At 30 fps, k = 7 and dt = 1/210 s (0.0047619 s). At 25 fps, k = 8 and dt = 0.005 s, the asset value.
    - The backend refuses unless fps·dt·k = 1 to 1e-9, and it checks that sim time advances 1/fps per frame.
    - Every recorded frame is used.
    - **dt is not physics-neutral.** QA measured per-joint RMSE between replays that differed only in k:
      - 1/210 s vs 1/240 s: 0.0016–0.0071 rad;
      - 1/210 s vs 1/420 s: 0.0028–0.0118 rad;
      - 1/210 s vs 1/180 s: 0.0033–0.0079 rad;
      - at 25 fps, 1/225 s vs 0.005 s: 0.0028–0.0080 rad.

      That is about 1/4 to 1/2 of the kp×0.3 effect. Results, and any fit made from them, hold for the dt
      recorded with them. `NOTES`, `summary.md` and `replay.json` say so.
  - **Per episode (R8).** `start_episode` builds a new `MuJoCoBackend` (new MjModel and MjData) for every
    episode. It sets qpos to the recorded first state and qvel to 0. `run_episode` holds `action[t]` for one
    frame.
  - **Moving start (owner decision).** The arm is still assumed at rest at the start of each episode, but an
    episode is flagged `moving_start: true` when a body joint moves more than 2 encoder steps between frames 0
    and 1.
    - The threshold is max |state[1]-state[0]|·fps > 2·(2π/4096 rad)·fps, which is 0.092 rad/s at 30 fps.
    - An STS3215 step is 0.088°, and an arm at rest can read ±1 step between frames.
    - The gripper is excluded, because its range_0_100 step depends on its calibration.
    - The per-episode `start_speed_rad_s` and the flag are written to `replay.json` (`episode_starts`) and
      `summary.md`. Flagged episodes are not trimmed, and no velocity is estimated.
  - **Alignment.** sim[0] = state[0], and sim[t+1] follows action[t]. RMSE uses frames 1..T-1, so n = T-1 per
    episode per joint. Pooled RMSE is computed over all compared frames, and its n is their sum.
  - **Outputs:**
    - `trace.csv`, with plain floats per frame: sim time, real and sim positions;
    - `summary.csv` and `summary.md`, with RMSE per episode and pooled, per joint, in radians plus derived
      degrees, and n;
    - `plots/<joint>.png`, with real vs sim and the error trace below (Agg backend). These are written only
      when matplotlib is installed. The base env records `plots: skipped ...` and still writes the tables;
    - `replay.json`, with the import, frames, joint-map, spec and model sha256, fps, dt, substeps, the limit
      excess, the episode starts, and the sha256 of every other artifact.
  - **Ledger (R6).** `--ledger` writes one row per joint for pooled RMSE. Each row has `backend: "mujoco"`,
    `reference_backend: "so101_hw"`, `import_sha256`, `joint_map_sha256`, `calibration_sha256` and `robot_id`,
    plus `notes` saying the replay is open loop, is not system identification, and is tied to its dt. The
    command refuses unless both of these hold:
    - the import's `raw/` copy still hashes to `source_sha256`, and `lerobot_robot_type` is a hardware type;
    - `import.json` has a calibration sha256 and a robot id (owner decision).

    Without `--ledger`, no rows are written.
  - `fit_actuator_params`: the `TODO(daniel)` stub.
- `xert/data/lerobot_io.py`, two owner decisions:
  - `import-lerobot --calibration <robot.id>.json` (path (c) only, optional) checks that the file is a LeRobot
    so101 follower calibration and records its sha256 and the robot id in `import.json`.
    - Per LeRobot 0.6.1 `robots/robot.py:49-56`, the calibration file is `<calibration_dir>/<robot.id>.json`.
      The dataset meta records no robot id, so the id comes from the file name.
    - Plain imports are unchanged.
  - `to-store` appends `environment_state` to the observations when the import has it (path (b)). The named
    columns go after the joint state, in recorded order. Path (c) imports stay joint-only. The derived metadata
    records `observation_contents` and the full `observation_names`.
- `xert/backends/mujoco_backend.py`: a new optional `timestep=` argument. Existing callers are unchanged.
- `xert/eval/ledger.py`: a row with `reference_backend` must use `backend: mujoco` and
  `reference_backend: so101_hw`. It must carry a 64-hex `import_sha256` and `calibration_sha256`, and a
  nonempty `robot_id`.
- `scripts/resume_numbers.py`: these rows print as `backend simulation replay against hardware recording
  (mujoco vs so101_hw)`, never as `so101_hw (hardware)`.
- `.github/workflows/optional.yml`: `tests/test_replay_real.py` is added to the lerobot and teleop jobs. The
  base `linux.yml` already runs the whole suite.

## Why
- **Only path (c).** Replay against hardware has meaning only for radians that went through a measured
  joint map. A missing map sha means the units are unproven.
- **Change dt, not the recording.** Resampling would create states that were never recorded. The cost is
  that the results depend on dt (QA's numbers above), so dt is recorded with every result.
- **New backend per episode.** Nothing can carry over. This holds by construction, not through a reset
  routine that a later edit could weaken. The model loads in about 0.08 s.
- **Opt-in ledger, raw/ plus calibration.** Synthetic test imports have no `raw/` copy. The calibration sha
  and robot id tie a row to one physical, calibrated arm. A fabricated calibration file would still pass,
  because this is provenance and not proof.

## Measured (tests/test_replay_real.py, 2026-11-02, Windows)
- **Self-replay.** A synthetic two-episode recording (20 frames each, different start poses) was simulated
  with raw MuJoCo at the replay dt. Replay RMSE was **exactly 0.0** on every joint and both episodes. The test
  tolerance is 1e-12 rad, as headroom for other platforms. `trace.csv` parses back bit-exact.
- **Gain mismatch.** The "recording" used kp × 0.3. Pooled RMSE was **0.015–0.035 rad** per joint. The test
  asserts min > 1e-3.
- **No reset.** A test helper reuses one backend across episodes. Episode 2 RMSE was **0.06–0.27 rad** per
  joint, and the test asserts max > 0.1. With the reset, the same episode gives 0.0.
- **Timing.** Sim time per frame is 1/30 s to 1e-12. dt = 1/210 s with 7 substeps is recorded in
  `replay.json`.
- **Limits.** A gripper command at limit + 5e-7 rad is accepted, and its excess is recorded. At
  limit + 2e-6 rad it is refused.

## How to verify
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_core.ps1   # after the merge: slice B changed the C++
cmake --build build/lerobot --config Release --target _reach_core             # the cp313 module for the LeRobot/combined envs
.\.venv\Scripts\python.exe -m pytest -q                       # 187 passed, 39 skipped, 5 xfailed, 8 subtests
ctest --test-dir build -C Release                              # 2/2
.\.venv-lerobot\Scripts\python.exe -m pytest -q tests/test_lerobot_io.py tests/test_replay_real.py tests/test_ledger.py tests/test_resume_numbers.py   # 42 passed, 1 xfailed
C:\Users\Khang\tmp\reach-public\build\venv-teleop\Scripts\python.exe -m pytest -q tests/test_replay_real.py tests/test_lerobot_io.py   # 31 passed, 1 xfailed
```
- Five xfails are expected: this slice's fit, three slice B `so101_expert` stubs, and the P6.2 e-stop
  blocker.
- The combined env has no fastapi, so a bare `pytest -q` stops at collection (`test_audit.py`,
  `test_backend.py`).

Real-arm path (not run here; no hardware). Camera keys are `scene` and `wrist` (docs/CAMERA_KEYS.md). LeRobot
0.6.1 requires a teleoperator to record (`lerobot_record.py:192-197`).
```powershell
lerobot-record --robot.type=so101_follower --robot.port=<COM?> --robot.id=<id> --robot.cameras="{scene: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --teleop.type=so101_leader --teleop.port=<COM?> --teleop.id=<id> --dataset.repo_id=<user>/<name> --dataset.root=<dir> --dataset.fps=30 --dataset.num_episodes=<n> --dataset.single_task="<task>" --dataset.push_to_hub=false   # follower with use_degrees=True
python -m xert import-lerobot --dataset <dir> --output <import> --motor-units degrees --calibration <HF_LEROBOT_CALIBRATION>/robots/so_follower/<id>.json   # needs the measured P6 joint map (no TBD)
python -m xert replay-real --dataset <import> --output <replay>                         # add --ledger for the evidence rows
python -m scripts.resume_numbers
```

## Interview questions
1. *Why is this not system identification yet?* It replays recorded goal positions open loop through the
   asset's parameters and only measures the gap. Nothing is fitted yet. The STS3215's internal PID is also not
   MuJoCo's position actuator, so even a low RMSE would not show that the parameters are right.
2. *Why change the timestep instead of resampling the 30 fps recording, and what does it cost?* Resampling
   would invent states that were never recorded. So each 30 fps frame is exactly 7 physics steps of 1/210 s.
   QA measured that changing k alone moves per-joint RMSE by up to about 0.012 rad, so results and any fitted
   parameters hold only for the dt recorded with them.
3. *How do you know one episode doesn't contaminate the next?* Every episode gets a new MjData at its own
   recorded first state. A test shows that reusing one simulator across episodes gives 0.06–0.27 rad RMSE in
   episode 2, while the per-episode reset gives exactly 0.

## TODO(daniel)
- `xert/eval/replay_real.py:269` `fit_actuator_params(dataset, *, held_out_episode=-1)`: fit kp, kv, damping,
  armature and frictionloss per joint by minimizing replay tracking RMSE, and report a held-out episode.
  - The docstring gives the inputs, outputs and invariants.
  - Fit at the same dt you will use later, because dt is not physics-neutral.
- `tests/test_replay_real.py:221` `test_fit_recovers_a_known_gain_and_generalizes_to_a_held_out_episode` is a
  strict xfail that defines done.
  - On a kp × 0.5 synthetic recording, held-out RMSE must be under 0.1 × the baseline.
  - The fitted kp must be within 10% of 499.11.
  - Remove the marker when it passes.

## README lines integration may add
- `python -m xert replay-real --dataset <import-lerobot output> --output <fresh> [--ledger]` runs an open-loop
  MuJoCo replay of a recorded SO-101 run. It gives RMSE per joint per episode, moving-start flags and plots.
  It is not system identification, and its results are tied to the recorded dt.
- `import-lerobot --calibration <robot.id>.json` records the follower calibration sha256 and robot id.
  `replay-real --ledger` requires them.
- Ledger rows with `reference_backend: so101_hw` are simulation replays against a hardware recording.

## Open questions
- None from this round. The three earlier questions (ledger provenance, limit slack, moving start) were
  settled by owner decisions and are applied above.
