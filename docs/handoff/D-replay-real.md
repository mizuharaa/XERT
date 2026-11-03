# Slice D: real-vs-sim replay (sysid step 1)

Branch `feat/d-replay-real`, from `integration/sim-mvp` @ b0f695c. This slice is tooling, tests, plots and
docs. Parameter fitting is CORE and left to Daniel as a stub. No ledger rows were written, no hardware or
camera was used, and no dependency was added. Every test uses synthetic episodes simulated by the same model.

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
    - timestamps that differ from `frame_index / fps` by more than 1e-4 s, because the replay never resamples.
  - **Timing.** k = ceil(1/(fps·asset_dt)) physics steps per frame, and dt = 1/(fps·k), which is never larger
    than the asset's 0.005 s. At 30 fps that gives k = 7 and dt = 1/210 s (0.0047619 s). At 25 fps it gives
    k = 8 and dt = 0.005 s, the asset value. The backend refuses unless fps·dt·k = 1 to 1e-9. Each frame
    checks that sim time equals frame/fps to 1e-9 s. Every recorded frame is used.
  - **Per episode (R8).** `start_episode` builds a new `MuJoCoBackend` (new MjModel and MjData) for every
    episode. It sets qpos = ctrl = the recorded first state, with qvel 0. `run_episode` holds `action[t]` for
    one frame, and `send_targets` refuses targets outside the spec limits instead of clipping them.
  - **Alignment.** sim[0] = state[0], and sim[t+1] follows action[t]. RMSE uses frames 1..T-1, so n = T-1 per
    episode per joint. Pooled RMSE is computed over all compared frames, and its n is their sum.
  - **Outputs:**
    - `trace.csv` (per frame: sim time, real and sim positions);
    - `summary.csv` and `summary.md` (RMSE per episode and pooled, per joint, in radians, with degrees derived,
      and n);
    - `plots/<joint>.png` (real vs sim with the error trace below, Agg backend). These exist only when
      matplotlib is installed. The base env records `plots: skipped ...` and still writes the tables;
    - `replay.json`, with the import, frames, joint-map, spec and model sha256, fps, dt, substeps, and the
      sha256 of every other artifact.
  - **Ledger (R6).** `--ledger` writes one row per joint for pooled RMSE: `backend: "mujoco"`,
    `reference_backend: "so101_hw"`, `import_sha256` and `joint_map_sha256`, plus `notes` saying this is
    open-loop replay and not system identification. It is refused unless the import's `raw/` copy still
    hashes to `source_sha256` and `lerobot_robot_type` is a hardware type. Without `--ledger`, no rows are
    written.
  - `fit_actuator_params`: the `TODO(daniel)` stub.
- `xert/backends/mujoco_backend.py`: a new optional `timestep=` argument. Existing callers are unchanged.
- `xert/eval/ledger.py`: when `reference_backend` is present, it must be `so101_hw`, `backend` must be
  `mujoco`, and `import_sha256` must be 64 lowercase hex.
- `scripts/resume_numbers.py`: these rows print as `backend simulation replay against hardware recording
  (mujoco vs so101_hw)`, never as `so101_hw (hardware)`.
- `.github/workflows/optional.yml`: `tests/test_replay_real.py` is added to the lerobot and teleop jobs. The
  base `linux.yml` already runs the whole suite.

## Why
- **Only path (c).** Replay against hardware has meaning only for radians that went through a measured
  joint map. A missing map sha means the units are unproven.
- **Change dt, not the recording.** Resampling would create states that were never recorded. Stepping at
  dt ≤ the asset dt keeps the implicitfast integrator at least as fine as the asset intended.
- **New backend per episode.** Nothing can carry over, by construction rather than by a reset routine that a
  future edit could weaken. The model loads in about 0.08 s.
- **Opt-in ledger with a raw/ check.** Synthetic test imports have no `raw/` copy, so they cannot write a row.
  The command cannot tell a synthetic LeRobot dataset from a real one (see Open questions).

## Measured (tests/test_replay_real.py, 2026-11-02, Windows)
- Self-replay: a synthetic two-episode recording (20 frames each, different start poses) simulated with raw
  MuJoCo at the replay dt. Replay RMSE was **exactly 0.0** on every joint and both episodes. The test tolerance
  is 1e-12 rad, as headroom for other platforms.
- Gain mismatch: the "recording" used kp × 0.3. Pooled RMSE was **0.015–0.035 rad** per joint. The test
  asserts min > 1e-3.
- No reset (a test helper that reuses one backend): episode 2 RMSE was **0.06–0.27 rad** per joint. The test
  asserts max > 0.1. With the reset, the same episode gives 0.0.
- Timing: sim time per frame is 1/30 s to 1e-12, and dt = 1/210 with 7 substeps is recorded in `replay.json`.

## How to verify
```powershell
.\.venv\Scripts\python.exe -m pytest -q                       # 172 passed, 37 skipped, 2 xfailed, 8 subtests (baseline 156/36/1/8)
ctest --test-dir build -C Release                              # 2/2
.\.venv-lerobot\Scripts\python.exe -m pytest -q tests/test_lerobot_io.py tests/test_replay_real.py tests/test_ledger.py tests/test_resume_numbers.py   # 39 passed, 1 xfailed
C:\Users\Khang\tmp\reach-public\build\venv-teleop\Scripts\python.exe -m pytest -q <optional.yml teleop list> tests/test_resume_numbers.py   # 135 passed, 1 skipped, 1 xfailed
```
The combined env has no fastapi, so a bare `pytest -q` stops at collection (`test_audit.py`,
`test_backend.py`) both before and after this slice. It runs the optional.yml teleop file list instead. That
list measured 115 passed and 1 skipped at baseline.

Real-arm path (not run here; no hardware):
```powershell
lerobot-record --robot.type=so101_follower --robot.port=COM? --robot.id=<id> --dataset.repo_id=<user>/<name> --dataset.root=<dir> --dataset.fps=30 --dataset.num_episodes=<n> --dataset.single_task="<task>" --dataset.push_to_hub=false   # follower with use_degrees=True
python -m xert import-lerobot --dataset <dir> --output <import> --motor-units degrees   # needs the measured P6 joint map (no TBD)
python -m xert replay-real --dataset <import> --output <replay>                         # add --ledger for the evidence rows
python -m scripts.resume_numbers
```

## Interview questions
1. *Why is this not system identification yet?* It only replays recorded goal positions open loop through
   the asset's parameters and measures the gap, and nothing is fitted yet; besides, the STS3215's internal PID
   is not MuJoCo's position actuator, so a low RMSE would still not show that the parameters are right.
2. *Why change the timestep instead of resampling the 30 fps recording?* 30 fps does not divide the 0.005 s
   asset step, and resampling would invent states that were never recorded, so the sim steps at
   dt = 1/(30·7) s and each recorded frame lasts exactly 7 physics steps.
3. *How do you know one episode doesn't contaminate the next?* Every episode gets a new MjData at its own
   recorded first state, and a test shows that reusing one simulator across episodes gives 0.06–0.27 rad
   RMSE in episode 2, while the per-episode reset gives exactly 0.

## TODO(daniel)
- `xert/eval/replay_real.py:234` `fit_actuator_params(dataset, *, held_out_episode=-1)`: fit kp, kv, damping,
  armature and frictionloss per joint by minimizing replay tracking RMSE, and report a held-out episode. The
  docstring gives the inputs, outputs and invariants.
- `tests/test_replay_real.py:167` `test_fit_recovers_a_known_gain_and_generalizes_to_a_held_out_episode`
  (strict xfail) defines done. On a kp × 0.5 synthetic recording, held-out RMSE must be under 0.1 × the
  baseline, and the fitted kp must be within 10% of 499.11. Remove the marker when it passes.

## README lines integration may add
- `python -m xert replay-real --dataset <import-lerobot output> --output <fresh> [--ledger]`: open-loop
  MuJoCo replay of a recorded SO-101 run, giving RMSE per joint per episode and plots. This is not system
  identification.
- Ledger rows with `reference_backend: so101_hw` are simulation replays against a hardware recording.

## Open questions
- `--ledger` trusts that the operator gives it a real recording. A synthetic LeRobot dataset passed through
  `import-lerobot` also has a verified `raw/` copy. Should the import record something stronger, such as the
  robot id or calibration file hash?
- Recorded actions outside the spec limits are refused (backend tolerance 1e-10 rad), not clipped. Real
  gripper commands may sit at or past a limit. Should replay use C's 1e-6 rad slack?
- The start velocity is assumed to be 0, because the import has no velocity. Should episodes that begin in
  motion be trimmed, or should velocity be estimated from the first two states?
