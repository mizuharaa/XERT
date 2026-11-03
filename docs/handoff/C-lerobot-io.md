# Slice C: LeRobot import/export CLI

Branch `feat/c-lerobot-io`, from `integration/sim-mvp` @ 5cd6853. This slice is data tooling only: no CORE
logic, no ledger rows, no hardware, no new dependency. The tests use scripted and synthetic data only.

## What changed
- `xert/data/lerobot_io.py` adds three commands, wired into `xert/__main__.py`.
  - `xert export-lerobot --episodes <collection.json | recording | session> --output <fresh> --task "<text>" [--cameras] [--robot-type]`
    - An NPZ `collection.json` (its Store is the sibling `dataset/`) goes through the P2.1 `export_episodes`.
    - A runtime recording is already LeRobot v3. It is copied verbatim and checked by hash.
    - The task text must match the recorded task. The command does not relabel recordings.
    - `--cameras` then runs P2.3 `render_cameras`. This works where render-cameras works (the XLeRobot bench); SO-101 is slice B2.
    - Unlike `python -m xert.data.lerobot_export`, it writes no ledger row.
  - `xert import-lerobot --dataset <LeRobot v3.0> --output <fresh> [--joint-map] [--motor-units degrees | --raw-units]` chooses its path from the manifest (`source_kind`).
    - `xert_export.json` → **(a)**: a Store at `dataset/` plus `collection.json`. Episodes get new IDs. The arrays are identical: executed actions, corrective labels, rewards, flags and the terminal observation (rebuilt from the last `xert.next_observation`, after checking `state[t+1] == next_obs[t]`).
      - **The terminal observation is trusted, not cross-checked.** No later row holds it, so the successor chain cannot confirm it. The manifest's original NPZ sha256 cannot be recomputed either, because `np.savez_compressed` writes zip timestamps. A corrupted final `xert.next_observation` would be restored as is.
    - `xert_recording.json` → **(b)**: `frames.npz` with `episode_index`, `frame_index`, `timestamp`, and float32 `state`/`action` in radians as recorded. When the recording has it, `environment_state` is included too. `import.json` holds the column names and the bench `model_sha256`.
    - No XERT manifest, with `robot_type` `so101_follower` or `so_follower` → **(c)**. The state and action names must be exactly the six `<motor>.pos` names. The values become float64 radians only when all of these hold:
      - `--motor-units degrees` is passed explicitly. It has no default because LeRobot 0.6.1 records `use_degrees` nowhere: not in `info.json`, not in the calibration file, not in `lerobot_record`. RANGE_M100_100 values fall inside degree ranges and could not be detected.
      - `load_joint_map(..., dry_run=False)` accepts the map. It refuses any TBD value and now validates `units` against MotorNormMode.
      - The map's units are `degrees` for all five body joints and `range_0_100` for the gripper.
      - `import.json` records the map's sha256, `motor_units` and `use_degrees_assumed: true`. With `--raw-units`, the values stay native and are labelled `raw_lerobot`.
    - If both manifests are present, the dataset is refused. A dataset that is not v3.0 is refused. Any other dataset is refused.
    - Every path keeps `raw/`, a bit-exact copy of the source (sha256 checked against the source), and writes `import.json` with provenance, units and the per-file sha256 of the source.
    - The frame-pair format has no rewards, no terminal successor and no termination flags (`import.json` `absent`). Features it does not convert (images, `xert.*`) are listed in `not_imported`.
    - Rows are always sorted by (`episode_index`, `frame_index`). The importer refuses an episode whose `frame_index` is not contiguous from 0.
  - `xert to-store --frames <import dir> --output <fresh>` is the **derived** conversion. It accepts only (b)/(c) imports in radians and refuses `raw_lerobot`. For each episode with T_e frames:
    - it writes T_e-1 transitions;
    - it normalizes actions to [-1,1] by the spec `position_limits` (with 1e-6 rad slack, the same as the recorder) and refuses out-of-limit actions instead of clipping them;
    - reward is 0, `terminated` stays false, and the last transition is truncated;
    - each episode's metadata has `derived: true` and the sha256 of the import.
- `docs/CAMERA_KEYS.md`: the rule, the evidence for it (file:line) and the findings.
- `tests/test_lerobot_io.py`: 2 tests that need no LeRobot (dispatch and refusals, `to-store` refusals) and 11 tests in the LeRobot env. Mutation checks were run by hand: deleting any one of these makes a test fail:
  - the successor-chain check;
  - the action/executed equality check without provenance;
  - the row sort;
  - frame_index contiguity;
  - the 1e-6 rad slack (both set to 0 and widened to 1e-5);
  - the to-store spec-sha check.
- `.github/workflows/optional.yml`: the new test file is added to the lerobot and teleop jobs.

## Why
- **Dispatch by manifest.** The three sources use different units: normalized NPZ labels, recorded radians, and servo degrees. Guessing the units from feature names would turn a wrong guess into a wrong number later. An unknown source is refused.
- **Separate `to-store` subcommand rather than a flag.** It is the smallest option that keeps the import faithful.
  - The derived Store gets its own fresh directory and provenance, so it cannot be mistaken for imported data.
  - Its refusal of `raw_lerobot` checks the import's declared units, so it also covers an import made earlier.
  - A flag on `import-lerobot` would mix faithful and derived outputs in one directory.
- **float64 radians on path (c), with no exact re-encode check at runtime.** float32 radians would lose precision.
  - A bit-exact check refused honest data. LeRobot writes exact 0.0 at the servo mid position (`motors_bus.py:877`), and with a nonzero offset float64 cancellation brings it back as about 1e-15.
  - That check also only proved that the formula undoes itself.
  - The test now checks re-encoding to 1e-9 motor units, with a zero reading on elbow_flex at offset 7.25. One servo step is 0.088 degrees.
- **robot_type `so_follower`.** LeRobot 0.6.1 `lerobot-record` writes `robot.name`, and `SO101Follower = SOFollower` has `name = "so_follower"` (`so_follower.py:44,242`; `lerobot_record.py:443`). The plan's `so101_follower` is what older releases wrote. Both are accepted.

## Verify (each command run once, one at a time)
```powershell
.\.venv\Scripts\python.exe -m pytest -q                      # measured: 153 passed, 36 skipped, 1 xfailed, 8 subtests
ctest --test-dir build -C Release                             # measured: 2/2
.\.venv-lerobot\Scripts\python.exe -m pytest -q tests/test_lerobot_export.py tests/test_recorder.py tests/test_cameras.py tests/test_lerobot_io.py   # measured: 34 passed
C:\Users\Khang\tmp\reach-public\build\venv-teleop\Scripts\python.exe -m pytest -q tests/test_lerobot_export.py tests/test_recorder.py tests/test_cameras.py tests/test_lerobot_io.py   # 29 passed before the QA fix round; not re-run since
```
Baseline at 5cd6853: base 151 passed / 25 skipped; both LeRobot envs 21 passed on the three old files.

## Interview questions
1. *Why does the importer refuse an SO-101 dataset when the joint map has TBD values?* Converting degrees to radians needs the measured sign, scale and zero offset, and guessing them would put a silent offset into every later comparison with sim, so the data stays in raw units until P6.1 measures the map.
2. *Why is `to-store` a separate command and labelled derived?* LeRobot frame pairs have no rewards, no termination flags and no successor for the last frame, so a Store episode has to invent zeros and truncate, and that must never look like recorded data.
3. *How do you know two episodes never leak into one transition?* Every consumer groups rows by `episode_index` and orders them by `frame_index`, and a test with two episodes, different start poses and restarting timestamps checks that each Store episode holds only its own frames (2 and 3 transitions).

## TODO(daniel)
None. This slice has no CORE logic.

## README lines integration may add
- `python -m xert export-lerobot --episodes <collection.json|recording> --output <fresh> --task "<text>" [--cameras]`: wraps P2.1 + P2.3.
- `python -m xert import-lerobot --dataset <LeRobot v3> --output <fresh> (--motor-units degrees | --raw-units)`: manifest-dispatched (XERT export / XERT recording / so101_follower). `python -m xert to-store --frames <import> --output <fresh>`: derived Store episodes.
- Camera naming: see `docs/CAMERA_KEYS.md`.

## Open questions
- Hardware camera config keys. Should they be `scene`/`wrist`, so one policy reads sim and real data without a rename? See `docs/CAMERA_KEYS.md`.
- `to-store` uses only joint `state` as observations, so path (b) `environment_state` is not in the derived Store. Should it be appended?
- Path (c) cannot tell from the dataset whether it was recorded with `use_degrees=True`, so the operator must assert it with `--motor-units degrees`. Supporting `range_m100_100` would need its own map units and a test.
