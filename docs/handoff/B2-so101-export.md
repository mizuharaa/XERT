# Slice B2: SO-101 in the LeRobot exporter and the offline camera renderer

Branch `feat/b2-so101-export`, from `integration/sim-mvp` @ 89e53d4. Data and rendering tooling only: no CORE
logic, no ledger rows, no hardware, no camera device, no new dependency, no vendored file edited.

## What changed and why

- **Exporter schema** (`xert/data/lerobot_export.py:26` `_schema`).
  - Before: the `so101` profile accepted only the six-position fixture, so a real SO-101 bench episode was refused.
  - Now an SO-101 bench episode exports with the same layout as `xlerobot_left`. That layout is the 26-value
    `ManipulationEnv` observation: positions, velocities, block pose, grasp, tray and time fraction, all in
    `observation.state`. `render-cameras` needs these values to rebuild the scene.
  - The six-position fixture still exports with its `.pos` names.
  - The composed SO-101 bench records `SOURCE.json` plus its composition provenance, so its identity check compares
    only the vendored-manifest keys.
  - Nothing changes on the XLeRobot path.
- **Cameras per robot** (`xert/data/cameras.py`).
  - `CAMERAS` is keyed by bench robot (`:23`).
  - `CameraRenderer(size, robot)` takes its joints from `sim.ROBOTS`.
  - `render-cameras` reads the robot from the recording's `embodiment` or the export's `robot_type` (`:145`). It
    refuses a session that mixes robots (`:280`).
  - The per-frame reconstruction check and the streaming verify are shared by both robots and were not changed.
- **SO-101 camera scene** (`_so101_spec`, `cameras.py:56`).
  - `sim.so101_scene()` returns a compiled model, and its XML does not parse back ("empty class name").
  - `sim.py` was not edited, because its bytes are hashed into every recording's `implementation_sha256`.
  - So the composition is rebuilt as an `MjSpec`, and the rebuild is refused unless `spec.to_xml()` equals
    `so101_scene()`'s XML exactly.
  - The fingerprint is `sha256(xml + sorted provenance JSON)`. That is the same formula as `ManipulationEnv`, so it
    equals the `model_sha256` that every SO-101 recording stores.
- `--cameras` help text, `docs/CAMERA_KEYS.md`, `docs/learn/P2.1.md`, `docs/learn/P2.3.md`.
- `optional.yml`: `tests/test_so101_cameras.py` is added to the lerobot and teleop jobs.
  - `test_lerobot_export.py` was already in both jobs.
  - The lerobot job was **not run locally**, because there is no `.venv-lerobot` on this PC. Its new tests import
    `test_so101_bench` (teleop and calibration modules), which `test_teleop.py` in that job also uses.

## SO-101 camera definitions (NOT measured on the real SO-101)

| Feature | Camera | Body | Pose | fovy |
|---|---|---|---|---|
| `observation.images.scene` | added `scene`, **by eye** | world | pos (-0.02, -0.25, 0.40), target (0.30, 0.04, 0.15), up +z | 45 |
| `observation.images.wrist` | Menagerie `wrist_cam`, rendered as vendored | `camera_mount` (identity pose in `gripper`) | pos (-0.00296, 0.05436, -0.05310), quat (0.958280, -0.285792, -0.001556, 0.004547) | 61 |

- The wrist entry is `{"model_camera": "wrist_cam"}`: the renderer renders the vendored camera by name, nothing is
  re-derived. `test_so101_wrist_is_menageries_wrist_cam_and_the_fingerprint_is_the_recorded_one` reads pos, quat and
  fovy from `so101_xert.xml` and checks the compiled camera against them.
- The arm base is at (0.42, 0.04, 0.08) with yaw pi, at the far end of the table. So the scene camera sits past the
  near end, off to -y, and faces it. It was chosen by looking at three 192 px renders.
- **Caveats.**
  - Scene: placed by eye in simulation, not measured.
  - Wrist: Menagerie's wrist_cam, plausibly the 32×32 UVC wrist mount; mount pose and fovy 61 are unsourced and not
    measured on Daniel's arm; measure the real lens FOV before any sim-to-real claim.
  - `xert_cameras.json` records the wrist camera's compiled body, pos, quat and fovy, and a per-robot
    `camera_placement` with the caveats above. The XLeRobot manifest text is unchanged.
  - The SO-101 contact solver differs from XLeRobot's (slice B). Rendering does not depend on contacts.

## Measured results

- **SO-101 reconstruction.** The original scripted teleop recording has 5 frames, labelled `kind: scripted`. It was
  driven by a scripted hand source with `configs/teleop/so101_bench.json`, not by an expert.
  - `render-cameras` and `export-lerobot --cameras` measured the same max errors (tolerance 1e-5): grasp
    **1.39e-8 m**, tray **4.17e-9 m**, block position **0.0**, block orientation **0.0**.
  - The NPZ bench episode path (3 held steps, `--robot-type so101 --cameras`) is also within 1e-5.
  - With +1e-4 m injected into the rebuilt block position, the run is refused ("disagrees") and no output is
    written.
- **Round trip.** The exported dataset loads through `LeRobotDataset`.
  - Every original column is identical.
  - The only new features are `observation.images.{scene,wrist}`, with `dtype: image` (lossless PNG) and shape
    (3, 128, 128).
  - The streaming verify reports 5 frames, source columns identical and images lossless.
- **Path (b) import** of the same recording (column equality only, no render):
  - `environment_state` names equal the recording's `observation.environment_state` names.
  - The `state` names are equal too.
  - `model_sha256` equals `camera_model("so101")` fingerprint.
  - `frames.npz` `environment_state` has shape (5, 14).

## XLeRobot byte-identity evidence (measured; scratch scripts outside the repo)

- **Exporter.** I re-exported all 100 episodes behind the stored P2.1 artifact `results/p2.1/legacy_v3`, from
  `results/p1.2/recovery_integrated`, before and after the change.
  - 6 of 7 files are byte-identical to the stored artifact both times: data Parquet, info.json, tasks, episodes
    Parquet, roundtrip.json and xert_export.json.
  - `meta/stats.json` has equal values (canonical JSON identical), but its key order changes from run to run. Two
    runs of the same code already differ, so this is LeRobot's writer, not B2.
- **Camera parameters and pixels.** `CameraRenderer()` at 3 fixed states: `cam_pos`, `cam_quat`, `cam_fovy`, the
  body names, `model_sha256` and the sha256 of all 6 images were identical before and after in my runs.
  - **Image hashes are not a stable property.** Renders are not bit-exact across processes on this PC: QA saw ±1 on
    one pixel between two runs of the BASE code, and HEAD sits within that noise. Camera parameters are the stable
    evidence; `tests/test_cameras.py::test_xlerobot_camera_poses_stay_the_p2_3_poses` now pins them (QA's fovy
    mutant 45→50 / 75→60 fails it).
- **render-cameras output.** The same fixed XLeRobot scripted session (2 episodes, 3 and 2 frames) was rendered
  before and after.
  - 16 of 18 files are byte-identical, including `xert_cameras.json` with its source path removed.
  - In the 2 data Parquets, every column is equal and every PNG is byte-equal. Only the Hugging Face
    `fingerprint` in the Parquet schema metadata differs, and two runs of the base code differ in the same way.

## How to verify (from the checkout root, one process at a time)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
ctest --test-dir build -C Release --parallel 1
.\.venv\Scripts\python.exe scripts\equivalence_check.py
C:\Users\Khang\tmp\reach-public\build\venv-teleop\Scripts\python.exe -m pytest -q tests/test_cameras.py tests/test_lerobot_export.py tests/test_lerobot_io.py tests/test_so101_bench.py tests/test_recorder.py tests/test_so101_cameras.py
# One SO-101 recording end to end (CLI):
.\.venv\Scripts\python.exe -m xert render-cameras --dataset <so101 recording> --output <fresh>
.\.venv\Scripts\python.exe -m xert export-lerobot --episodes <so101 recording> --output <fresh> --task "<recorded task>" --cameras
```

Measured on this branch (before → after B2):

| Gate | Before (89e53d4) | After |
|---|---|---|
| base pytest | 169 passed, 37 skipped, 4 xfailed, 8 subtests | 171 passed, 42 skipped, 4 xfailed, 8 subtests |
| ctest (after `scripts\build_core.ps1`) | 2/2 | 2/2 |
| equivalence_check | 16 seeds, identical, max diff 0.0 | same |
| combined env, the 5 listed files (+ `test_so101_cameras.py` after) | 45 passed | 53 passed |

After merging `origin/integration/sim-mvp` @ ca9c90c (slice D; no C++ change, no rebuild): base pytest 190 passed,
44 skipped, 5 xfailed, 8 subtests; ctest 2/2; equivalence 16 seeds identical, max diff 0.0; combined env on
`test_cameras`, `test_so101_cameras`, `test_lerobot_export`, `test_lerobot_io`, `test_replay_real`: 52 passed,
1 xfailed.

Also run once in the combined env (no baseline, they do not import the changed modules): `test_gate_experiment`,
`test_teleop`, `test_lerobot_act`, `test_quality` gave 35 passed, 1 skipped.

## Interview questions

1. *How do you know the SO-101 images show the scene that was recorded?* Every frame's rebuilt grasp site, tray and
   block pose must match the recorded values within 1e-5, and the measured worst error was 1.4e-8 m.
2. *Why rebuild the SO-101 scene in cameras.py instead of calling `so101_scene()`?* That function only returns a
   compiled model and an XML that does not parse back, and editing sim.py would change every recording's
   implementation hash, so the rebuild is refused unless it writes exactly the same XML.
3. *How do you know XLeRobot output did not change?* A test pins the P2.3 camera poses, and the same inputs give
   identical export files and manifests; the only differences (stats.json key order, a random Parquet fingerprint,
   ±1 render noise on a pixel) also appear between two runs of the old code.

## TODO(daniel)

None. This slice has no CORE logic.

## Lines for integration to add (not edited here)

- MVP_PROGRESS / TASKS.md: "B2 SO-101 export + cameras: done on `feat/b2-so101-export`; SO-101 recording renders
  with reconstruction error ≤ 1.4e-8 m; XLeRobot camera poses pinned, export and manifests unchanged."
- README camera section: "`render-cameras` / `export-lerobot --cameras` serve both bench robots (`xlerobot_left`,
  `so101`); the SO-101 scene camera is placed by eye and the wrist is Menagerie's unmeasured wrist_cam."

## Open questions

- Wrist camera: Menagerie's wrist_cam, plausibly the 32×32 UVC wrist mount; mount pose and fovy 61 are unsourced
  and not measured on Daniel's arm; measure the real lens FOV before any sim-to-real claim.
- The SO-101 bench NPZ path is tested on 3 held steps only. Real SO-101 Store episodes arrive with the TODO(daniel)
  expert.
- Merged `origin/integration/sim-mvp` @ ca9c90c. Conflicts resolved: `optional.yml` lists both
  `test_so101_cameras.py` and `test_replay_real.py` in the lerobot and teleop jobs; `docs/CAMERA_KEYS.md` takes the
  owner's "Decided (2026-11-03)" bullet plus one B2 line on the SO-101 sim camera poses.
