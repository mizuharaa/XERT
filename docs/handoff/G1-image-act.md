# Slice G1: scaffolding to train and evaluate image-conditioned official LeRobot ACT

Branch `feat/g1-image-act`, from `integration/sim-mvp` @ c7e2ca1. Plumbing, tests and docs only. **No trained
policy and no success number exist.** Every training choice is a `TODO(daniel)` stub. No ledger rows, no hardware,
no camera device, no new dependency.

## What changed and why

- **`xert/eval/image_act.py`**, CLI `python -m xert image-act train | evaluate`.
  - **Input.** A `render-cameras` output: one LeRobot v3 dataset, or a session directory of `episode_*` datasets,
    on either bench robot.
    - `describe()` refuses a dataset that lacks `observation.images.scene` or `observation.images.wrist`.
    - It also refuses a dataset without `xert_cameras.json`, or one rendered with other camera definitions, sizes
      or models than `CAMERAS`. Evaluation renders with `CAMERAS`, so the training images must match it.
    - It reads the source kind and bench seeds per dataset. For recordings it also checks that the per-frame
      `xert.source` values agree with the manifest kind (the P5.1 check).
  - **Features.** `policy_config()` takes the inputs from `dataset_to_policy_features` (the smoke-test path):
    images, `observation.state` and `observation.environment_state`. The output is `action`. Nothing is hardcoded.
  - **Train.** The plumbing:
    - checks the settings, the split and the dataset;
    - computes normalization statistics (mean/std/min/max; images per channel) from the TRAIN episodes only, before
      any augmentation;
    - loads action chunks with `delta_timestamps` (padded at episode ends, `action_is_pad`);
    - seeds torch and the shuffle generator;
    - runs the epochs, with prior-inference validation L1 per epoch as in P1.4;
    - saves every epoch, then keeps the one `select_epoch` chooses;
    - writes the official policy files, the official pre/post-processors (which hold the train-only statistics) and
      `xert_image_act.json`. That file holds robot, dataset sha256, split, source kinds, training bench seeds, camera
      definitions, feature specs, settings, history, selected epoch, git commit and library versions.
    - Until the stubs exist, `train` stops with `image-ACT settings not chosen (TODO(daniel))` before it writes
      anything.
  - **Evaluate.** Loads the checkpoint, its processors and the bench for the robot named in its metadata. Then:
    - It refuses evaluation seeds that overlap the training bench seeds.
    - It refuses if `CAMERAS` changed since training.
    - Each held-out seed gets a fresh `BenchBackend` and one `CameraRenderer` (closed after the rollout).
    - Success is the native core's contact rule. Commands get a joint-limit clip only, as in P5.3.
    - It always writes a report JSON: trials, successes, Wilson 95% interval, per-tick timing.
    - **Ledger rows** (`slice G1`, `metric success_rate`, experiment `image_act_v1` by default) are written only
      when every training source kind is human (`webcam`/`recorded_human`, the P5.3 rule) and there are at least
      30 seeds. Otherwise the report says why there is no row.
- **Live render at evaluation.** Each tick, `live_frame()` takes the live joint positions and the block pose from
  the native observation, by name. `CameraRenderer.render()` uses them, the same code and camera definitions as
  `render-cameras`. Images go to the policy as float (3, H, W) in [0, 1], the same form LeRobotDataset returns the
  stored PNGs in. State vectors are built by feature name (`gate_experiment.bench_observation`).
- **Timing contract.** Rollouts run in simulated time with no wall-clock deadline. Rendering and inference happen
  between two bench steps, and every step advances physics by exactly 1/25 s. So the 25 Hz contract holds in sim
  time whatever rendering costs. This says nothing about real-time feasibility on hardware.
- **Actions.** Recording actions (`<joint>.pos`, radians) are sent as they are. Export actions (joint names,
  normalized to [-1, 1]) are mapped back through the actuator ranges. Any other action naming is refused.
- `tests/test_image_act.py`; `optional.yml` runs it in the lerobot and teleop jobs.

## How to verify (one process at a time)

```powershell
# 1. Render cameras onto a bench recording, session or export (use --size 64 for a quick look)
.\.venv-lerobot\Scripts\python.exe -m xert render-cameras --dataset <recording|session|export> --output <rendered>
# 2. Train (refuses until the TODO(daniel) stubs are written)
.\.venv-lerobot\Scripts\python.exe -m xert image-act train --dataset <rendered> --output <checkpoint> --seed 0
# 3. Evaluate on held-out bench seeds (rows only for human data and >= 30 seeds)
.\.venv-lerobot\Scripts\python.exe -m xert image-act evaluate --checkpoint <checkpoint> --seeds 10000 10001 ... --output <report.json>
# Tests
.\.venv\Scripts\python.exe -m pytest -q
.\.venv-lerobot\Scripts\python.exe -m pytest -q tests/test_cameras.py tests/test_lerobot_act.py tests/test_image_act.py
```

Tests use a 2-episode, 5-frame scripted recording rendered at 64 px, on CPU. A fixture settings dict and a
fixture update step (the smoke-test shape) are injected with monkeypatch.

| Gate | Before (c7e2ca1) | After |
|---|---|---|
| base pytest | 192 passed, 49 skipped, 5 xfailed, 8 subtests | 192 passed, 50 skipped, 5 xfailed, 8 subtests |
| `.venv-lerobot`: test_cameras + test_lerobot_act (+ test_image_act) | 12 passed | 18 passed, 2 xfailed |
| combined `venv-teleop`, same list | 12 passed | 18 passed, 2 xfailed |

## TODO(daniel)

All four are in `xert/eval/image_act.py`. Their contracts are in the docstrings.

| Stub | Line | Owns |
|---|---|---|
| `image_act_settings(dataset_meta)` | :37 | ACTConfig fields (chunk size, n_action_steps, backbone + pretrained weights, dims, layers, latent, dropout, KL weight, temporal ensembling, lr, normalization modes, device), epochs, batch size, augmentation (`image_transforms`), validation episodes |
| `make_optimizer(policy)` | :56 | optimizer and learning rates |
| `training_step(policy, batch, optimizer)` | :64 | loss, backward, clipping, step. It takes the optimizer because the plumbing never steps it. |
| `select_epoch(history)` | :76 | checkpoint selection and early stopping |

**Done** = both strict xfails in `tests/test_image_act.py` pass with their markers removed:
- `:148` `test_daniel_settings_build_a_valid_image_act_config`: the settings build a valid ACTConfig with both
  image keys, and they are JSON-serializable.
- `:158` `test_daniel_training_gives_a_checkpoint_whose_evaluation_writes_rows`: `train` with your stubs gives a
  checkpoint, and `evaluate` on 30 held-out seeds writes one ledger row (temporary ledger; the scripted fixture
  is treated as human only for this check). This is a harness check, not a performance bar. Its rollouts take
  30 × 600 ticks at your model's inference cost.

**Performance targets are yours to pre-register** (commit them before the first evaluate, as in P5.3). The code
has no success threshold.

## Interview questions

1. *Why normalize on the train split only?* Validation statistics would leak held-out data into the inputs the
   model trains on, so the validation error that picks the checkpoint would no longer be honest.
2. *Why render live at evaluation instead of using recorded images?* A closed-loop rollout visits states no
   recording contains, so the policy must see images of where the arm actually is now, rendered by the same
   cameras it was trained on.
3. *Why no ledger row for scripted data?* The ledger holds portfolio evidence about learning from demonstrations,
   and scripted teleop is a software fixture, so its numbers go to a report file only (the P5.3 rule).

## What is NOT shown

- No trained image policy, no success rate, no comparison with the P1.4 state-only rows.
- No real camera data: images are MuJoCo renders with by-eye camera poses (see `docs/CAMERA_KEYS.md`).
- No real-time claim: rollouts run in simulated time.
- Live renders use float64 state while the training renders used the stored float32 state, so a few pixels may
  round differently. B2 also saw ±1 pixel noise between processes.

## Lines for integration to add (not edited here)

- TASKS.md table: `| G1 | image-act train/evaluate scaffolding: official ACT on rendered scene + wrist images | feat/g1-image-act | Scaffolding; training is TODO(daniel) | [G1-image-act.md](G1-image-act.md) |`
- TASKS.md queue: "**G1, image-ACT training choices**: `image_act_settings` :37, `make_optimizer` :56,
  `training_step` :64, `select_epoch` :76 in `xert/eval/image_act.py`. Done = the two strict xfails in
  `tests/test_image_act.py` (:148, :158) pass unmarked; pre-register the performance target first."
- README / MVP_PROGRESS: "`python -m xert image-act train|evaluate`: official LeRobot ACT on rendered
  scene + wrist images, live renders at evaluation, ledger rows only for human data on ≥ 30 held-out seeds.
  No trained policy yet."
- `docs/CAMERA_KEYS.md` table: "Image ACT (G1) | `xert/eval/image_act.py` `policy_config` | every VISUAL key from
  `dataset_to_policy_features`; both `scene` and `wrist` required."
