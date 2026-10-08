# Reach: robot manipulation & learning

XLeRobot arm manipulation workbench on MuJoCo + PyTorch. A C++17 port of the simulation core is in progress.

A local reference implementation for learning the full robotics-data loop:

**MuJoCo contact simulation → demonstration episodes → imitation learning → held-out evaluation → failure replay.**

The task uses the pinned upstream XLeRobot left-arm kinematic chain on a stationary benchmark bench. The block must be lifted, released into the tray, and remain settled without robot contact. This is not a calibrated digital twin of your WOWROBO kit. There is no hardware command transport, cloud execution, or industrial safety certification.

The observations are simulator state. The models are a behavior-cloning MLP and a compact **state-based ACT-style CVAE Transformer**, not camera-based ACT or a pretrained VLA. See [the engineering notebook](docs/ENGINEERING_NOTEBOOK.md) and [rebuild-it-yourself guide](docs/BUILD_IT_YOURSELF.md) before changing them.

**Measured status after QA:** 27 tests and 8 subtests pass. The corrected simulator completed 16/16 scripted validation trials with finger contact during lift and release before success. An independent audit and verified fixes covered simulation timing, replay, data validation and job lifecycle. The [fresh reference experiment](docs/audit/REFERENCE_POST_FIX.json) records new demonstrations, both trained models and paired evaluations under the corrected simulator. The [original experiment](docs/REFERENCE_EXPERIMENT.json) is historical evidence: scripted 9/9, each learned policy 0/9. Learned manipulation remains an open research task; passing software checks is not physical deployment or broad robustness.

## Run the installed workspace

The fresh post-fix paired evaluation also scored **scripted 9/9, MLP 0/9, ACT-style 0/9**. Both learning pipelines ran successfully; neither learned controller completed the task in these trials. Their failure traces are preserved for the next manual debugging exercise.

```powershell
Set-Location reach
.\.venv\Scripts\python.exe -m xle_workbench.cli serve
```

Open **http://127.0.0.1:8765/docs** for the interactive API. This is the current operator interface. Bind to localhost; the service has no remote authentication or multi-user scheduler. Closing it interrupts active jobs visibly and discards unsaved live sessions.

## Complete pipeline from the CLI

Commands below write JSON themselves; no PowerShell redirection/encoding tricks are needed.

```powershell
.\.venv\Scripts\python.exe -m xle_workbench.cli --output data/collection.json collect --episodes 10 --seed 0
.\.venv\Scripts\python.exe -m xle_workbench.cli --output data/baseline.json train --collection data/collection.json --config '@configs/baseline.json'
.\.venv\Scripts\python.exe -m xle_workbench.cli --output data/act.json train --collection data/collection.json --config '@configs/act.json'
```

The output includes each model's `id`. Evaluate one on new seeds:

```powershell
.\.venv\Scripts\python.exe -m xle_workbench.cli evaluate --model MODEL_ID --episodes 10 --seed 10000
.\.venv\Scripts\python.exe -m xle_workbench.cli evaluate --episodes 10 --seed 10000
.\.venv\Scripts\python.exe -m xle_workbench.cli replay EPISODE_ID --image data/replay.png
```

For a short integration run rather than a performance experiment:

```powershell
.\.venv\Scripts\python.exe -m xle_workbench.cli --data data/smoke smoke --epochs 2 --model-type act
```

Smoke runs collect four episodes, train, evaluate scripted and learned policies on the same two new seeds, and verify recorded-action replay. A successful pipeline does **not** imply the briefly trained policy succeeds at manipulation. Inspect success/failure counts separately.

## API flow

| Endpoint | What it does |
|---|---|
| `GET /environment` | Robot schema, joint order, units, action conventions and default configuration |
| `POST /sessions` | Create a seeded live simulation |
| `POST /sessions/{id}/step` | Apply normalized joint targets or request an expert step; record transitions |
| `GET /sessions/{id}/frame.png` | Render the current real simulation state |
| `POST /sessions/{id}/save` | Save the current manual/assisted episode, including partial/failed attempts |
| `DELETE /sessions/{id}` | Close a session |
| `POST /jobs/collect` | Collect scripted demonstrations for explicit seeds |
| `POST /jobs/train` | Train selected episode IDs with an episode-level validation split |
| `POST /jobs/evaluate` | Evaluate one model or the scripted baseline on new seeds |
| `POST /jobs/benchmark` | Compare models and baseline on identical seeds across friction/delay conditions |
| `GET /jobs/{id}` | Read durable progress, completion or failure |
| `POST /jobs/{id}/cancel` | Request cooperative cancellation |
| `GET /episodes/{id}/arrays` | Inspect all state/action transitions |
| `GET /episodes/{id}/replay` | Re-run recorded actions and report maximum state discrepancy |
| `GET /episodes/{id}/frame.png` | Render the replay's final state |
| `GET /episodes/{id}/download` | Download the checked NPZ payload |
| `POST /episodes/import` | Import raw NPZ only after seed/config resimulation verifies its transitions |
| `GET /models`, `GET /evaluations` | Browse checkpoint provenance and measured reports |

Session actions are absolute normalized joint targets, not Cartesian deltas. `GET /environment` supplies the ordered joint names. `expert=true` selects the scripted controller; do not claim those are human demonstrations. Simulation import requires the original seed and configuration and is not a LeRobot dataset or hardware-log importer.

Example benchmark body for `/jobs/benchmark`:

```json
{
  "seeds": [10000, 10001, 10002],
  "model_ids": ["REPLACE_WITH_MODEL_ID"],
  "conditions": [
    {"friction": 1.0, "delay_steps": 0},
    {"friction": 0.6, "delay_steps": 0},
    {"friction": 1.0, "delay_steps": 2}
  ]
}
```

All attempts are retained, including failures. Evaluation seeds cannot overlap the source demonstration seeds. Changing the horizon changes the time observation and invalidates a checkpoint's schema contract. Reports include seeds, exact configuration, outcome reasons, trace IDs and a success-rate interval; small trial counts remain small evidence.

## Reproduce the environment

Python 3.11 is used here. Install with `uv venv --python 3.11 .venv`, then `uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt` and `uv pip install --python .venv/Scripts/python.exe --no-deps -e .`. Upstream assets are vendored with their license, source revision and checksums under `assets/xlerobot`; see [simulation decisions](docs/SIMULATION.md).

The initial environment uses CPU PyTorch even though this laptop has an NVIDIA GPU. `device: cuda` is accepted only if the installed PyTorch actually supports CUDA; it never silently falls back. No GPU hours or external compute have been purchased.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Artifacts live under `data/episodes`, `data/models`, `data/evaluations`, and `data/jobs`. Checkpoints bind their data, normalization metadata, model weights and simulation contract. Hashes detect accidental changes; they are not digital signatures or proof that untrusted artifacts are authentic. Hardware support should begin with measured kit geometry, joint signs/limits and timing, followed by supervised low-speed trials—not by forwarding simulation actions to motors.
