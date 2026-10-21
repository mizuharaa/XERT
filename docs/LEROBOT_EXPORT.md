# Lossless LeRobot export

The exporter writes the LeRobot v3 container using the official `LeRobotDataset`
API. Existing NPZ episodes and the shared simulation environment stay unchanged.
The legacy robot remains `xlerobot_left`, whose pinned asset declares SO-ARM100.
Six action dimensions do not establish SO-101 compatibility.

## Optional isolated environment

User approval covered installing this separate stack. The base
`requirements-lock.txt` is unchanged. Use CPython 3.13 on Windows AMD64 or Linux
x86_64; the platform locks pin the complete dependency resolution and CPU wheel
URLs for PyTorch, torchvision, and TorchCodec. They include the test runner.

```powershell
uv venv .venv-lerobot --python 3.13
uv pip sync requirements-lerobot-windows-lock.txt --python .venv-lerobot/Scripts/python.exe --only-binary :all:
.venv-lerobot/Scripts/python.exe -m pytest -q tests/test_lerobot_export.py
```

On Linux, use `requirements-lerobot-linux-lock.txt` and `.venv-lerobot/bin/python`.
The optional job must import `lerobot.datasets.lerobot_dataset.LeRobotDataset`
and assert installed LeRobot version `0.6.1` before testing. The base environment
skips this dependency-specific file; that skip does not prove a v3 round trip.
No native simulator extension, FFmpeg process, motor driver, or camera is used
by these state-only export tests.

The selected stack is LeRobot `0.6.1`, PyTorch `2.11.0+cpu`, torchvision
`0.26.0+cpu`, TorchCodec `0.11.0+cpu`, NumPy `2.2.6`, datasets `4.8.0`, Arrow
`21.0.0`, pandas `2.3.3`, and PyAV `15.1.0`. These are version pins, not results.
The shared Python 3.11 / PyTorch 2.14 / NumPy 2.4 environment conflicts with
[LeRobot's declared bounds](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml).
CPU package pairing follows [PyTorch's version table](https://pytorch.org/get-started/previous-versions/)
and [TorchCodec's compatibility table](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions).

## Run a fresh export

```powershell
.venv-lerobot/Scripts/python.exe -m xert.data.lerobot_export --source results/p1.2/recovery_integrated/dataset --collection results/p1.2/recovery_integrated/collection.json --output results/p2.1/legacy_v3 --task "Move the block to the tray"
```

Choose a new output path for each run. The CLI verifies all arrays through the
installed LeRobot loader, publishes only a complete dataset, and appends an
evidence row with artifact checksums. Existing destinations, even empty ones,
are preserved. Failed exports remove only their private staging directory.
Output inside the source store is rejected.

| Export feature | Source and units |
| --- | --- |
| `observation.state` | Complete pre-action observation, original order and names; legacy joint radians, rad/s, world metres, quaternion and time fraction |
| `action` | Verified training target: corrective sidecar when present, otherwise executed action; normalized absolute joint target in `[-1,1]` |
| `xert.executed_action` | Original executed action, same normalized units; replay stream |
| `xert.next_observation` | Complete successor observation, including the final terminal state |
| `next.reward`, `next.done` | Original transition reward and termination flag |
| `timestamp` | **Derived simulation time**, frame index divided by control rate; no capture time was measured |

The manifest retains original episode metadata, hashes, joint ranges, and
observation/action conventions. No terminal action is invented. LeRobot's
episode/frame indices describe the selected export order; the manifest maps
them back to immutable source IDs. The source task string is supplied explicitly.

`--robot-type so101` supports a separate six-position simulation profile only
when source provenance matches the pinned Menagerie SO-101 asset, ordered joint
names match, and the metadata declares radians and normalized target decoding.
The installed SO follower defines the `.pos` names in
[this source](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/so_follower.py).
Names do not change units: exported simulation targets are not calibrated motor
commands. Its regression fixture is synthetic schema data, not a demonstration.

## Optional official ACT reference (P1.4)

The reference runner adapts the full observation to
[ACT's input requirements](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/act/configuration_act.py):
the first twelve features are joint position/velocity, and the remaining fourteen
are environment state. It uses the official
[policy, loss, checkpoint format and temporal ensemble](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/act/modeling_act.py).
Normalization is explicit and verified against the original training-only
statistics; the automatically generated whole-export statistics are not used.
The official linear action head is retained. Validation scores its raw outputs;
execution clips to the existing `[-1,1]` limits after temporal ensembling. The
older model instead bounds predictions with tanh before ensembling. Runtime
provenance includes hashes of the installed official model and configuration
source, alongside package versions and the XERT runner hash.

`requirements-act-windows-lock.txt` and `requirements-act-linux-lock.txt` extend
the corresponding export lock with `mujoco==3.14.0` and `pybind11==3.1.0` plus
resolved dependencies. They were compiled with the same Python/platform flags
as the export locks, using `-r requirements-lerobot-<platform>-lock.txt` and those
two direct pins as input. CPU Torch/vision/codec wheel hashes remain unchanged.
Sync only the optional environment:

```powershell
uv pip sync requirements-act-windows-lock.txt --python .venv-lerobot/Scripts/python.exe --only-binary :all:
# In a VS 2022 developer shell, with dumpbin and lib on PATH:
cmake -S . -B build/lerobot -G "Visual Studio 17 2022" -A x64 -DPython_EXECUTABLE="$PWD/.venv-lerobot/Scripts/python.exe"
cmake --build build/lerobot --config Release --parallel 1
ctest --test-dir build/lerobot -C Release
.venv-lerobot/Scripts/python.exe -m pytest -q tests/test_lerobot_act.py
.venv-lerobot/Scripts/python.exe -m xert.eval.lerobot_act --output results/p1.4/official_act_v1
```

On Linux, sync the Linux ACT lock, use `.venv-lerobot/bin/python`, and configure
the separate build directory with `-DCMAKE_BUILD_TYPE=Release` without the
Windows generator/architecture flags. The Python ABI suffix lets the optional
native module coexist with the original interpreter's module. Do not run two
native builds in the same build directory. Neither build changes the C++ source.

The [predeclared protocol](../configs/lerobot_act_experiment.json) fixes the
budget before evaluation. The runner refuses changed export hashes, split
statistics, source contracts, or actual held-out reset states. It trains one seed
at a time on one CPU thread and writes a ledger-generated comparison with P1.3.
Use a fresh output directory for every attempt; interrupted artifacts are retained.
