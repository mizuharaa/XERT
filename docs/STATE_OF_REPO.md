# XERT repository state — P0.1

Audited locally on 2026-10-20, starting from commit `d20ef48` in
`C:\Users\Khang\tmp\reach-public`, on branch `docs/p0.1-state-of-repo`.
The original working tree was clean. The separate
`C:\Users\Khang\robotics\xlerobot-workbench` checkout contains uncommitted work;
it was not used as the source of this audit or edited.

This is a simulation-only manipulation workbench. The cross-embodiment system in
[the master prompt](../XERT_MASTER_PROMPT.md) is a plan, not implemented behavior.
An embodiment means a particular robot body and its joint arrangement.

## Corrections to master-prompt section 1

| Claim | Finding and source |
|---|---|
| Repository renamed from Reach to XERT | The local remote still points to `https://github.com/mizuharaa/reach.git`; the hosted rename was not independently checked. The README title was Reach and is now XERT. Package, C++ namespace, extension and executable names remain unchanged. |
| C++17 core, MuJoCo C API, pybind11 | Confirmed in `CMakeLists.txt`, `cpp/src/env.cpp`, `ik.cpp` and `bindings.cpp`. Python owns the MuJoCo model and state; the binding borrows them and keeps their owners alive. |
| Damped-least-squares IK, delay queue, physics substeps, contact-based success | Confirmed. Inverse kinematics (IK) calculates joint positions for a desired gripper pose. Damping stabilizes that calculation near configurations where some directions are difficult to move. The expert is a task-specific timed script, not a reusable retargeting solver. `Env::step` advances 20 physics steps per command and refreshes geometry before scoring. |
| Windows-only build | Confirmed. CMake rejects non-MSVC/non-64-bit builds, generates an import library from the MuJoCo wheel DLL, and uses Windows linker options. The PowerShell build script locates VS 2022 Build Tools. CMake requires at least 3.20 and the runtime asserts MuJoCo 3.14.0. No Linux build script or CI workflow exists. |
| Python CLI and FastAPI API | Confirmed in `cli.py`, `api.py`, `jobs.py` and `pipeline.py`. All named CLI commands exist. Benchmarking is a shared Python function and HTTP job endpoint; there is **no `benchmark` CLI subcommand**. The interface is FastAPI's interactive documentation and HTTP sessions, not a dedicated teleoperation UI. |
| XLeRobot left arm with SO-101 geometry | **Incorrect identification for this asset.** `assets/xlerobot/xlerobot.xml` declares `so_arm100`; the engineering notebook explicitly describes SO-ARM100-generation geometry. The verified description is the pinned XLeRobot left arm. SO-101 geometric equivalence or kit compatibility is not established. Do not silently relabel it when adding an embodiment spec. |
| Vendored model, license and checksums | Confirmed in `assets/xlerobot/SOURCE.json`, `LICENSE`, and `sim.verified_source`. Revision is `749abc837d5d771f26aeff961009c290e574b024`. The upstream XML contains the larger robot, but only the selected left-arm meshes are vendored; `model_xml()` extracts that arm into the custom bench. The full upstream robot is not the runtime scene. |
| NPZ episodes, state observations, no cameras | NPZ and state-only learning are confirmed. “No cameras” needs qualification: `ManipulationEnv.render()` and PNG API routes render an inspection view. Images are not episode observations and there is no webcam source or image-conditioned policy. |
| Behavior-cloning MLP and ACT-style CVAE transformer | Confirmed in `learning.py`. Behavior cloning learns demonstration actions from states; the MLP is a feed-forward neural network. The ACT-style model predicts action sequences using a conditional variational autoencoder (a training-time latent-variable model) and a transformer. It executes queued chunks without using new observations to revise the remaining chunk. No temporal ensembling, official LeRobot ACT, or vision encoder exists. |
| Scripted and learned-policy results | The historical JSON reports support the cited outcomes, but the paired friction/delay study and the separate scripted validation suite are **different experiments**, not a before/after improvement. The same starting seeds repeat across benchmark conditions. Learned-policy failure remains the recorded outcome; it was not remeasured as a new performance study here. See the evidence boundaries below. |
| No safety layer or data-quality gate | Correct for the proposed general-purpose layers, but too broad without qualification. Action validation, actuator bounds, expert joint/update clipping, checksums, bounded archive parsing, replay verification and training-data checks already exist. There is no shared position/velocity/acceleration safety controller, stale-input watchdog, emergency stop, or teleoperation quality-threshold experiment. |
| No retargeting, multiple embodiments, hardware backend, LeRobot export | Confirmed by the tracked implementation and dependencies. Retargeting means mapping human motion to robot commands; no human-motion source exists. The C++ core hardcodes bench joints, sites and task logic. There is no `xert/` package, embodiment loader, hardware transport, LeRobot exporter or results ledger. |

## Existing flow and contracts

`CLI / HTTP session or job -> pipeline -> Python scene/state wrapper -> C++ expert
or learned policy -> C++ step -> recorded arrays -> NPZ and JSON metadata`.
Training reads these episodes, saves a checkpoint, and evaluation runs the learned
controller or scripted expert in the same simulator. Replay resets the simulator
and reissues recorded actions; it checks the resulting observations, rewards and
termination flags rather than just displaying saved poses.

The following numbers are **source-defined interface constants, not measured
performance** (`sim.metadata`, `Env::observation`, `Env::step`):

- Actions are 6 absolute joint-position targets normalized to `[-1, 1]`, ordered
  `Rotation_L, Pitch_L, Elbow_L, Wrist_Pitch_L, Wrist_Roll_L, Jaw_L`.
- Observations have 26 float32 values: joint positions and velocities, block
  position and orientation, gripper and tray positions, and elapsed episode fraction.
  Positions use world-frame metres, joint angles use radians, and orientation is
  a `wxyz` quaternion. The world frame is the simulator's fixed coordinate system.
- Physics timestep is 0.002 seconds; control advances 20 substeps, giving 25 Hz of
  **simulated** control time. This is not measured wall-clock latency.
- Episodes hold `T` actions and `T+1` observations, plus rewards and terminal flags.
  There are no per-frame capture timestamps, confidence values, or clamp-event arrays.
- Checkpoint compatibility includes the observation/action ordering, horizon,
  model identity, simulator source identity, and MuJoCo version. The horizon is
  the maximum episode length and also scales the time observation.

Success requires a prior lift and then a sufficiently still block inside the tray
without robot contact. The block moves through simulated contact, not attachment
or teleportation. The runtime lift flag checks height; regression tests separately
verify contact during the expert's lift. This is task scoring, not a general
safety assessment. Friction changes apply to every scene geometry.

## What is tested

| Area | Checks actually present |
|---|---|
| Simulator | `tests/test_sim.py`: seeded reset/replay, delayed targets, contact lift/release, current transforms after stepping, invalid inputs, asset integrity and portable model identity. |
| Native ownership | `tests/test_core_binding.py`: direct extension import, shape/range validation, state mirroring, matching model/data owners, and owner lifetime after deleting the Python wrapper. |
| C++/Python equivalence | `tests/test_equivalence.py`: exact observation and outcome agreement against `tests/reference_sim.py` for default, delayed and low-friction configurations. `scripts/equivalence_check.py` additionally runs its full default seed range. |
| CTest | `export_bench` generates the XML/position fixture; `reach_core` checks reset layout, invalid actions, command delay, float32 conversion, contact lift/release and terminal guards. These are a fixture check and a native test executable, not separate policy evaluations. |
| Learning | `tests/test_learning.py`: split isolation, train-only normalization, padding masks, prior-only validation, synthetic training/reload, chunk execution/reset, corruption, cancellation and finite-value checks. Passing does not establish learned manipulation success. |
| Pipeline/API/storage | `tests/test_backend.py` and `test_audit.py`: NPZ bounds/integrity, replay, short training/evaluation wiring, model contracts, overlapping states/seeds, imported-transition verification, sessions, request validation and job submission. |
| Jobs | `tests/test_jobs_audit.py`: failed writes/submissions, slot release, error preservation, reader/writer locking and Windows file-sharing behavior. |

No tests exercise the proposed XERT interfaces, human motion, multiple bodies,
LeRobot conversion, the quality-gate experiment or hardware; those systems do not
exist yet. The separate live HTTP/render probe is `scripts/audit_http_check.py`;
its historical JSON report is not a fresh live-server test in this audit.

## Evidence boundaries and README drift

The README's implementation description is broadly accurate. Its result claims
point to historical simulation reports, not the proposed append-only ledger.
Keep these reports as historical evidence; do not promote them to fresh XERT
resume metrics. Ledger-backed current performance: **TBD (not measured)**.
P0.2 owns measurement recording; P0.1 adds no benchmark numbers or ledger code.

- `docs/REFERENCE_EXPERIMENT.json`: original paired experiment. It records scripted
  success and unsuccessful learned controllers, with repeated seeds across conditions.
- `docs/audit/REFERENCE_POST_FIX.json`: later paired experiment after the Python
  simulator timing correction; the learned controllers still fail. This predates
  the C++ source-identity contract in the audited checkout.
- `docs/audit/SIMULATION_POST_FIX.json`: separate scripted validation with contact
  observations. `SIMULATION_D8_BEFORE_RENDER_COMMENT.json` is an earlier source
  snapshot; neither should be pooled with the paired study.
- `docs/SIMULATION_CHECK.json` and the simulation PNGs are earlier simulation
  evidence, not hardware records or fresh tests of learned controllers.
- `docs/audit/NORMALIZATION_EXPERIMENT_MLP.json` records a controlled preprocessing
  comparison that did not recover the task. `NORMALIZATION_EXPERIMENT.json` is an
  interrupted attempt with no completed variants. The canonical learner remains unchanged.
- `docs/audit/EXPERT_OBSERVABILITY.json` records that command history can change
  the expert action without changing the policy observation. It is a useful
  diagnostic lead, not proof of the complete cause of learned-policy failure.
- The clean checkout does not contain the referenced historical datasets or model
  payloads. JSON IDs and hashes are retained, but they do not make those payloads
  available here or prove that an old checkpoint loads under today's contract.
- Older teaching docs still describe physics/IK bodies in `sim.py`; those now
  delegate to C++. `BUILD_IT_YOURSELF.md` also says no corrective experiment has
  run, while the normalization experiment files record one. `SIMULATION.md`'s
  worker-thread rendering account is historical; the current API deliberately
  renders on the event-loop/main thread. These documents need a later scoped update.

There is no `benchmark` CLI command for P0.2 to patch: integrate the existing
`pipeline.benchmark`/HTTP route or explicitly add a command in that slice.
The P1.1 single-episode overfit diagnostic also needs an intentional isolated path:
current training rejects fewer than two episodes to preserve a validation split.
Do not weaken that production check incidentally during this documentation slice.

## Local validation and checkout caveat

The checkout initially had neither `.venv` nor `build/`. For validation only,
`.venv` is a Windows directory junction to the existing workbench environment;
no dependencies were installed. The unchanged Windows script built this checkout's
native extension and executables. Python imports were checked against this checkout.
This is local Windows verification, not a clean dependency-install or Linux check.

The first baseline attempts failed because Git's Windows line-ending conversion
changed the bytes of `assets/xlerobot/LICENSE` and `xlerobot.xml`. Their LF forms
match the already-pinned SHA-256 values. Restoring those exact bytes made the
checksum guard pass and produced no tracked asset diff. `.gitattributes` currently
pins Python/C++ line endings only: a future checkout can recreate this failure.
The durable Git-attributes fix is deferred so P0.1 changes no build configuration.
No checksum was weakened or regenerated.

| Command from repository root | Before documentation edits, after byte restoration | After documentation edits |
|---|---|---|
| `.\.venv\Scripts\python.exe -m pytest -q` | PASS | PASS |
| `ctest --test-dir build -C Release` | PASS | PASS |
| `.\.venv\Scripts\python.exe scripts/equivalence_check.py` | PASS | PASS |

Raw command output is retained locally under ignored `build/p0.1/`, including
the initial failures and the successful reruns. These files are not committed or
ledger rows. Pytest reports a Starlette/httpx deprecation warning; it does not fail
the suite. No fresh learned-policy benchmark, live HTTP/render run, Linux test,
remote-rename verification or hardware check was performed.

## Review inventory and change boundary

The audit covered the tracked C++ headers/sources/test, Python package and tests,
scripts, build/package/dependency/config files, README and teaching docs, historical
JSON reports, upstream model/license/manifest, mesh integrity and simulation PNGs.
Binary meshes were checked against the manifest and loaded through the simulator;
this does not validate their dimensions against a physical robot.

P0.1 preserves the supplied master prompt, changes only the README title, and adds
this audit plus `docs/learn/P0.1.md`. There are no runtime, schema, dependency,
test or hardware changes. Stop here; the next slice is P0.2.
