# XLeRobot workbench: engineering notebook

This is an AI-assisted reference implementation intended to be understood and rebuilt by hand. The goal is a small, inspectable simulation-to-dataset-to-policy-to-evaluation loop. It is not evidence of physical robot deployment. The owner is assembling a WOWROBO XLeRobot kit; hardware integration is a later, separate milestone.

## Build record

| Area | State | Evidence |
| --- | --- | --- |
| Simulator, task and observation/action contract | Local simulation checks passed | `sim.py`; four simulator tests and `SIMULATION_CHECK.json` |
| Episode storage and pipeline | End-to-end ACT smoke completed | `data/act-smoke/smoke-report.json`; NPZ payloads and JSON manifests |
| Local job runner and HTTP interface | Backend integration checks passed | `tests/test_backend.py`; sessions, bounded jobs, import/replay |
| State-based imitation learning | MLP and ACT-style models trained and evaluated | `REFERENCE_EXPERIMENT.json`; 80-epoch MLP and 40-epoch ACT-style runs |
| Evaluation and failure analysis | Paired reference experiment and exact action replay completed | `REFERENCE_EXPERIMENT.json`; scripted controller 9/9, each learned controller 0/9 in this small study |
| Physical robot control | Planned | No hardware execution authorized or claimed |
| Vision-based ACT, VLA, reinforcement learning | Outside this first version | No trained-policy claim |

Update a row only after inspecting its code and recording the command or artifact that proves it. A passing unit test proves the tested contract, not general task success or transfer to hardware.

## Why this project

The useful experiment is whether a learned controller can complete a manipulation task at object positions it did not train on, and where it fails. A dashboard supports that experiment; it is not the experiment. A fixed-base, single-arm task keeps observation, action, contact and timing errors understandable before introducing navigation or two-arm coordination.

## Decisions and their engineering consequences

- **Simulation first.** MuJoCo supplies physics and contact integration. A simulated grasp is meaningful only if movement comes from actuator/contact dynamics, rather than teleporting or attaching the object to the hand. Neither path certifies hardware safety.
- **Explicit contracts before models.** Record ordered observation fields, units, coordinate frames, action limits and sample timing. A policy is compatible only when those match its training checkpoint.
- **State-based learning first.** Simulator state makes it possible to debug control and dataset alignment before camera calibration and visual representation learning. State observations can contain privileged object information unavailable on the real robot. A state action-chunking transformer is not the full published vision-based ACT method.
- **Episode splits, not frame splits.** Adjacent samples from one rollout are strongly correlated. Split whole episodes by seed before fitting normalization. A held-out episode seed is not automatically an unseen spatial region; record which distribution changed.
- **Persist evidence.** An episode needs the task/configuration, seed, model identity, controller identity, timestamps, observations, commanded actions and outcome. A checkpoint needs its training episode identifiers and normalization. An evaluation needs the checkpoint and scenario identifiers.
- **One readable backend.** Local artifacts and a small job runner are sufficient for a local workbench. Distributed workers, cloud infrastructure and arbitrary code execution do not make the robotics experiment stronger.

## Contracts to trace in the code

Current source map (implementation inspected; runtime gates recorded separately below):

| File | Responsibility |
| --- | --- |
| `xle_workbench/sim.py` | Derive a stationary left-arm scene from the upstream model; step physics, form observations, apply targets and score the task |
| `xle_workbench/storage.py` | Validate NPZ episodes, assign internal IDs, publish JSON manifests, check payload hashes |
| `xle_workbench/learning.py` | Episode split, observation normalization, MLP and state CVAE/transformer training, checkpoint loading |
| `xle_workbench/pipeline.py` | Shared collect/train/evaluate/replay functions; compatibility checks and provenance |
| `xle_workbench/jobs.py` | One local worker, a bounded submission queue, durable states and cooperative cancellation |
| `xle_workbench/api.py` | Local HTTP sessions, collection/training/evaluation jobs, artifact import/export and replay |
| `xle_workbench/cli.py` | Command-line access to the same pipeline and an end-to-end smoke command |
| `assets/xlerobot/SOURCE.json` | Upstream repository/revision, license and downloaded asset hashes |

The initial scene derives its SO-ARM100-generation left-arm geometry from XLeRobot revision `749abc837d5d771f26aeff961009c290e574b024`; it is a custom stationary benchmark, not the complete mobile robot model or an exact WOWROBO kit digital twin. The upstream source and Apache-2.0 license are preserved with the assets. Compare the assembled kit's mechanical generation and calibration before assuming model compatibility.

The benchmark replaces the original actuator setup with position servos, chooses task-specific damping and force limits, and adds a table, block and tray. These are explicit simulation choices. Servo gains, collision meshes, contact friction and force limits have not been fitted to measurements from the owner's kit. A controller that succeeds here still needs mechanical-model validation and calibration before any transfer claim.

Model initialization verifies the downloaded asset hashes against `SOURCE.json` and checks that the selected meshes are covered. The model identity combines XML with a normalized mesh-directory path and the source manifest, so copying the same assets to another project directory does not itself change the identity. The manifest is a reproducibility record, not an authenticated signature from a hardware manufacturer.

Artifacts also record the simulator source-file hash and MuJoCo version. This is deliberately conservative: even a comment-only change to the simulator invalidates the recorded code identity. It avoids treating changed task or observation logic as equivalent merely because the robot XML stayed the same.

The current simulator contract is:

- **Joints/actions:** `Rotation_L, Pitch_L, Elbow_L, Wrist_Pitch_L, Wrist_Roll_L, Jaw_L`, in that order. Six absolute position targets are normalized to [-1, 1] using each actuator's configured control range. They are not joint velocities or Cartesian displacements.
- **Observation:** six joint positions, six joint velocities, block XYZ and wxyz quaternion, grasp-site XYZ, tray XYZ and episode time fraction: 26 values. Positions are world-frame metres; revolute positions/velocities are radians/radians per second. Object pose is privileged simulator state.
- **Clock:** a 0.002-second physics timestep and 20 physics steps per control action give 25 Hz control. Action-command delay is an explicit FIFO in control steps; it does not delay the observation.
- **Transition alignment:** actions have shape `(T, 6)` and observations `(T+1, 26)`. Row `observations[t]` is the state from which `actions[t]` is selected. The final observation has no training action label.
- **Task:** a free block must first lift, then settle inside the tray without robot contact. The predicate checks height, horizontal bounds, translational speed and consecutive settled samples; it is not a general manipulation or safety certificate.

The scripted demonstrator is a timed approach, descend, close, lift, transfer, release and retract sequence. It uses a damped least-squares site Jacobian to obtain joint targets and limits target changes before issuing commands. It does not attach the block to the gripper. Contact dynamics determine object movement. Its label `expert` is a controller identifier, not a guarantee that every collected trajectory succeeds; collection preserves each outcome instead of silently discarding failures.

An early seed-0 grasp pushed the block sideways during closure and lifted empty, producing `timeout_no_lift`. The benchmark was adjusted in several places: mount/grasp geometry, block dimensions and jaw servo settings. Because these changes were combined, the later grasp result does not isolate a single causal fix. See [SIMULATION.md](SIMULATION.md) for the simulator's tuning record, measured scenarios and assumptions. For a manual rebuild, change one parameter at a time and retain the before/after traces.

### Debugging record: the action saved must be the action applied

The integrated replay check exposed a precision mismatch: the expert produced float64 commands, the first simulation applied those commands, but the episode archive stored float32 values. Replay applied the rounded values. Tiny command differences can change a contact trajectory, so the resulting replay diverged even with the same seed and model.

The fix canonicalizes commands to float32 before both application and storage, including the simulator's public step boundary. This is a data-contract correction rather than a looser replay tolerance. The regression check compares a run using the original command with a run using its recorded float32 command and verifies every observation. Reproducibility still means the tested software/model configuration; it is not a promise across MuJoCo versions or hardware architectures.

### Debugging record: validate array declarations before allocation

An NPZ is a ZIP archive containing NPY arrays. Limiting the ZIP's total expanded bytes does not establish that an inner NPY header's claimed shape is reasonable: a small payload can declare a huge array. The importer therefore checks the inner header, numeric dtype, rank, bounded dimensions and exact expected payload byte count before constructing an array. It accepts only the four expected members and never enables pickle. This extra parsing protects the file-input boundary; it is not a general-purpose NPY implementation.

The action bound check also uses direct comparisons with the lower and upper limits. `abs(minimum_signed_integer)` can overflow in its original integer dtype, so checking only `max(abs(actions))` can miss an invalid value. Values are checked again after conversion to float32 because a finite float64 value can become infinity at the storage precision. These cases belong in regression tests even though normal simulator output never produces them.

`learning.py` normalizes observations with training-episode statistics, floored at a small positive scale. Actions already follow the bounded command contract and are not standardized with a learned action mean/scale. The MLP predicts one action; the state CVAE/transformer predicts an action chunk. Training includes a masked reconstruction objective and (for the chunker) KL regularization. Validation and deployment inference use the prior latent zero, so future validation actions are not fed into the encoder. Inference executes a complete chunk before requesting another one; there is no temporal ensemble.

For the learning equations, checkpoint format and further implementation rationale, see [LEARNING.md](LEARNING.md).

Checkpoints record episode hashes, split indices, observation normalization, dimensions/configuration, validation history and a weight-file hash. A validation MAE is not a task success rate. The pipeline binds hashes for both the weight file and training metadata, and the loader uses weights-only deserialization.

The pipeline rejects reused demonstration seeds for learned-policy evaluation, and rejects mixed episode contracts or shared simulation seeds before training. It checks each stored contract against the episode's simulation metadata. The contract includes the action convention and horizon, because changing the horizon changes the meaning of the `time_fraction` observation. Friction, command delay and object placement remain controllable evaluation variations. Replay means resimulating recorded actions from the saved seed/configuration and comparing every observation, not just showing stored states. An episode archive's SHA-256 protects its content against unnoticed alteration; it does not prove the episode came from hardware or that its metadata is truthful.

The job runner admits at most four unfinished jobs and executes one at a time. It publishes state to disk, marks formerly queued/running jobs interrupted after restart, and checks cancellation between simulation steps or training batches. It does not provide distributed execution, process isolation or resumable optimizer training.

The HTTP layer also supports up to four live simulation sessions. A session accepts explicit normalized joint targets or scripted assistance, records the resulting transitions, and can save an episode. Sessions themselves live in memory and are lost when the server restarts; saved episodes and job records persist. This is API-level control, not a gamepad or leader-arm teleoperation interface. FastAPI's `/docs` is the initial interface; a custom visual dashboard is not part of the implemented backend.

Episode import accepts raw NPZ data plus the original simulation seed/configuration. It rejects oversized or malformed arrays, then resimulates each submitted command and checks observations, rewards and terminal flags before saving a fresh manifest. Import therefore cannot simply assert that an arbitrary trajectory succeeded. This route is for this benchmark's simulation episodes, not arbitrary LeRobot datasets or physical recordings. Training checkpoints are created locally; there is no arbitrary checkpoint-upload or executable-plugin endpoint.

### One episode through the backend

1. `collect` resets the environment with a seed and asks the scripted controller for an action from the current state.
2. `step` queues the commanded target, advances physics and evaluates the independent task predicate. `rollout` records the transition without shifting its action label.
3. `save_episode` validates arrays, writes a fresh NPZ and publishes its JSON manifest last. An incomplete payload without a manifest is not listed as a completed episode.
4. `train_model` resolves explicit episode IDs, checks compatible contracts and duplicate data, then delegates to the learner. A completed `pipeline.json` links the model to episode IDs and payload hashes.
5. `evaluate` reloads a compatible checkpoint and uses new scenario seeds. Every trial receives an episode artifact; the final report contains its seed, outcome and episode ID.
6. `replay` resets from saved configuration and reissues recorded commands, comparing resulting observations. A rendered final frame is an aid to inspection, not a replacement for the full trace comparison.

Before modifying a controller, answer these with the implementation open:

1. Does the observation describe the state before or after the action stored beside it?
2. Is an action a joint target, velocity, torque, or Cartesian displacement? What frame and unit does each element use?
3. Does one policy sample advance one physics step or several? What is simulated time versus wall-clock time?
4. Are gripper opening and joint signs consistent across the robot model, expert and policy?
5. Is success checked after release and a settling interval, or only when the object crosses a target boundary?
6. Which failures are directly observed, and which are only heuristics? Do not call a timeout a grasp failure without evidence.

## Training and evaluation interpretation

A supervised loss measures agreement with demonstration actions. It does not measure closed-loop task success: a small prediction error can accumulate into states the demonstrations never visited. Compare the learned controller with the demonstration controller on the same evaluation seeds and perturbations. A completed evaluation reports all requested trials; if an exception aborts a run, the job fails instead of publishing a misleading success rate over only completed trials. Completed episode artifacts remain available. Do not pool a partially failed run with successful runs without explicitly accounting for the missing trials.

`POST /jobs/benchmark` runs the scripted baseline and selected checkpoints on the same seeds for each supplied environment condition. Its comparison report links to each underlying evaluation, rather than replacing individual failure traces with one pooled score. The local limit is 200 total rollouts across conditions and controllers. Success intervals express sampling uncertainty within those trials; they do not correct an inaccurate robot model or a weak success predicate.

Different seeds can still produce identical initial states when a fixed object position overrides random sampling. Evaluation fingerprints the actual initial observation, reports the number of distinct states, and omits the binomial interval for repeated states. For new checkpoints it also reports whether evaluated initial states are disjoint from the demonstrations. Counting duplicate deterministic trials as independent evidence would create unjustified confidence; seed uniqueness alone does not prevent that mistake.

When varying latency, state whether it delays observations, action commands, or both. When varying friction, identify the contact surfaces changed. Seeds make a scenario repeatable within the recorded software/model configuration; they do not guarantee identical results across different physics versions or machines.

In the current benchmark, the friction setting changes the sliding-friction coefficient of every model geometry, rather than only the block or gripper. Fresh evaluation seeds sample the same default position bounds unless the caller explicitly changes those bounds. Therefore, describe the default evaluation as unseen seeded initial states, not as evidence of out-of-distribution manipulation. Failure labels distinguish a fallen object and timeouts with or without an observed lift; they do not yet diagnose individual perception, grasp-planning or motor failures.

## Hardware boundary

Before connecting the physical kit, independently verify joint order, signs, zero offsets, joint limits, gripper calibration, camera transforms, control timing and emergency-stop behavior. The first hardware experiment should be a slow, bounded single-joint check under direct supervision. No current simulation metric should be presented as a physical success rate or safety certificate.

### Kit integration roadmap

1. **Inventory the assembled kit.** Record actual arm generation, motor/encoder models, firmware, mechanical limits and communication interface. Compare its geometry with the pinned SO-ARM100-generation simulation model. Do not infer compatibility from the XLeRobot name alone.
2. **Read state before commanding motion.** Add a separate hardware adapter only after the physical transport is known. Read joint positions with the kit's units, then verify order, sign, offsets and limit handling one joint at a time. Test disconnect and stop behavior independently of the learning policy.
3. **Bound direct joint commands.** Under supervision, execute small low-speed moves with explicit position, velocity and timing limits. Measure actual response, gripper opening and load behavior. Do not copy the simulation's normalized action ranges directly into motor commands without this calibration.
4. **Replace privileged state.** The current policy receives block pose and grasp/tray coordinates directly from MuJoCo. Physical operation needs measured perception or a redesigned observation contract: camera intrinsics/extrinsics, timestamp synchronization, object pose estimation, joint-state-derived end-effector position, and a policy for missing or uncertain measurements. A camera feed alone does not supply those state fields.
5. **Record matched observations/actions.** Establish a hardware episode format that preserves commanded and measured values with timestamps, calibration identity and stop events. Keep it separate from the current simulation importer until a validated conversion exists. Compare simulated and physical responses before choosing retraining, adaptation or a vision-based policy.
6. **Evaluate gradual task progression.** Start with supervised empty-gripper motions, then grasp/release a suitable test object under the kit's operating limits. Log failures and compare repeatable conditions. Only measured physical trials can support deployment claims.

None of these hardware components is implemented by the current simulation backend. The assembled kit will make them possible; it does not make a simulator checkpoint ready to execute on motors.

## 2026-09-29 audit: methods and decisions to learn by hand

An independent static review produced findings; each was reproduced before a fix was patched. No hardware was commanded.

### One control tick, before and after the timing fix

Start with `qpos(t)` and `qvel(t)`. Convert a normalized action into the six actuator position targets, pass it through the configured delay queue, and integrate 20 physics steps of 0.002 seconds. The new generalized state is at `t + 0.04` seconds. After the last integration, derived quantities such as the grasp site's world position can still describe the preceding state. Returning that geometry beside the new joints produces an internally inconsistent observation.

The regression moved the arm once, saved the returned observation, and independently recomputed transforms from the same resulting state. The old grasp position differed by about 1.52 mm. The fix is one `mj_forward` after integration, before contact checks and observation construction. It refreshes derived quantities without advancing time. Across the corrected scripted checks, the measured discrepancy is zero. Try writing this probe yourself before reading the fixed `step` function.

### An episode is more than a list of poses

For T actions, store T+1 observations and T rewards/terminal flags. A replay claim must compare all these outputs. Previously, altering a saved reward or final terminal flag could still produce `matched: true` because the poses matched. The regression now changes each of those fields independently while preserving valid storage checksums; replay must detect the semantic mismatch. Checksums alone only answer whether bytes changed relative to a manifest.

### A different random seed is not proof of a different state

Set a fixed block position equal to one demonstration's starting position and change only the seed. The scene can still be identical. Evaluation now records `held_out: false` and omits its interval when it overlaps demonstration states. Keeping this diagnostic is useful; counting it as unseen-state evidence is not.

### Test a learning hypothesis without rewriting the whole learner

On the frozen original simulator, two MLP experiments changed only the treatment of floored normalization features. Replacing their scales with 1.0 reduced seed-20000 action saturation from 99.806% to 23.889%; masking them reduced it to 20.306%. Every variant still failed all three nominal pickups. Thus normalization affects saturation but these changes do not establish a solution. Read [the experiment note](audit/NORMALIZATION_EXPERIMENT.md) for exact seeds, source identities and remaining failures.

A separate counterfactual changed the previous Rotation_L control from 0 to 0.2 rad without stepping physics. Observations remained byte-identical, while the scripted target changed by 0.047619 in normalized units. The expert's rate limiter uses command history that is absent from the policy input. This motivates a controlled observation-history experiment; it does not prove that hidden state caused all failed rollouts. Do not combine a new observation schema, normalization, architecture and dataset size in one experiment and then attribute improvement to one of them.

### Failure handling belongs to the resource owner

The job queue owns four active slots. A failed status write must not permanently retain a slot, and optional progress publication must not kill useful work. Both initial and final write failures were reproduced. Shared job read/write locking also fixes a reproduced Windows handle conflict during atomic replacement. Persistent disk failure can still leave stale status; that limitation is logged and restart marks it interrupted.

Rendering uses GLFW, whose context lifecycle belongs on the main thread. The local API's PNG handlers now run on the standard uvicorn main/event-loop thread. This simple choice blocks that local API briefly during bounded replay/rendering. If interactive latency becomes a real issue, move rendering into its own process rather than assuming a mutex makes arbitrary rendering threads portable.

### Evidence and restart trace

The corrected source passed 27 tests and 8 subtests. The scripted arm completed 16/16 fixed validation trials with finger contact during lift and release before success. Fresh HTTP recordings are in `data/audit-post-fix`; [REFERENCE_POST_FIX.json](audit/REFERENCE_POST_FIX.json) is the source for current learned-policy results. The first run was interrupted during ACT training after completing collection and MLP training. The unfinished job is retained as interrupted; `scripts/reference_experiment.py --resume` reuses completed report stages only when the simulator identity matches and reruns unfinished training. Its total elapsed time includes the interruption; `last_session_elapsed_seconds` measures the resumed session.

### Completed post-fix comparison

The corrected simulator produced 12/12 successful scripted demonstrations. The reused completed MLP checkpoint is `51f756a52daf4522989fc986cec675da` (80 epochs); the completed ACT-style checkpoint is `ff3daf9e8ce0443e8321faaca34901bf` (40 epochs). On the same three held-out seeds and three conditions, the scripted controller scored 9/9 and each learned model scored 0/9. MLP failures were nine `timeout_no_lift`; ACT-style failures were seven `timeout_no_lift` and two `fell_off_table`. This is a verified operational pipeline with unresolved learned control, not a successful learned manipulation system.

## Validation log

Historical results below precede the 2026-09-29 audit correction unless explicitly labeled otherwise. The corrected simulator has a new source identity; old episodes/checkpoints remain preserved and cannot be loaded as current-source evidence.

The local pipeline validation and reference experiment are complete. The learned manipulation task remains unsolved in the recorded evaluation. Record actual commands, date, outcome and artifact paths here; never promote intended behavior into a completed result.

| Date | Command / check | Observed result | Scope |
| --- | --- | --- | --- |
| 2026-09-28 | `.venv/Scripts/python.exe -m pytest -q tests/test_backend.py tests/test_learning.py` | 11 passed in 14.80 seconds (integrating agent's run) | Intermediate backend/learning integration; final simulator and full pipeline results follow below |
| 2026-09-28 | `.venv/Scripts/python.exe -m pytest -q` | 15 tests and 8 subtests passed in 17.64 seconds (integrating agent's run); one Starlette/httpx deprecation warning | Full suite at this checkpoint; implementation changes require relevant gates again |
| 2026-09-28 | `.venv/Scripts/python.exe -m unittest discover -s tests -p test_sim.py` | Four tests passed in 5.430 seconds (simulator agent's run) | Contact lift/release, float32 action replay, validation, portable asset integrity |
| 2026-09-28 | Scripted controller on seeds `0,1,2,3,4,17,42`; [raw results](SIMULATION_CHECK.json) | All seven completed the release-and-settle predicate at step 426; combined 26.548 seconds | Small default position rectangle, one scripted controller, simulation only; not learned-policy performance |
| 2026-09-28 | `.venv/Scripts/python.exe -m xle_workbench.cli --data data/act-smoke smoke --epochs 2 --model-type act` | Four demonstrations collected; ACT-style checkpoint trained; scripted evaluation 2/2, learned evaluation 0/2; replay matched with maximum observation error 0 | Full 600-step horizon; two-epoch CPU integration check, not a tuned learned controller; [report](../data/act-smoke/smoke-report.json) |

The ACT smoke checkpoint is `f8cb66dd5ba64fc7814030620c5bfcb3`. Its best held-out action MAE was approximately 0.08449 in normalized action units after two epochs; that supervised error coexists with zero successful learned rollouts in the two evaluation trials. The distinction is the main learning result of the smoke test: an operational training pipeline is not yet a reliable manipulation policy. Increase data coverage and inspect closed-loop failure traces before claiming improved task performance.

### Completed reference experiment: pipeline execution, not learned competence

The real local HTTP API collected 12 successful scripted demonstrations on seeds 100 through 111, trained an MLP for 80 epochs and an ACT-style model for 40 epochs, then compared the scripted and learned controllers on seeds `20000, 20001, 20002` under three conditions. The raw record is [REFERENCE_EXPERIMENT.json](REFERENCE_EXPERIMENT.json), including checkpoint IDs and links to every saved evaluation. The run took approximately 266.73 seconds on the local CPU environment.

| Condition | Scripted controller | MLP | State ACT-style model |
| --- | --- | --- | --- |
| Friction 1.0, no command delay | 3/3 successful | 0/3 successful | 0/3 successful |
| Friction 0.6, no command delay | 3/3 successful | 0/3 successful | 0/3 successful |
| Friction 1.0, two control steps of command delay (80 ms) | 3/3 successful | 0/3 successful | 0/3 successful |

Each learned controller finished all nine trials with `timeout_no_lift`. The same three seeded starting positions are reused across conditions for paired comparison; this is nine scenarios per controller, not nine independently sampled positions. The task's small position rectangle and limited conditions do not establish broad robustness or hardware performance.

The MLP's best validation action MAE was approximately 0.00606235 at epoch 74; the ACT-style model's was approximately 0.00976908 at epoch 37. Neither lower supervised error translated into a successful learned rollout. Checkpoints `4a4da53545254522b8a84b78d7ac7a0b` (MLP) and `5790fe37301d44c79619fb82d13c699a` (ACT-style) are evidence of completed training, not task competence.

Replay of recorded demonstration `67fc1cebde7c42a6be4d76d90515061d` matched all 426 steps with maximum observation discrepancy 0. This validates the tested command-recording and resimulation path. It does not validate a hardware adapter, perception pipeline or deployment.

### Failure investigation: small action error can leave the training distribution

A read-only review of nominal seed 20000 found a concrete state-distribution shift. Demonstration wrist-roll velocity ranged from approximately -0.0003244 to 0.0002579 rad/s; its normalization scale hit the floor of 0.0001. The MLP's first wrist action was -0.006806 in normalized command units, while the scripted controller commanded zero. After one control tick, wrist velocity was approximately -0.240629 rad/s, or 2,406 normalized units; after the second tick it reached approximately 8.004195 rad/s, or 80,042 normalized units. About 99.81% of this failed MLP episode's action components had magnitude above 0.99. The trace is episode `1ea39921dea8492ba80e504dedbe0a76` in the local data store.

The ACT-style episode `d7982e4b304342e9b8de6539c94190d7` also entered states far outside the demonstration range: its first wrist command was approximately 0.009089 and first-tick wrist velocity approximately 0.320887 rad/s, or 3,209 normalized units. Its action components did not meet the same saturation threshold, so the MLP saturation mechanism must not be asserted for both models.

These measurements establish distribution shift. They suggest that near-constant feature scaling amplifies small closed-loop deviations, but they do not establish the sole cause of failed grasping. The next manual exercise is a controlled comparison: preserve the dataset, architecture, seeds and physics; change only normalization handling for near-constant channels; retrain and compare the same failure traces and task outcomes. Inspect initial action errors and wrist dynamics as well as loss. Do not change several controller and normalization settings together and claim a causal explanation.
