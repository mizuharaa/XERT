# Rebuild the fundamentals by hand

Treat the generated implementation as a reference you can run and interrogate. Make a separate learning checkout or directory, type each milestone yourself, and compare behavior with the reference only after your own check passes. Do not rewrite the entire system at once.

## Use the reference without letting it do the learning

For each milestone, keep a short handwritten prediction, your implementation, one failing input, and the check that now passes. Read the named reference function only after trying your version. Ask an assistant to explain a failing trace or review your reasoning before asking it to replace your code.

| Milestone | Reference to open afterward |
| --- | --- |
| Model, frames and state | `sim.py`: `model_xml`, `ManipulationEnv.metadata`, `_obs` |
| Actions and timing | `sim.py`: `normalize`, `reset`, `step` |
| Scripted controller and task | `sim.py`: `expert_action`, success predicate in `step` |
| Episode storage | `pipeline.py`: `rollout`; `storage.py`: `validate_arrays`, `save_episode` |
| Behavior cloning | `learning.py`: `episode_split`, `ActionChunks`, `BaselineMLP`, `train` |
| Action chunks | `learning.py`: `StateACT`, `masked_loss`, `LoadedPolicy.act` |
| Evaluation | `pipeline.py`: `evaluate`, `replay`, `load_verified_policy` |
| Backend jobs | `jobs.py`: `Jobs.submit`, `cancel`, `close` |

All source names above are relative to `xle_workbench/`. Keep your exercise directory separate so the working reference remains available for comparisons.

## 1. Load the robot and name the state

Load the model, print every joint and actuator name, then inspect its range. Write down the mapping from each control element to the joint it affects. Find the world frame, robot base frame and end-effector frame; calculate one point expressed in two frames.

**Check:** moving one target changes the intended joint; resetting restores the same initial state. Explain why a generalized-position index need not equal an actuator index.

## 2. Make time and actions explicit

Write a function that validates one action, clamps or rejects it according to the chosen contract, applies it and advances exactly one control interval. Keep the smaller physics timestep separate. Expose the limits and timing as readable values.

**Check:** the expected number of physics steps advances the expected simulated duration. NaN, infinity and incorrectly sized actions are rejected before reaching MuJoCo. Describe what happens when the policy takes longer than its control interval.

**Hand calculation:** for joint limits `lo` and `hi`, derive `target = lo + (action + 1) * (hi - lo) / 2`. Show that -1, 0 and +1 map to the lower limit, midpoint and upper limit. Twenty physics steps at 0.002 seconds make a 0.04-second control interval; that is 25 Hz, regardless of how fast the computer finishes it. Trace a two-step command delay with three different commands and the initial hold target. The controller records issued commands; the simulator FIFO determines when each reaches the actuator.

## 3. Define the task before the controller

Place a free object and target tray. Define a success predicate using position, release and stability, not the controller's internal phase name. Write a scripted approach/grasp/lift/place/release controller. Inspect contact behavior when it fails.

**Check:** an object carried over the tray is not yet a completed placement. A missed grasp must not move the object by attachment or teleportation. Save one success and one failure if the controller can produce them; otherwise record that the task is still unsolved.

**Hand calculation:** start with position-only damped least-squares inverse kinematics: `dq = J.T @ solve(J @ J.T + lambda * I, position_error)`. Write the shape and units of each term. Explain what damping does near a singularity and why clipping the update differs from clipping the final joint target. The reference adds an orientation objective; its 0.1 weight balances different residual units and is a task-specific choice, not a calibrated physical constant. IK finds a target configuration; motor commands and contacts still determine whether a grasp works.

## 4. Record an episode without shifting labels

Store the observation at time t beside the command chosen from it, then store the resulting next state. Add the seed, task settings, robot model identity and controller identity. Start with a small NumPy archive and JSON metadata.

**Check:** archive lengths and finite values agree; commands have the documented ordering; interrupted writes cannot appear as completed episodes. Replay an episode and explain whether you are replaying recorded states or resimulating actions.

**Three-step trace:** write `o0, a0, o1, a1, o2, a2, o3` on paper. The archive contains four observations and three actions. Training pairs are `(o0,a0)`, `(o1,a1)`, `(o2,a2)`; `o3` has no target action. Then introduce an off-by-one error deliberately and identify why the resulting supervised problem is wrong even though the tensor dimensions can look reasonable.

## 5. Train the simplest behavior-cloning baseline

Split entire episodes into train and validation groups. Fit observation mean and scale on training data only. Write a PyTorch MLP that maps normalized state to actions already expressed in the simulator's [-1, 1] command contract, then save its architecture, weights, normalization and source episode identifiers together.

**Check:** it can overfit a tiny demonstration subset. Changing validation data does not change normalization. Closed-loop evaluation is separate from validation loss. Explain why frame-wise random splits would leak near-duplicate states.

## 6. Predict action chunks

Replace the one-action head with a small sequence model that predicts a fixed chunk of future commands. Mask missing targets at an episode's end instead of training on invented padded actions. State when inference replans and how many predicted actions execute before replanning.

**Check:** a masked padded target cannot change the loss. Chunk boundaries cannot read actions from the following episode. Trace the conditional variational encoder used during training and the zero latent used at inference. Describe why the absence of camera inputs and temporal ensembling makes this a state ACT-style baseline, not a reproduction of the full published ACT system.

**Boundary exercise:** at `t=3` in a five-action episode, a four-action chunk contains actions 3 and 4 followed by two padding rows. Its validity mask is `[true, true, false, false]`. Neither the reconstruction loss nor the encoder attention should treat padded rows as real commands. At deployment, the reference predicts a chunk from the current observation, then executes the whole chunk before replanning. Explain how increasing chunk length changes the interval during which fresh observations cannot alter the queued plan.

## 7. Evaluate the policy as a controller

Run the scripted baseline and learned checkpoint on the same fixed scenario list. Start with new episode seeds, then vary one factor at a time: object position, contact friction or command delay. Preserve per-episode outcomes, configuration and traces.

**Check:** the reported success denominator includes all requested trials. A repeat uses the same scenario list. Report the actual failure reason only when measured; retain an unknown/timeout category.

**Experiment exercise:** hold the seeds, starting positions and friction fixed while changing only command delay. Compare the scripted and learned controllers on that same list. Preserve failures, including runs where no object was lifted. Distinguish a fresh random seed from a held-out spatial region: a new seed drawn from the training bounds tests new samples, not extrapolation outside those bounds.

## 8. Put a small API around the working loop

Only after the command-line loop works, expose dataset collection, training and evaluation as bounded jobs. Persist job state and artifact paths, validate identifiers and allow only known operations. Add the interface last.

**Check:** malformed requests cannot escape the artifact directory or execute arbitrary commands. A failed job retains its error; a process restart does not claim an interrupted job succeeded. Two simultaneous requests cannot silently overwrite the same artifact.

## What to be able to explain in an interview

- Trace one observation through normalization, the model, action-to-joint-target conversion and robot control.
- Explain one real failure using a recorded episode, not only a scalar score.
- Identify which simulator observations require perception on hardware.
- Reproduce an evaluation from its checkpoint and recorded scenario configuration.
- Show a small component you personally rebuilt and the check that caught a mistake.

Do not count reading generated code as having mastered these steps. The evidence is being able to reconstruct, change and debug each component yourself.

## Your first investigation using the saved failures

The completed reference pipeline trained both models, but neither completed the held-out pick-and-place trials. Use the saved evidence to investigate one concrete issue before adding features.

Open MLP episode `1ea39921dea8492ba80e504dedbe0a76` and ACT-style episode `d7982e4b304342e9b8de6539c94190d7` in the `data/episodes` store. Both start from nominal seed 20000. Find `joint_velocity.Wrist_Roll_L` by the saved observation-name list, not a guessed column number.

1. Write down the raw initial state, its saved normalization mean/scale, the first wrist command, and the next two observed wrist velocities.
2. Calculate `normalized = (raw - mean) / scale` by hand. This feature's saved scale is 0.0001 because the demonstration velocities barely varied. The MLP trace moves to about -0.240629 rad/s after 0.04 seconds, roughly 2,406 normalized units; after 0.08 seconds the magnitude is roughly 80,042 normalized units.
3. Trace the next policy output. The MLP episode shows widespread command saturation, whereas the ACT-style episode has severe normalized-state shift without the same saturation pattern. Explain why these are distinct observations.
4. Form one hypothesis about near-constant feature normalization. Write a small check for your proposed transform before changing training. Keep data, model, physics and evaluation seeds fixed while testing that one change.
5. Compare task success and complete state/action traces, including failures. Lower validation MAE alone is not evidence that your change fixed closed-loop control.

The existing code has not been modified to resolve this finding, and no corrective experiment has yet been performed. This is a measured distribution-shift observation with a plausible hypothesis to test, not a proven diagnosis or promised fix. The fuller measurements are in [ENGINEERING_NOTEBOOK.md](ENGINEERING_NOTEBOOK.md).
