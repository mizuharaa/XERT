# Rebuild the imitation-learning module by hand

This file records what was implemented in `xle_workbench/learning.py`, why, and
how to reconstruct it. The first version was agent-written at the user's request.
Reading it is not a substitute for implementing and explaining the fundamentals.

## Scope and evidence

Two real PyTorch learners train on simulator **state vectors**:

- `baseline_mlp`: two hidden ReLU layers, one bounded action per observation.
- `act`: a compact **ACT-style state-conditioned CVAE Transformer**, predicting
  a chunk of future actions. This is a local teaching implementation, not the
  official visual ACT model, a LeRobot checkpoint, or a hardware deployment.

The algorithm references are the [ACT paper](https://arxiv.org/abs/2304.13705),
the authors' [model implementation](https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py),
and [loss implementation](https://github.com/tonyzhaozh/act/blob/main/policy.py).
The code here was written independently using PyTorch's standard Transformer
layers. ACT's core pattern conditions an action-sequence generator on state and
a training-time latent inferred from demonstrations. We retain that pattern.
We omit image features, use a smaller network and latent, bound outputs with
tanh, and default to non-overlapping chunks. Optional temporal ensembling
queries the model every tick and averages predictions for that exact tick.

State inputs can include simulator-only object measurements. A working policy
therefore does **not** establish visual perception or sim-to-real performance.
The training routine reports held-out action error, never physical task success.
Use the rollout evaluator for grasp/release outcomes.

## Data contract: align time before building a model

An episode is a dictionary with these arrays:

```python
observations.shape == (T + 1, D)  # float32; ordered state values
actions.shape == (T, A)          # float32; normalized controls in [-1, 1]
```

At time `t`, the policy sees `observations[t]`, issues `actions[t]`, and the
simulator produces `observations[t+1]`. The last observation has no corresponding
demonstration action. Do not shift targets by one and silently learn a different
problem. The simulator defines the units and ordering; pass its exact ordered
`observation_names` and `action_names` in the training config. Dimensions alone
cannot distinguish different robot joints or action conventions.

The module rejects mismatched dimensions, missing terminal states, empty
episodes, non-finite data, and out-of-range actions. It does not silently clip bad
training data. Success flags are not inserted into the input vector by this
module: using future success to predict current actions would leak information.

## Episode split and normalization

1. Shuffle **episode indices** with `numpy.random.default_rng(seed)`.
2. Hold out `ceil(n * val_fraction)` complete episodes, with at least one episode
   on each side. Two is a software minimum, not a useful dataset size.
3. Reject byte-identical episodes appearing on both sides.
4. Compute observation mean and population standard deviation from **training
   observations with action targets only**. Use a floor of `1e-4` for scale.
5. Apply `(observation - mean) / scale` to both partitions and future inference.

Neighboring frames share almost the same state. Randomly splitting frames would
let validation see parts of training trajectories. The split and normalization
rules prevent that obvious leakage. Exact duplicate checks do not detect near
duplicates or repeated random seeds; the surrounding dataset/evaluation service
must preserve seed and task provenance. Validation is used to select a model and
is not an untouched final test set.

Actions are already physically normalized by the simulator. They are not
standardized again. Metadata stores normalization arrays, split indices, episode
SHA-256 fingerprints, dimensions, names if supplied, and the resolved config.
Subtraction runs in float64 before returning float32 tensors: two individually
finite float32 values can have a difference outside float32 range. Normalized
inputs outside that range are rejected. This also prevents a tiny feature scale
from silently turning a large inference observation into infinity.

## Chunk construction and padding

For a chunk length `K`, one training example at time `t` contains:

```text
input:    state[t]
targets:  actions[t : min(t+K, T)]
padding:  zero actions to reach K entries
valid:    True for actual actions, False for padding
```

Example: `T=5, t=3, K=4` gives two real actions and two padding slots.
`ActionChunks` forms these windows lazily; it never crosses an episode boundary.
The posterior attention mask excludes padding. Reconstruction error also excludes
padding and divides by `number_of_valid_steps * action_dimension`, so terminal
chunks are not rewarded for learning artificial zeros.

## Baseline first

The baseline is `state -> Linear/ReLU -> Linear/ReLU -> Linear/Tanh -> action`.
It minimizes squared action error. `tanh` gives bounded controls; inference also
clips defensively. This is supervised behavior cloning, not reinforcement learning.

Start here because a model that cannot overfit a small clean demonstration set
usually exposes timing, normalization, or observation problems. A more complicated
Transformer will not repair a mislabeled dataset.

## State ACT: understand each tensor

Let the batch size be `B`, state dimension `D`, action dimension `A`, model width
`H`, chunk length `K`, and latent dimension `Z=16`.

During training:

1. Project the state to `(B,1,H)` and actions to `(B,K,H)`.
2. Concatenate a learned summary token, state token, and action tokens.
3. Add learned position embeddings and run one four-head Transformer encoder
   layer, masking padded action tokens.
4. Project the summary token to latent mean `mu` and log variance `logvar`, each
   `(B,Z)`. Clamp log variance to `[-10,10]` for numerical stability.
5. Reparameterize: `z = mu + exp(0.5 * logvar) * epsilon`, with standard Gaussian
   `epsilon`. This permits gradients through the sample transformation.
6. Use projected state and latent as decoder memory. Decode `K` learned query
   slots in parallel with one Transformer decoder layer. A linear head and tanh
   produce `(B,K,A)`.

The loss is:

```text
reconstruction = sum(abs(prediction - target) * valid) / (sum(valid) * A)
KL = mean_over_batch(0.5 * sum(mu^2 + exp(logvar) - 1 - logvar))
loss = reconstruction + kl_weight * KL
```

The encoder can see demonstration future actions during training, but the rollout
policy cannot. Validation therefore calls the decoder with `z=0` and **does not
feed validation targets into the encoder**. Best-checkpoint selection uses this
prior-inference action MAE. Reporting posterior reconstruction instead would be
overly optimistic about what the deployed policy actually knows.
The MAE pools every valid action element across overlapping evaluation chunks.
It is action-space prediction error at demonstration states, not rollout reward
or an estimate of grasp success. Later actions can appear in multiple chunks;
this is intentional next-chunk evaluation, not independent observations.

## Execution, cancellation, and checkpoints

`train(episodes, output_dir, config, progress=None, cancelled=None)` returns
JSON-safe metadata. Defaults are 20 epochs, CPU, batch size 64, hidden width 64,
and learning rate 0.001. ACT defaults to 8-step chunks and KL weight 0.01. The
baseline always uses chunk size one. Set a small CPU thread count once in the
calling application for small models rather than paying large thread-pool costs.

Training checks cancellation at every batch and before saving. Gradients are
clipped to norm one; non-finite loss or gradients abort. The best state dictionary
is copied to CPU and saved after training, with metadata published last. Output
directories containing a previous model are rejected to avoid accidental replacement.
The model contains inference weights, not optimizer state for resuming training.

`load_policy(path)` verifies a weight hash and uses `torch.load(...,
weights_only=True)`. Hashing catches accidental file mismatch; it does not prove
who created an artifact. Only load locally managed checkpoints. Inference is CPU
and requires the saved dimensions and finite state values.
Loading also rejects non-finite tensors and invalid normalization, including
values that cannot be represented in float32 and scales below the training
floor. The pipeline's separate provenance record ties training data and the
robot observation/action contract to the model; matching dimensions alone is
insufficient.

`policy.act(observation)` computes a chunk only when its queue is empty, then
returns queued actions one per simulator control tick. That is **open-loop within
each chunk**, even though callers provide a fresh observation each tick. Call
`policy.reset()` at every episode boundary. No hidden phase estimator, scripted
expert fallback, or target actions are used during inference.

With `temporal_ensemble=true` for ACT, the policy instead predicts a fresh chunk
on every observation. Each active chunk contributes its action for the current
tick, weighted by `exp(-ensemble_decay * age_in_ticks)`. This implementation
favors newer predictions; zero decay gives equal weights. Reset clears every
active chunk, so predictions cannot cross episode boundaries. Old checkpoints
and the default configuration retain their original non-overlapping execution.

Seeded episode splits, initialization, shuffling, and latent sampling make CPU
runs repeatable in the same software environment when jobs are serialized.
Global PyTorch RNG state is used, so do not train concurrently in separate threads
and claim reproducibility. CUDA is optional and must be requested explicitly;
cross-device or cross-version bitwise equality is not promised.

## Hand-coding sequence and checks

1. Implement one tiny trajectory and print `(state[t], action[t], state[t+1])`.
2. Write the episode split and prove held-out outliers cannot change training
   normalization. Exclude the terminal state too.
3. Write `ActionChunks`; assert the last window's valid mask by hand.
4. Write the baseline and fit a synthetic known mapping. Save and reload it,
   checking the exact same input produces the same output.
5. Write masked L1. Set padding errors to 1000 and prove loss is unchanged.
6. Add a deterministic Transformer chunk decoder; then add the CVAE encoder,
   reparameterization, and KL term one at a time.
7. Verify inference never reads target actions. Compare `z=0` validation with
   training reconstruction and explain why they differ.
8. Run real simulator evaluations on new seeds. Inspect failures before adding
   layers or increasing epochs.

The executable regression checks are in `tests/test_learning.py`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_learning.py -q
```

Passing these tests establishes data-contract and training/inference behavior on
small synthetic examples. It does not establish that a trained robot completes
the manipulation task. That requires separately recorded rollout evidence.

## Verification performed for this implementation

The command above completed with **7 passed in 13.03 seconds** in this workspace
using Python 3.11.9, NumPy 2.4.6, and PyTorch 2.14.0+cpu. CUDA availability was
`False`; these checks did not train on the installed graphics card.

The tests actually train and reload both model types. They check decreasing
baseline held-out error on a synthetic mapping, episode-level split isolation,
exclusion of terminal observations from normalization, constant-feature scales,
ACT validation without posterior target access, posterior padding invariance,
deterministic prior inference, chunk reset behavior, cancelled jobs, duplicate
episodes, corrupted weights, and non-finite checkpoint/input rejection.
This is a recorded run, not a promise that tests remain green after later edits.

When rebuilding by hand, first reproduce the baseline round trip and masked
chunk assertions. Then recreate the ACT test with the same random seed and
change only padded actions: the posterior output must remain unchanged. Finally
remove target actions entirely for validation and explain why prior error can
stay high even when posterior reconstruction improves.
