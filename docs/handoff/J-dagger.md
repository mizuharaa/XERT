# Slice J: DAgger scaffolding on the pick-and-place bench

Branch `feat/j-dagger`, from `integration/sim-mvp` @ fe96707. Plumbing, tests and docs only. **No DAgger run has
happened and no DAgger number exists.** The mixing rule, the stopping rule and the training settings are
`TODO(daniel)` stubs. No ledger rows, no hardware, no new dependency.

## What changed and why

- **`xle_workbench/storage.py:27` `CORRECTIVE_SOURCES`** (review J-01). A Store episode is two files:
  `episode.npz` holds the EXECUTED actions and `corrective.npz` the expert labels for the same transitions.
  `Store.save_episode` and `Store.load_training_episode` used to accept labels only for
  `source == "scripted_simulation_noise"`. Both now share one tuple that also holds the new
  **`source = "dagger_policy_rollout"`**. The existing checks are unchanged: a corrective source cannot be saved
  without labels (otherwise training would silently fit the executed actions), any other source cannot carry
  labels, and the loader still checks hashes and alignment.
- **`xert/eval/dagger.py`**, CLI `python -m xert dagger init | iterate`.
  - **`init --collection <dir> --output <fresh run dir> [--eval-seeds ...]`** (review J-02). The collection is
    the P1.2 format (`collection.json` beside a `dataset/` Store). Every seed episode must have the current
    contract, `pipeline.contract(env.metadata())`, for the pick-and-place bench env (`xlerobot_left`). The
    contract is recomputed from each episode's `simulation` metadata, not trusted from its stored
    `contract_sha256`. Each episode's `config` must also equal the bench env's config (robot, friction,
    delay_steps, horizon, object_position_bounds), because the contract omits friction, delay and bounds while
    rollouts and evaluation use the bench config; the error names the differing keys. Owner decision: randomized
    seed data would be a separate decision and would need the same randomization in the rollouts. These checks
    read metadata first, so a stale collection is refused before any payload is decoded. Then each
    episode loads through `load_training_episode` (payload hash, labels, alignment). Seeds must be distinct and
    must not overlap the held-out seeds (the P5.3 check; default 10000-10029). `init` writes
    `iteration_000/seed.json` (collection path, `collection.json` sha256, per-episode payload and label hashes)
    and `manifest.json`. It references the seed data and never copies or edits it.
  - The checked-in `results/p1.2/recovery_integrated` is refused. Its episodes carry `implementation_sha256`
    473fff47 (from before slice B). The current core is 07f21e5d (contract fd4b60c9, measured on this branch).
    The error names the recollection command:
    `python -m scripts.collect_recovery --config configs/recovery_demos.json --output <fresh directory>`.
  - **`iterate --run <dir> [--retry]`** runs ONE iteration:
    1. **Train.** `pipeline.train_model` on the aggregate (the seed episodes plus every committed iteration's
       episodes), with `dagger_settings(...)["training"]`. A small `_Aggregate` adapter gives train_model the two
       Store methods it calls across several Stores, so `train_model` still runs every check (single contract,
       duplicate payloads, distinct seeds). `learning.train` is never called directly.
    2. **Roll out.** `dagger_settings(...)["rollouts"]` bench episodes. The seeds are the lowest unused seeds
       above the aggregate's, skipping held-out ones. The episodes run through `pipeline.rollout`, with a policy
       slot shaped like P1.2's `_NoisyExpert`: at each tick it records `env.expert_action()` as the label, BEFORE
       the step, and then executes `dagger_action(policy_action, expert_action, iteration, step)`.
    3. **Stage.** The episodes are saved with `source="dagger_policy_rollout"` and the provenance under
       `metadata["dagger"]`: iteration, attempt, seeds, policy model id, sha256 and path, expert
       (`Env::expert_action`), expert `implementation_sha256`, and the mixing rule's qualified name.
       `collection.json` lists them in the P1.2 format, so `xert export-lerobot` can read an attempt directly.
    4. **Evaluate.** The checkpoint loads through `pipeline.load_verified_policy` against the bench env.
       `xert.eval.gate_experiment.rollout` (the P5.3 per-seed rollout) runs each fixed held-out seed, and
       `pipeline.wilson` gives the interval. The results go to `report.json`.
    - `should_stop(history)` runs before every iteration, on the reports of the completed ones.
  - **Until the stubs exist**, `iterate` stops with `DAgger mixing rule not chosen (TODO(daniel))` before it
    writes anything. It does this by calling `dagger_action` once on zero actions (`dagger.py:225`). `init`
    works now.
- **CLI**: `xert/__main__.py` registers `dagger`.
- **CI**: `tests/test_dagger_lerobot.py` added to the `lerobot` and `teleop` jobs in `.github/workflows/optional.yml`.
  `tests/test_dagger.py` runs in the base job (`linux.yml` runs all of `tests/`).

## Observation contract (per tick t, both see the same pre-step state)

- **Policy:** the P1.3 vector. These are the 26 float32 values named by
  `ManipulationEnv.metadata()["observation_names"]`: 6 joint positions, 6 joint velocities, the block pose
  (xyz + wxyz), the grasp site, the tray and the time fraction. `reset()`/`step()` return this vector, and it is
  what the P5.3 `bench_observation` builds by name on evaluation.
- **Expert:** `Env::expert_action()` on the same MjData. It is privileged: it reads the simulator directly, not
  the vector.
- Both are queried before `env.step(executed)`. `episode.npz` stores observation[t], the executed action[t] and
  observation[t+1]. `corrective.npz` stores the same observations, rewards and flags with expert[t] as the action.
  Actions are normalized absolute joint targets in [-1, 1]; index 5 is the gripper.

## Atomic iterations (review J-04)

```
<run>/manifest.json                          the ONLY commit record (storage.atomic_json: temp file + os.replace)
<run>/iteration_000/seed.json                seed reference written by init
<run>/iteration_NNN/attempt_K/dataset/       Store: models/<id> (checkpoint), episodes/<id> (rollouts)
<run>/iteration_NNN/attempt_K/collection.json
<run>/iteration_NNN/attempt_K/report.json
```

- Iteration N is complete iff `manifest.json` lists it. There is no COMPLETE marker. Any attempt directory the
  manifest does not reference is incomplete.
- Each attempt directory is created fresh (`mkdir` without `exist_ok`). The Store's per-save UUIDs and the
  trainer's refusal of a populated checkpoint directory therefore never collide.
- An incomplete attempt blocks a plain `iterate`. `--retry` makes `attempt_{K+1}` and deletes nothing. The
  committed entry lists the abandoned attempts in `incomplete_attempts`. `--retry` with nothing to retry is
  refused.
- Nothing referenced is rewritten. The manifest stores each episode's payload and label sha256, and each report's
  sha256. Training (`_Aggregate`) and the `should_stop` history read check those hashes.
- A retry draws the same rollout seeds, because they come from the committed aggregate. An abandoned attempt can
  therefore never cause a duplicate seed or episode.
- Compare before replace: `iterate` keeps the manifest bytes it started from and re-reads them just before the
  replacement. If they differ it refuses with "manifest changed during this iteration" and the attempt stays
  unreferenced. The re-read and the replace are not one atomic step, so this guards a single writer against a
  stray edit, not concurrent writers (that would need a lock file).

## How to verify

```
.\.venv\Scripts\python.exe -m pytest -q tests/test_dagger.py              # 10 passed, 2 xfailed
.\.venv-lerobot\Scripts\python.exe -m pytest -q tests/test_dagger_lerobot.py tests/test_lerobot_export.py tests/test_lerobot_io.py
.\.venv\Scripts\python.exe -m xert dagger init --collection results/p1.2/recovery_integrated --output build/dagger_check   # refused (473fff47)
```

The real loop, once the stubs exist:
```
.\.venv\Scripts\python.exe -m scripts.collect_recovery --config configs/recovery_demos.json --output results/p1.2/recovery_<name>
.\.venv\Scripts\python.exe -m xert dagger init --collection results/p1.2/recovery_<name> --output results/dagger/<run>
.\.venv\Scripts\python.exe -m xert dagger iterate --run results/dagger/<run>      # repeat; --retry after a failure
```
`collect_recovery` is P1.2's script and appends its own P1.2 ledger rows. This slice writes none.

What each test shows (`tests/test_dagger.py` unless named):
- `test_dagger_source_shares_the_corrective_label_rule`: the new source needs labels, `load_episode` returns the
  executed stream and `load_training_episode` returns the labels. Other sources still refuse labels.
- `test_init_*`: provenance, run through the real `xert` CLI. The checked-in collection is refused with
  473fff47 and the command. Evaluation seeds that overlap the demonstrations are refused. Nothing is written on
  refusal.
- `test_iterate_refuses_until_the_rules_exist`: the exact message; no attempt directory; manifest bytes unchanged.
- `test_two_iterations_relabel_replay_train_on_labels_and_append` (J-03). The fixture rule is
  `clip(0.5 * (policy + expert))`: bounded, and the test asserts it differs from the expert. The test asserts:
  - `episode.npz` equals the rule's outputs, and `corrective.npz` equals the expert values passed to it;
  - replaying the executed actions from the seed gives the same expert value at every state (queried before the
    step), and `pipeline.replay` matches;
  - the provenance fields are present;
  - the evaluation covers 2 seeds, with Wilson;
  - iteration 2 leaves every iteration-1 file hash and manifest entry unchanged;
  - the arrays handed to `learning.train` are the corrective labels, which differ from the executed ones.
- `test_failure_after_collection_before_the_report_then_retry`: evaluation fails after staging. The manifest bytes
  are unchanged, a plain `iterate` is refused, `--retry` commits attempt 2, and attempt 1 stays on disk unchanged
  and unreferenced. There are no duplicate ids or seeds.
- `test_interruption_around_the_manifest_replacement[before|after]`: the crash is injected at `atomic_json` of
  `manifest.json`. "before": the manifest is unchanged and the retry adds iteration 1 once. "after": iteration 1
  is committed, and the next `iterate` runs iteration 2 without touching iteration 1's files. In both cases every
  referenced payload still loads with its hash.
- `tests/test_dagger_lerobot.py`: P2.1 export writes `action` = corrective and `xert.executed_action` = executed
  for DAgger episodes. The slice C import restores both streams bit-exactly and keeps `source` and the `dagger`
  provenance.

## TODO(daniel) (CORE: the DAgger loop)

- `xert/eval/dagger.py:43` `dagger_action(policy_action, expert_action, iteration, step) -> (6,) action`: the
  mixing rule (which action is executed).
- `xert/eval/dagger.py:58` `should_stop(history) -> bool`: the stopping rule, asked before each iteration.
- `xert/eval/dagger.py:71` `dagger_settings(iteration, aggregate) -> {"training": P1.3 config, "rollouts": N}`:
  hyperparameters and rollouts per iteration.
- **Data weighting:** there is none. `learning.train` fits every aggregated transition once. Weighting would be a
  trainer change (CORE: policy training logic), so no stub was added for it.
- Done tests (strict xfail, `raises=(NotImplementedError, AssertionError)`):
  - `tests/test_dagger.py:267` `test_daniel_rules_exist_and_return_valid_values`;
  - `tests/test_dagger.py:280` `test_daniel_two_iterations_grow_the_aggregate_by_the_rollout_count`. It runs YOUR
    settings on a 3-episode fixture, so big epochs or rollout counts make it slow. It is a harness check with no
    performance bar. Pre-register the success target before the first real iteration.

## What is NOT shown

- No DAgger result, no success rate, no learning curve. The tests' 2-seed evaluations are harness checks on
  untrained, 1-epoch policies, and no number from them is reported anywhere.
- No recollected 100-episode seed collection exists on the current core. The checked-in one is stale (473fff47).
- No ledger rows. The seed data and the relabelling expert are scripted (the P5.3 rule), so per-iteration
  `report.json` files are the record.
- Only the `xlerobot_left` bench. The SO-101 scripted expert is still TODO(daniel) (`xle_workbench/so101_expert.py`).

## Interview questions

1. **Why record the expert label while executing the mixed action?** The executed action decides which states
   the robot visits, and those are the policy's own mistakes. The expert label says what to do in those states,
   so training on labels at visited states fixes the distribution shift that plain behaviour cloning has.
2. **Why is the manifest the only commit record, with no COMPLETE marker?** One atomic `os.replace` has no
   window where a marker and the data disagree: an iteration either is in the manifest or it is not, so a crash
   at any point leaves a state that a retry can reason about.
3. **Why refuse the checked-in P1.2 collection instead of training on it?** Its contract hash names an older
   simulator. Pooling it with new rollouts would mix two physics implementations under one policy, and
   `train_model` would refuse anyway. Recollecting is cheap and keeps the contract honest.

## Open questions

- `dagger_action` gets no RNG or episode seed. A stochastic beta-mixing rule (classic DAgger) needs the seed in
  the signature, or it has to be deterministic in (iteration, step). Changing the signature is your call.
- Resolved (owner default = yes): `init` requires the bench config as well as the contract. The real
  `configs/recovery_demos.json` uses `{}`, which is the bench default.

## Integration lines (not edited here)

- README, SO-101 track / learning section: "`python -m xert dagger init|iterate`: DAgger plumbing on the
  pick-and-place bench; rules are TODO(daniel) (slice J, docs/handoff/J-dagger.md)."
- `docs/handoff/TASKS.md`: "J DAgger scaffolding: built on feat/j-dagger; awaits QA. TODO(daniel): dagger_action,
  should_stop, dagger_settings (xert/eval/dagger.py:43/58/71)."
- MVP_PROGRESS: "DAgger plumbing done, no DAgger result; blocked on Daniel's rules and a recollected seed set."
