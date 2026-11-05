"""Slice J: DAgger plumbing on the pick-and-place bench (xlerobot_left, the one bench with a working scripted expert).

CORE is Daniel's: `dagger_action` (mixing rule), `should_stop` (stopping rule) and `dagger_settings` (training
hyperparameters and rollouts per iteration) are TODO(daniel) stubs. This module owns everything around them:
seed-collection validation, rollouts with expert relabelling, staging with provenance, held-out evaluation with
the P5.3 harness, per-iteration reports and the run manifest. No ledger rows: the seed data and the expert are
scripted (the P5.3 rule), so reports are files only.

Per-step observation contract. Both see the same pre-step simulator state at tick t:
- policy: the P1.3 vector, 26 float32 values named by ManipulationEnv.metadata()["observation_names"]
  (6 joint positions, 6 joint velocities, block pose, grasp site, tray, time fraction), as reset()/step() return it;
- expert: Env::expert_action() on the same MjData (privileged: it reads the simulator, not the vector).
Both are queried BEFORE env.step(executed). episode.npz holds observation[t], the executed action and
observation[t+1]; corrective.npz holds the same transitions with the expert's action as the label.

Run layout (stable paths; every attempt is a fresh directory):
  manifest.json                              the ONLY commit record, replaced with storage.atomic_json
  iteration_000/seed.json                    init: the seed collection reference (paths, payload hashes, seeds)
  iteration_NNN/attempt_K/dataset/           Store: models/<id> (this iteration's checkpoint), episodes/<id> (rollouts)
  iteration_NNN/attempt_K/collection.json    rollout episode ids (P1.2 format, so export-lerobot reads it)
  iteration_NNN/attempt_K/report.json        training, rollout and held-out evaluation record
An iteration is complete iff manifest.json references it; any other attempt directory is incomplete and kept.
"""
import json
from pathlib import Path

import numpy as np

from xert.eval import ledger
from xle_workbench import pipeline
from xle_workbench.storage import Store, atomic_json, sha256

SOURCE = "dagger_policy_rollout"
SEED_SOURCE = "scripted_simulation_noise"  # P1.2 format: executed + corrective streams.
SPEC = Path(__file__).resolve().parents[1] / "spec/xlerobot_left.yaml"
EVAL_SEEDS = list(range(10000, 10030))  # The P1.3 / P5.3 held-out bench seeds.
RECOLLECT = "python -m scripts.collect_recovery --config configs/recovery_demos.json --output <fresh directory>"
NO_LEDGER = "No ledger row: scripted seed data and a scripted expert (P5.3 rule). This file is the record."


# ---------------------------------------------------------------- TODO(daniel): the DAgger loop's choices.

def dagger_action(policy_action, expert_action, iteration, step):
    """TODO(daniel): the mixing rule, i.e. which action the bench EXECUTES at this tick.

    Inputs: policy_action, expert_action: float32 (6,) normalized absolute joint targets in [-1, 1] (the
    ManipulationEnv action convention; index 5 is the gripper), both for the same pre-step state.
    iteration: 1-based DAgger iteration (the policy was trained on the seed data plus iterations < iteration).
    step: 0-based tick within the episode.
    Output: (6,) finite values in [-1, 1]. The plumbing executes it and stores it in episode.npz.
    Invariants: the expert label goes to corrective.npz whatever this returns (returning expert_action makes the
    two streams identical). Deterministic in its arguments: `iterate` calls it once with zero actions to refuse
    before training, and a stochastic rule would need the episode seed, which this signature does not pass.
    """
    raise NotImplementedError("DAgger mixing rule not chosen (TODO(daniel))")


def should_stop(history):
    """TODO(daniel): the stopping rule, asked before every iteration.

    Input: history = the report.json dicts of completed iterations, oldest first ([] before iteration 1). Each has
    "iteration", "aggregate_episodes", "training" {"settings", "episodes", "frames", "best_validation_mae"},
    "rollouts" {"seeds", "episode_ids", "successes"} and "evaluation" {"seeds", "trials", "successes", "n",
    "success_rate", "ci95"}.
    Output: bool. True makes `iterate` refuse; False runs one more iteration.
    Invariants: a rule that reads the held-out evaluation selects on the test seeds; pre-register it first.
    """
    raise NotImplementedError("DAgger stopping rule not chosen (TODO(daniel))")


def dagger_settings(iteration, aggregate):
    """TODO(daniel): the training hyperparameters and rollout count for one iteration.

    Input: iteration (1-based); aggregate = {"episodes": count, "frames": transitions} it will train on.
    Output: {"training": the P1.3 config shape (xle_workbench.learning.DEFAULTS keys: model_type, epochs,
    batch_size, chunk_size, hidden_dim, seed, val_fraction, lr, kl_weight, device, temporal_ensemble,
    ensemble_decay), "rollouts": int in [1, 100], bench episodes rolled out with the new policy}.
    Invariants: JSON-serializable (recorded in report.json); learning.validate_config accepts "training".
    The trainer has no data weighting: every aggregated transition counts once.
    """
    raise NotImplementedError("DAgger settings not chosen (TODO(daniel))")


# ---------------------------------------------------------------- Plumbing.

def _bench(seed):
    """The P5.3 evaluation bench. Its env also runs the DAgger rollouts, so one contract covers both."""
    from xert.backends.base import task_backend
    from xert.spec import load_spec
    spec = load_spec(SPEC)
    return spec, task_backend(spec, seed, "pick_place")


def _bounded(action):
    action = np.asarray(action, np.float32)
    if action.shape != (6,) or not np.isfinite(action).all() or np.abs(action).max() > 1:
        raise ValueError("dagger_action must return 6 finite normalized targets in [-1, 1]")
    return action


def next_seeds(used, held_out, count):
    """The `count` lowest seeds above every used seed, skipping held-out ones (train_model needs distinct seeds)."""
    seeds, seed = [], max(used) + 1
    while len(seeds) < count:
        if seed not in held_out:
            seeds.append(seed)
        seed += 1
    return seeds


class _Relabel:
    """pipeline.rollout's policy slot, like P1.2's _NoisyExpert: label at the actual state, then execute the mix."""

    def __init__(self, env, policy, iteration):
        self.env, self.policy, self.iteration = env, policy, iteration

    def reset(self):
        self.policy.reset()
        self.labels = []

    def act(self, observation):
        expert = np.asarray(self.env.expert_action(), np.float32)  # Before env.step: the label for this state.
        self.labels.append(expert.copy())
        policy_action = np.asarray(self.policy.act(observation), np.float32)
        return _bounded(dagger_action(policy_action, expert.copy(), self.iteration, len(self.labels) - 1))


def collect_rollouts(store, env, policy, seeds, provenance):
    """Roll out the mixed policy; episode.npz = executed actions, corrective.npz = expert labels."""
    records = []
    for seed in seeds:
        actor = _Relabel(env, policy, provenance["iteration"])
        arrays, outcome = pipeline.rollout(env, seed, actor)
        records.append(store.save_episode(arrays, {
            "source": SOURCE, "seed": seed, "config": env.config, "simulation": env.metadata(),
            "contract_sha256": pipeline.contract(env.metadata()), "success": bool(outcome.get("success", False)),
            "outcome": outcome, "hardware": False, "dagger": provenance,
        }, corrective_actions=np.asarray(actor.labels)))
    return records


def _reference(store_path, record):
    return {"id": record["id"], "store": str(store_path), "seed": record["seed"], "steps": record["steps"],
            "sha256": record["sha256"], "corrective_sha256": record["corrective_sha256"]}


class _Aggregate:
    """The two Store methods train_model uses, over committed episodes in several Stores (no copies).

    train_model still runs every check (contract, duplicate payloads, distinct seeds); never learning.train directly.
    """

    def __init__(self, run, references, models):
        self.run, self.references, self.models = run, {ref["id"]: ref for ref in references}, models

    def load_training_episode(self, identifier):
        ref = self.references[identifier]
        arrays, metadata = Store(self.run / ref["store"]).load_training_episode(identifier)
        if (metadata["sha256"], metadata.get("corrective_sha256")) != (ref["sha256"], ref["corrective_sha256"]):
            raise ValueError(f"Episode {identifier} differs from the run manifest")
        return arrays, metadata

    def create(self, kind):
        return self.models.create(kind)


def init(collection, output, eval_seeds=EVAL_SEEDS):
    """Validate a P1.2 collection against the current core and write manifest.json + iteration_000/."""
    collection, output = Path(collection).resolve(), Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    eval_seeds = list(eval_seeds)
    if not eval_seeds or len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("Choose distinct held-out evaluation seeds")
    if not (collection / "dataset/episodes").is_dir():
        raise ValueError(f"{collection} is not a P1.2 collection (collection.json beside dataset/)")
    ids = json.loads((collection / "collection.json").read_text("utf-8"))["episode_ids"]
    store = Store(collection / "dataset")
    _, backend = _bench(eval_seeds[0])
    env, current = backend.env, pipeline.contract(backend.env.metadata())
    records = [store.read("episodes", identifier) for identifier in ids]
    for record in records:  # Metadata first: refuse a stale collection before decoding any payload.
        if (record.get("contract_sha256") != current
                or pipeline.contract(record.get("simulation", {})) != record.get("contract_sha256")):
            raise ValueError(
                f"Seed episode {record.get('id')} was recorded under contract {record.get('contract_sha256')} "
                f"(implementation_sha256 {record.get('simulation', {}).get('implementation_sha256')}); the current "
                f"core is {current} (implementation_sha256 {env.implementation_sha256}). "
                f"Recollect on the current core: {RECOLLECT}")
        config = record.get("config") or {}
        differing = sorted(key for key in set(config) | set(env.config) if config.get(key) != env.config.get(key))
        if differing:  # The contract omits friction, delay and bounds; rollouts and evaluation use the bench config.
            raise ValueError(f"Seed episode {record.get('id')} config differs from the bench in {differing}; "
                             f"recollect with the bench config: {RECOLLECT}")
        if record.get("source") != SEED_SOURCE:
            raise ValueError(f"Seed episode {record.get('id')} is {record.get('source')!r}, not {SEED_SOURCE!r}")
    for identifier in ids:
        store.load_training_episode(identifier)  # Payload hashes, corrective labels and their alignment.
    seeds = [record["seed"] for record in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed episodes must have distinct bench seeds")
    if set(eval_seeds) & set(seeds):
        raise ValueError("Evaluation seeds overlap demonstration bench seeds; held-out states required")
    references = [_reference(store.root, record) for record in records]
    output.mkdir(parents=True)
    atomic_json(output / "iteration_000/seed.json", {
        "collection": str(collection), "collection_sha256": sha256(collection / "collection.json"),
        "source": SEED_SOURCE, "contract_sha256": current, "episodes": references})
    manifest = {"schema_version": 1, "kind": "xert_dagger_run", "task_id": "pick_place", "robot": "xlerobot_left",
                "contract_sha256": current, "evaluation_seeds": eval_seeds, "hardware": False,
                "iterations": [{"iteration": 0, "attempt": 0, "path": "iteration_000",
                                "seed_reference": "iteration_000/seed.json", "episodes": references}]}
    atomic_json(output / "manifest.json", manifest)
    return manifest


def iterate(run, retry=False):
    """One DAgger iteration: train, roll out with relabelling, stage, evaluate, report, then commit the manifest."""
    from xert.eval import gate_experiment
    from xle_workbench.learning import validate_config
    run = Path(run).absolute()
    manifest = json.loads((run / "manifest.json").read_text("utf-8"))
    iteration = len(manifest["iterations"])
    folder = run / f"iteration_{iteration:03d}"
    attempts = sorted(int(path.name.split("_")[1]) for path in folder.glob("attempt_*"))
    if attempts and not retry:
        raise ValueError(f"Iteration {iteration} has incomplete attempts {attempts} that the manifest does not "
                         "reference; rerun with --retry (a new attempt; nothing is deleted)")
    if retry and not attempts:
        raise ValueError(f"Nothing to retry: iteration {iteration} has no incomplete attempt")
    _bounded(dagger_action(np.zeros(6, np.float32), np.zeros(6, np.float32), iteration, 0))  # Refuse before training.
    history = []
    for entry in manifest["iterations"][1:]:
        if sha256(run / entry["report"]) != entry["report_sha256"]:
            raise ValueError(f"{entry['report']} differs from the run manifest")
        history.append(json.loads((run / entry["report"]).read_text("utf-8")))
    if should_stop(history):
        raise ValueError(f"should_stop ended the run after iteration {iteration - 1}")
    references = [ref for entry in manifest["iterations"] for ref in entry["episodes"]]
    aggregate = {"episodes": len(references), "frames": sum(ref["steps"] for ref in references)}
    settings = dagger_settings(iteration, aggregate)
    rollouts = settings.get("rollouts") if isinstance(settings, dict) else None
    if set(settings or {}) != {"training", "rollouts"} or type(rollouts) is not int or not 1 <= rollouts <= 100:
        raise ValueError("dagger_settings must return {'training': ..., 'rollouts': int in [1, 100]}")
    validate_config(settings["training"])
    eval_seeds = manifest["evaluation_seeds"]
    spec, backend = _bench(eval_seeds[0])
    env = backend.env
    if pipeline.contract(env.metadata()) != manifest["contract_sha256"]:
        raise ValueError(f"The core changed since init; start a new run from a collection recollected with {RECOLLECT}")
    attempt = (attempts[-1] if attempts else 0) + 1
    path = folder / f"attempt_{attempt}"
    path.mkdir(parents=True)  # Fresh per attempt: new Store UUIDs and an empty checkpoint dir, never a collision.
    relative = path.relative_to(run).as_posix()
    store = Store(path / "dataset")
    # 1. Train on the aggregate with the existing trainer and its checks.
    trained = pipeline.train_model(_Aggregate(run, references, store), [ref["id"] for ref in references],
                                   settings["training"])
    policy, proof = pipeline.load_verified_policy(store, trained["id"], env)  # Contract of the evaluation bench.
    if set(eval_seeds) & set(proof["training_source_seeds"]):
        raise ValueError("Evaluation seeds overlap demonstration bench seeds; held-out states required")
    # 2-3. Roll out with relabelling and stage the episodes with provenance.
    seeds = next_seeds([ref["seed"] for ref in references], set(eval_seeds), rollouts)
    model_path = f"{relative}/dataset/models/{proof['id']}"
    provenance = {"iteration": iteration, "attempt": attempt, "seeds": seeds, "policy_model_id": proof["id"],
                  "policy_model_sha256": proof["model_sha256"], "policy_model_path": model_path,
                  "expert": "Env::expert_action (native)", "expert_implementation_sha256": env.implementation_sha256,
                  "mixing_rule": f"{dagger_action.__module__}.{dagger_action.__qualname__}"}
    records = collect_rollouts(store, env, policy, seeds, provenance)
    atomic_json(path / "collection.json", {"episode_ids": [record["id"] for record in records], "source": SOURCE,
                                           "iteration": iteration, "attempt": attempt})
    # 4. Held-out evaluation with the P5.3 per-seed rollout and Wilson interval.
    names = env.metadata()["observation_names"]
    trials = [gate_experiment.rollout(policy, spec, seed, names) for seed in eval_seeds]
    successes = sum(trial["success"] for trial in trials)
    report = {"iteration": iteration, "attempt": attempt, "path": relative,
              "model": {"id": proof["id"], "path": model_path, "model_sha256": proof["model_sha256"],
                        "metadata_sha256": proof["metadata_sha256"],
                        "training_action_sources": proof["training_action_sources"]},
              "training": {"settings": settings["training"], **aggregate,
                           "best_validation_mae": proof["metrics"]["best_validation_mae"]},
              "rollouts": {"seeds": seeds, "episode_ids": [record["id"] for record in records],
                           "successes": sum(record["success"] for record in records),
                           "note": "Mixed policy/expert execution; not a policy evaluation."},
              **{key: provenance[key] for key in ("expert", "expert_implementation_sha256", "mixing_rule")},
              "evaluation": {"harness": "xert.eval.gate_experiment.rollout (P5.3)", "seeds": eval_seeds,
                             "trials": trials, "successes": successes, "n": len(trials),
                             "success_rate": successes / len(trials), "ci95": pipeline.wilson(successes, len(trials))},
              "aggregate_episodes": len(references) + len(records), "incomplete_attempts": attempts,
              "ledger": NO_LEDGER, "hardware": False, **ledger.source_provenance()}
    atomic_json(path / "report.json", report)
    entry = {"iteration": iteration, "attempt": attempt, "path": relative, "model": model_path,
             "collection": f"{relative}/collection.json", "report": f"{relative}/report.json",
             "report_sha256": sha256(path / "report.json"),
             "incomplete_attempts": [f"{folder.name}/attempt_{number}" for number in attempts],
             "episodes": [_reference(f"{relative}/dataset", record) for record in records]}
    atomic_json(run / "manifest.json", {**manifest, "iterations": [*manifest["iterations"], entry]})  # The commit.
    return report


def add_command(commands):
    parser = commands.add_parser("dagger", help="DAgger plumbing on the pick-and-place bench (rules: TODO(daniel))")
    operations = parser.add_subparsers(dest="operation", required=True)
    start = operations.add_parser("init", help="Validate a P1.2 seed collection and start a fresh run")
    start.add_argument("--collection", type=Path, required=True,
                       help="P1.2 collection directory (collection.json + dataset/) recollected on the current core")
    start.add_argument("--output", type=Path, required=True, help="Fresh run directory")
    start.add_argument("--eval-seeds", type=int, nargs="+", default=EVAL_SEEDS)
    step = operations.add_parser("iterate", help="Run one iteration (refuses until the TODO(daniel) rules exist)")
    step.add_argument("--run", type=Path, required=True)
    step.add_argument("--retry", action="store_true", help="New attempt after an incomplete one; deletes nothing")


def execute(args):
    if args.operation == "init":
        manifest = init(args.collection, args.output, args.eval_seeds)
        print(json.dumps({"run": str(args.output), "seed_episodes": len(manifest["iterations"][0]["episodes"])}))
        return
    report = iterate(args.run, retry=args.retry)
    print(json.dumps({"iteration": report["iteration"], "attempt": report["attempt"],
                      "aggregate_episodes": report["aggregate_episodes"],
                      **{key: report["evaluation"][key] for key in ("successes", "n", "ci95")}}))
