"""P5.3: train ACT on all (A), gated (B) and random same-size (C) human demos; report from the ledger.

Run in the LeRobot environment once P4.2 sessions exist and P5.2 has committed configs/gate.yaml:
python -m xert.eval.gate_experiment --sessions <session dir> [...] --output results/p5.3/gate_v1
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from xert.data.quality import HUMAN_SOURCES, verdict
from xert.eval import ledger

MIN_TRAIN_SEEDS, MIN_EVAL_EPISODES = 3, 30
CONDITIONS = ("A", "B", "C")
LABELS = {"A": "all demos", "B": "gate-passing demos", "C": "random subset of A, same size as B"}
RULE = ("Pre-stated rule: B differs measurably from a comparator only if their pooled Wilson 95% intervals "
        "do not overlap. B's interval entirely above means the gate helped, entirely below means it hurt, "
        "and any overlap means no measurable difference.")


def preregistered(path):
    """Return (sha256, commit, bytes) of a file committed in THIS repo whose disk bytes are that commit's blob."""
    path, root = Path(path).resolve(), Path(ledger.REPO_ROOT).resolve()
    if not path.is_file():
        raise ValueError(f"No {path.name} at {path}; P5.2 thresholds and the experiment config must be committed first")
    if not path.is_relative_to(root):
        raise ValueError(f"{path} is outside the repository {root}; its commit would prove nothing here")
    relative = path.relative_to(root).as_posix()
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

    def git(*args, data=None):
        return subprocess.run(["git", "--no-replace-objects", "-C", str(root), *args], input=data, capture_output=True,
                              **options).stdout.decode().strip()

    data = path.read_bytes()  # Hash, compare and parse exactly these bytes.
    commit = git("log", "-1", "--format=%H", "--", relative)  # The commit that introduced the committed bytes.
    if not commit:
        raise ValueError(f"{relative} is not committed; pre-registration requires a commit")
    if git("status", "--porcelain", "--", relative):
        raise ValueError(f"{relative} has uncommitted changes; changed settings are a new experiment")
    # Catches assume-unchanged/skip-worktree edits that status hides; hash-object applies the same eol filters.
    if git("hash-object", "--stdin", f"--path={relative}", data=data) != git("rev-parse", f"{commit}:{relative}"):
        raise ValueError(f"{relative} on disk differs from commit {commit}")
    return hashlib.sha256(data).hexdigest(), commit, data


def conditions(passed, train_seed, subset_seed):
    """Episode indices per condition. C is drawn per training seed, so it is deterministic but not one lucky draw."""
    every = list(range(len(passed)))
    gated = [index for index in every if passed[index]]
    rng = np.random.default_rng([subset_seed, train_seed])
    return {"A": every, "B": gated, "C": sorted(rng.choice(every, len(gated), replace=False).tolist())}


def load_recording(path):
    """Policy inputs are every `observation.*` feature the dataset declares; the output is `action`."""
    from xert.data.recorder import load_runtime_dataset
    dataset = load_runtime_dataset(path)
    try:
        features = dataset.features
        keys = [key for key in features if key.startswith("observation.")]
        if any(features[key]["dtype"] in ("image", "video") for key in keys):
            raise ValueError("Image observations are not supported by the state-only ACT path")
        batch = dataset.hf_dataset[:]
        columns = {key: np.stack([np.asarray(value) for value in batch[key]])
                   for key in (*keys, "action", "frame_index", "episode_index")}
        names = [name for key in keys for name in features[key]["names"]]
        action_names = list(features["action"]["names"])
    finally:
        del dataset
        gc.collect()  # Windows keeps Parquet files locked while the reader lives.
    if len(np.unique(columns["episode_index"])) != 1:
        raise ValueError(f"{path} must hold exactly one episode")
    order = np.argsort(columns["frame_index"])
    observation = np.concatenate([columns[key][order].reshape(len(order), -1) for key in keys], axis=1)
    if observation.shape[1] != len(names):
        raise ValueError(f"{path}: observation feature names do not match their widths")
    return names, action_names, observation.astype(np.float32), columns["action"][order].astype(np.float64)


def bench_observation(backend, names, native_names):
    """Build the policy input by feature name from bench state: `<joint>.pos/.vel`, or a native observation name."""
    state = backend.read_state()
    values = dict(zip(native_names, backend.env._obs()))  # ponytail: private accessor; the native env has no public one.
    values.update({f"{joint}.pos": q for joint, q in zip(backend.spec.joint_names, state.q)})
    values.update({f"{joint}.vel": qd for joint, qd in zip(backend.spec.joint_names, state.qd)})
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"The bench cannot provide observation features {missing}")
    return np.asarray([values[name] for name in names], dtype=np.float32)


def rollout(policy, spec, seed, names):
    from xert.backends.bench import BenchBackend
    backend = BenchBackend(spec, seed)
    native_names = backend.env.metadata()["observation_names"]
    low, high = backend.env.limits.T
    policy.reset()
    while not backend.finished:
        action = policy.act(bench_observation(backend, names, native_names))  # Clipped to [-1, 1].
        backend.send_targets(low + (action + 1) / 2 * (high - low), None)
        if backend.info["success"]:
            break
    return {"seed": seed, "success": bool(backend.info["success"]), "steps": backend.info["steps"],
            "failure_reason": backend.info["failure_reason"]}


def run(config_path, sessions, output, report_path):
    from xert.data.quality import dataset_metrics
    from xert.spec import load_spec
    from xle_workbench.learning import load_policy, train
    from xle_workbench.pipeline import wilson
    config_sha256, config_commit, config_bytes = preregistered(config_path)
    config = json.loads(config_bytes)
    train_seeds, eval_seeds = config["training_seeds"], config["evaluation_seeds"]
    if (len(set(train_seeds)) < MIN_TRAIN_SEEDS or len(set(eval_seeds)) < MIN_EVAL_EPISODES
            or len(set(train_seeds)) != len(train_seeds) or len(set(eval_seeds)) != len(eval_seeds)):
        raise ValueError(f"Require >= {MIN_TRAIN_SEEDS} distinct training seeds and >= {MIN_EVAL_EPISODES} eval seeds")
    if Path(output).exists() or Path(report_path).exists():
        raise ValueError("Output or report already exists; every run is fresh evidence")
    tag = "p5.3_" + Path(output).name
    if any(row["experiment"] == tag for row in ledger.read_rows()):
        raise ValueError(f"Ledger already has rows tagged {tag}; choose a new output name")
    gate_file = Path(ledger.REPO_ROOT) / config["gate_path"]  # Relative paths mean this repo, not the cwd.
    gate_sha256, gate_commit, gate_bytes = preregistered(gate_file)
    provenance = ledger.source_provenance()
    if provenance["git_dirty"]:
        raise ValueError("Repository has uncommitted changes; commit them so every row names the exact source")
    try:  # JSON-compatible YAML subset, like the embodiment specs: {metric: {"min": x, "max": y}}.
        thresholds = json.loads(gate_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("gate.yaml uses the JSON-compatible YAML subset (quoted keys, no comments)") from exc
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("gate.yaml must map metric names to {min, max} bounds")
    spec_bytes = Path(config["spec_path"]).read_bytes()
    spec = load_spec(config["spec_path"], contents=spec_bytes)
    recordings = [path for session in sessions for path in sorted(Path(session).glob("episode_*"))]
    if not recordings:
        raise ValueError("No episode_* recordings in the given sessions")
    episodes, verdicts, bench_seeds, kinds = [], [], set(), set()
    for path in recordings:
        manifest = json.loads((path / "xert_recording.json").read_text(encoding="utf-8"))
        kind = manifest["source"]["kind"]
        if kind not in HUMAN_SOURCES:
            raise ValueError(f"Refusing {kind!r} data in {path}; only {HUMAN_SOURCES} count as demonstrations")
        if (manifest["status"] != "complete" or manifest["backend"] != "mujoco" or manifest["embodiment"] != spec.name
                or manifest["spec_sha256"] != hashlib.sha256(spec_bytes).hexdigest()):
            raise ValueError(f"{path} is incomplete or was not recorded on this bench spec")
        [metrics], sources = dataset_metrics(path)
        if sources - {kind}:
            raise ValueError(f"Refusing {path}: frame sources {sorted(sources - {kind})} conflict with manifest {kind!r}")
        passed, reasons = verdict(metrics, thresholds)
        names, action_names, observation, action = load_recording(path)
        if episodes and (names, action_names) != (episodes[0]["names"], episodes[0]["action_names"]):
            raise ValueError(f"{path} declares different features from {recordings[0]}")
        if action_names != [f"{joint}.pos" for joint in spec.joint_names]:
            raise ValueError("Recorded actions must be absolute bench joint targets in spec order")
        low, high = spec.position_limits.T
        # ponytail: terminal row is padding; train() never reads it (it has no action target).
        episodes.append({"names": names, "action_names": action_names,
                         "observations": np.vstack([observation, observation[-1:]]),
                         "actions": (2 * (action - low) / (high - low) - 1).astype(np.float32)})
        verdicts.append({"recording": str(path), "passed": passed, "reasons": reasons, "metrics": metrics})
        bench_seeds.add(manifest["source"]["bench_seed"])
        kinds.add(kind)
    if set(eval_seeds) & bench_seeds:
        raise ValueError("Evaluation seeds overlap demonstration bench seeds; held-out states required")
    passed = [row["passed"] for row in verdicts]
    if not 2 <= sum(passed) < len(passed):
        raise ValueError(f"The gate kept {sum(passed)} of {len(passed)} episodes; "
                         "B needs >= 2 episodes and must differ from A")
    names = episodes[0]["names"]
    from xert.backends.bench import BenchBackend
    probe = BenchBackend(spec, eval_seeds[0])
    bench_observation(probe, names, probe.env.metadata()["observation_names"])  # Refuse before any training.
    output = Path(output)
    output.mkdir(parents=True)
    gate = {"gate_path": config["gate_path"], "gate_sha256": gate_sha256, "gate_commit": gate_commit,
            "experiment_config_sha256": config_sha256, "experiment_config_commit": config_commit}
    common_paths = [output / "config.json", output / "verdicts.json", gate_file]
    (output / "config.json").write_text(json.dumps({
        **config, **provenance, **gate,
        "sessions": [str(s) for s in sessions], "observation_names": names,
        "action_names": episodes[0]["action_names"]}, indent=2), encoding="utf-8")
    (output / "verdicts.json").write_text(json.dumps(verdicts, indent=2, allow_nan=False), encoding="utf-8")
    import torch
    torch.set_num_threads(1)
    for seed in train_seeds:
        split = conditions(passed, seed, config["control_subset_seed"])
        for condition in CONDITIONS:
            model_dir = output / condition / f"seed_{seed}"
            train([{key: episodes[index][key] for key in ("observations", "actions")} for index in split[condition]],
                  model_dir,
                  {**config["training"], "seed": seed, "observation_names": names,
                   "action_names": episodes[0]["action_names"]},
                  progress=lambda record: print(json.dumps({"condition": condition, "seed": seed, **record}), flush=True))
            policy = load_policy(model_dir)
            trials = [rollout(policy, spec, eval_seed, names) for eval_seed in eval_seeds]
            successes = sum(trial["success"] for trial in trials)
            evaluation = model_dir / "evaluation.json"
            evaluation.write_text(json.dumps({"condition": condition, "train_seed": seed, "trials": trials,
                                              "episodes": [verdicts[i]["recording"] for i in split[condition]]},
                                             indent=2), encoding="utf-8")
            artifacts = [model_dir / "model.pt", model_dir / "metadata.json", evaluation, *common_paths]
            row = ledger.new_row(
                **provenance, slice="P5.3", experiment=tag, backend="mujoco", embodiment=spec.name,
                metric="success_rate", value=successes / len(trials), ci95=wilson(successes, len(trials)),
                n=len(trials), seeds={"train": [seed], "eval": list(eval_seeds)},
                config_path=ledger.artifact_path(output / "config.json"),
                artifact_paths=[ledger.artifact_path(path) for path in artifacts],
                artifact_sha256={ledger.artifact_path(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in artifacts},
                condition=condition, successes=successes, train_episodes=len(split[condition]),
                train_frames=sum(len(episodes[index]["actions"]) for index in split[condition]),
                source_kinds=sorted(kinds), **gate,
                notes=f"Condition {condition} ({LABELS[condition]}): XERT ACT-style policy on human teleop demos, "
                      "rolled out on held-out MuJoCo bench seeds. Simulation only; no hardware claim.")
            ledger.append_row(row)
            print(json.dumps(row), flush=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(report_path).open("x", encoding="utf-8") as stream:
        stream.write(report(ledger.read_rows(), tag))


def compare(b, other):
    """The pre-stated rule on two Wilson intervals."""
    if b[0] > other[1]:
        return "helped"
    if b[1] < other[0]:
        return "hurt"
    return "made no measurable difference"


def report(rows, experiment):
    """Markdown built only from ledger rows of one experiment tag."""
    from xle_workbench.pipeline import wilson
    rows = [row for row in rows if row["slice"] == "P5.3" and row["experiment"] == experiment]
    by = {(row["condition"], row["seeds"]["train"][0]): row for row in rows}
    seeds = sorted({seed for _, seed in by})
    if len(by) != len(rows) or set(by) != {(c, s) for c in CONDITIONS for s in seeds} or not seeds:
        raise ValueError(f"Experiment {experiment} lacks exactly one row per condition and training seed")
    if len(seeds) < MIN_TRAIN_SEEDS:
        raise ValueError(f"Experiment {experiment} has fewer than {MIN_TRAIN_SEEDS} training seeds")
    if len({(row["gate_sha256"], row["gate_commit"], row["experiment_config_sha256"], row["experiment_config_commit"],
             tuple(row["seeds"]["eval"])) for row in rows}) != 1:
        raise ValueError("Rows disagree on gate, experiment config or evaluation seeds")
    for row in rows:
        if (row["n"] < MIN_EVAL_EPISODES or row["n"] != len(row["seeds"]["eval"])
                or row["value"] != row["successes"] / row["n"] or row["ci95"] != wilson(row["successes"], row["n"])):
            raise ValueError(f"Row {row['id']} has too few trials, or its count, rate and interval disagree")
        if not row["source_kinds"] or set(row["source_kinds"]) - set(HUMAN_SOURCES):
            raise ValueError(f"Row {row['id']} was not trained only on human demonstrations")
    first = rows[0]
    lines = [f"# P5.3 quality-gate experiment ({experiment})", "",
             f"Generated from `{ledger.artifact_path(ledger.ledger_path())}` rows; "
             "MuJoCo bench (`xlerobot_left`) simulation only.", "",
             f"Gate `{first['gate_path']}` sha256 `{first['gate_sha256']}`, committed in `{first['gate_commit']}`. "
             f"Experiment config sha256 `{first['experiment_config_sha256']}`, committed in "
             f"`{first['experiment_config_commit']}`. Same {first['n']} held-out bench seeds for every model.", "",
             RULE, "", "C matches B in demo count, not in frames or gradient steps; a gate on frames or duration "
             "changes B's training volume relative to C (see Train frames).", "",
             "| Condition | Training seed | Train episodes | Train frames | Successes / trials | Wilson 95% | Ledger row |",
             "|---|---|---|---|---|---|---|"]
    pooled = {}
    for condition in CONDITIONS:
        for seed in seeds:
            row = by[condition, seed]
            lines.append(f"| {condition} | {seed} | {row['train_episodes']} | {row['train_frames']} | "
                         f"{row['successes']} / {row['n']} | "
                         f"[{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] | {row['id']} |")
        wins, count = (sum(by[condition, seed][key] for seed in seeds) for key in ("successes", "n"))
        pooled[condition] = (wins / count, wilson(wins, count), wins, count)
    lines += ["", "| Condition | Pooled successes / trials | Rate | Wilson 95% |", "|---|---|---|---|"]
    for condition in CONDITIONS:
        rate, ci, wins, count = pooled[condition]
        lines.append(f"| {condition}: {LABELS[condition]} | {wins} / {count} | {rate:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    versus_a, versus_c = (compare(pooled["B"][1], pooled[other][1]) for other in ("A", "C"))
    lines += ["", f"- B - A = {pooled['B'][0] - pooled['A'][0]:+.4f}: the gate {versus_a} relative to all demos.",
              f"- B - C = {pooled['B'][0] - pooled['C'][0]:+.4f}: the gate {versus_c} relative to a random "
              "same-size subset."]
    if versus_a != "made no measurable difference":
        lines.append("- " + ("The same-size control agrees, so the effect follows which demos were kept, not only "
                             "how many." if versus_c == versus_a else
                             "The same-size control does not agree, so this does not separate cleaner data from less data."))
    lines += ["", "Pooled intervals count every rollout as independent although models share evaluation states, "
              "and ignores variation between training seeds; treat them as approximate. This is not a hardware result and does not tune any threshold."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/gate_experiment.json"))
    parser.add_argument("--sessions", type=Path, nargs="+", required=True, help="P4.2 session directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("results/gate_experiment.md"))
    args = parser.parse_args()
    run(args.config, args.sessions, args.output, args.report)
