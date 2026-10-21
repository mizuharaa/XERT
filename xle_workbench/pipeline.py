"""The same collect/train/evaluate functions serve the CLI and HTTP jobs."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from xert.eval import ledger

from .storage import Store, atomic_json, sha256


def contract(metadata: dict) -> str:
    fields = {key: metadata.get(key) for key in
              ("observation_dim", "action_dim", "observation_names", "action_names",
               "control_hz", "horizon", "model_sha256", "action_convention",
               "implementation_sha256", "mujoco_version")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def check_cancel(cancelled) -> None:
    if cancelled and cancelled():
        raise InterruptedError("Cancelled by user")


def initial_state_hash(observation) -> str:
    return hashlib.sha256(np.asarray(observation, dtype="<f4").tobytes()).hexdigest()


def rollout(env, seed: int, policy=None, cancelled=None) -> tuple[dict, dict]:
    observation = env.reset(seed=seed)
    if policy is not None:
        policy.reset()
    observations, actions, rewards, terminals = [observation.copy()], [], [], []
    info = {}
    for _ in range(env.config["horizon"]):
        check_cancel(cancelled)
        action = np.asarray(env.expert_action() if policy is None else policy.act(observation), dtype=np.float32)
        observation, reward, terminated, info = env.step(action)
        actions.append(np.asarray(action, dtype=np.float32).copy())
        observations.append(observation.copy())
        rewards.append(reward)
        terminals.append(terminated)
        if terminated:
            break
    arrays = {"observations": np.asarray(observations, dtype=np.float32),
              "actions": np.asarray(actions, dtype=np.float32),
              "rewards": np.asarray(rewards, dtype=np.float32),
              "terminated": np.asarray(terminals, dtype=bool)}
    return arrays, info


def collect(store: Store, seeds: list[int], config: dict | None = None,
            progress=None, cancelled=None) -> dict:
    from .sim import ManipulationEnv
    if not seeds or len(seeds) > 100 or len(set(seeds)) != len(seeds):
        raise ValueError("Choose 1..100 unique collection seeds")
    records = []
    env = ManipulationEnv(config)
    try:
        for index, seed in enumerate(seeds):
            arrays, outcome = rollout(env, seed, cancelled=cancelled)
            record = store.save_episode(arrays, {
                "source": "scripted_simulation", "seed": seed, "config": env.config,
                "simulation": env.metadata(), "contract_sha256": contract(env.metadata()),
                "success": bool(outcome.get("success", False)), "outcome": outcome,
                "hardware": False,
            })
            records.append(record)
            if progress:
                progress({"completed": index + 1, "total": len(seeds),
                          "episode_id": record["id"], "success": record["success"]})
    finally:
        env.close()
    return {"episode_ids": [item["id"] for item in records], "episodes": len(records),
            "successes": sum(item["success"] for item in records),
            "source": "scripted_simulation", "hardware": False}


def train_model(store: Store, episode_ids: list[str], config: dict | None = None,
                progress=None, cancelled=None) -> dict:
    from .learning import train
    if len(episode_ids) < 2 or len(episode_ids) > 1000 or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Training requires 2..1000 distinct episodes for an episode-level validation split")
    arrays, records = [], []
    for identifier in episode_ids:
        check_cancel(cancelled)
        data, metadata = store.load_episode(identifier)
        arrays.append(data)
        records.append(metadata)
        if metadata.get("contract_sha256") != contract(metadata.get("simulation", {})):
            raise ValueError("Episode simulation metadata no longer matches its contract hash")
    signatures = {record.get("contract_sha256") for record in records}
    if len(signatures) != 1 or None in signatures:
        raise ValueError("Training episodes must share a verified observation/action/model contract")
    if len({record["sha256"] for record in records}) != len(records):
        raise ValueError("Duplicate episode payloads would contaminate the validation split")
    identity = [record.get("seed") for record in records]
    if len(set(identity)) != len(identity):
        raise ValueError("Use distinct source/seed episodes to keep validation initial states independent")
    identifier, output = store.create("models")
    simulation = records[0]["simulation"]
    settings = dict(config or {})
    settings.update({"observation_names": simulation["observation_names"],
                     "action_names": simulation["action_names"]})
    try:
        metrics = train(arrays, output, settings, progress=progress, cancelled=cancelled)
        check_cancel(cancelled)
        proof = {"id": identifier, "state": "completed", "contract_sha256": signatures.pop(),
                 "simulation": simulation, "episode_ids": episode_ids,
                 "episode_sha256": [record["sha256"] for record in records],
                 "training_source_seeds": [record.get("seed") for record in records],
                 "demonstration_initial_states": [initial_state_hash(data["observations"][0]) for data in arrays],
                 "dataset_successes": sum(bool(record.get("success")) for record in records),
                 "dataset_episodes": len(records), "hardware": False,
                 "model_sha256": sha256(output / "model.pt"), "metrics": metrics}
        proof["metadata_sha256"] = sha256(output / "metadata.json")
        atomic_json(output / "pipeline.json", proof)
        return proof
    except BaseException:
        atomic_json(output / "pipeline.json", {"id": identifier, "state": "incomplete", "hardware": False})
        raise


def load_verified_policy(store: Store, identifier: str, env):
    from .learning import load_policy
    path = store.path("models", identifier)
    proof = json.loads((path / "pipeline.json").read_text("utf-8"))
    if proof.get("state") != "completed":
        raise ValueError("Checkpoint training did not complete")
    if proof["contract_sha256"] != contract(env.metadata()):
        raise ValueError("Checkpoint observation/action/model contract does not match this environment")
    if sha256(path / "model.pt") != proof["model_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    if sha256(path / "metadata.json") != proof["metadata_sha256"]:
        raise ValueError("Checkpoint metadata checksum mismatch")
    return load_policy(path), proof


def wilson(successes: int, count: int) -> list[float]:
    z = 1.96
    p = successes / count
    denominator = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [0.0 if successes == 0 else max(0.0, centre - margin),
            1.0 if successes == count else min(1.0, centre + margin)]


def evaluate(store: Store, seeds: list[int], config: dict | None = None,
             model_id: str | None = None, progress=None, cancelled=None, *,
             experiment: str | None = None) -> dict:
    from .sim import ManipulationEnv
    if not seeds or len(seeds) > 100 or len(set(seeds)) != len(seeds):
        raise ValueError("Choose 1..100 unique evaluation seeds")
    provenance = ledger.source_provenance()
    env = ManipulationEnv(config)
    rows = []
    started = time.perf_counter()
    try:
        policy, proof = (None, None) if model_id is None else load_verified_policy(store, model_id, env)
        if proof and set(seeds).intersection(proof["training_source_seeds"]):
            raise ValueError("Evaluation seeds overlap the demonstration dataset")
        for index, seed in enumerate(seeds):
            arrays, info = rollout(env, seed, policy, cancelled)
            record = store.save_episode(arrays, {
                "source": "policy_evaluation" if model_id else "expert_evaluation",
                "seed": seed, "config": env.config, "simulation": env.metadata(),
                "contract_sha256": contract(env.metadata()), "model_id": model_id,
                "success": bool(info.get("success", False)), "outcome": info, "hardware": False,
            })
            rows.append({"seed": seed, "episode_id": record["id"], "steps": record["steps"],
                         "success": record["success"], "outcome": info,
                         "initial_state_sha256": initial_state_hash(arrays["observations"][0])})
            if progress:
                progress({"completed": index + 1, "total": len(seeds), **rows[-1]})
        successes = sum(row["success"] for row in rows)
        initial_states = {row["initial_state_sha256"] for row in rows}
        disjoint = (initial_states.isdisjoint(proof["demonstration_initial_states"])
                    if proof and "demonstration_initial_states" in proof else None)
        identifier, path = store.create("evaluations")
        report = {"id": identifier, "model_id": model_id, "policy": "learned" if model_id else "scripted",
                  "seeds": seeds, "config": env.config, "simulation": env.metadata(),
                  "episodes": rows, "successes": successes, "trials": len(rows),
                  "success_rate": successes / len(rows),
                  "unique_initial_states": len(initial_states),
                  "initial_states_disjoint_from_demonstrations": disjoint,
                  "held_out": disjoint,
                  "success_rate_ci95": (wilson(successes, len(rows))
                                        if len(initial_states) == len(rows) and (proof is None or disjoint is True)
                                        else None),
                  "interval_note": "Wilson binomial interval; omitted for repeated states or unverified demonstration disjointness. Simulator coverage is not hardware confidence.",
                  "failure_counts": dict(Counter(str(row["outcome"].get("failure_reason", "unsuccessful"))
                                                 for row in rows if not row["success"])),
                  "elapsed_seconds": time.perf_counter() - started,
                  "runtime": {"python": platform.python_version(), "platform": platform.platform()},
                  "hardware": False, "safety_certification": False}
        train_seeds = [policy.metadata["config"]["seed"]] if policy is not None else []
        artifacts = [path / "metadata.json"]
        artifacts.extend(store.path("episodes", episode["episode_id"]) / name
                         for episode in rows for name in ("metadata.json", "episode.npz"))
        if model_id:
            artifacts.extend(store.path("models", model_id) / name
                             for name in ("pipeline.json", "metadata.json", "model.pt"))
        row = ledger.new_row(
            **provenance, slice="P0.2", experiment=experiment or f"evaluation_{identifier}",
            backend="mujoco", embodiment="xlerobot_left", metric="success_rate",
            value=report["success_rate"], ci95=report["success_rate_ci95"], n=report["trials"],
            seeds={"train": train_seeds, "eval": list(seeds)},
            config_path=ledger.artifact_path(path / "config.json"),
            artifact_paths=[ledger.artifact_path(item) for item in artifacts],
            notes=("Simulation only; pinned XLeRobot left arm (asset declares so_arm100), not verified SO-101. "
                   + report["interval_note"]),
            evaluation_id=identifier, model_id=model_id,
            demonstration_seeds=proof["training_source_seeds"] if proof else [])
        report["ledger_id"] = row["id"]
        atomic_json(path / "config.json", {"environment": env.config, "eval_seeds": seeds,
                                          "model_id": model_id, "train_seeds": train_seeds,
                                          "simulation": report["simulation"], **provenance})
        atomic_json(path / "metadata.json", report)
        ledger.append_row(row)
        return report
    finally:
        env.close()


def replay(store: Store, identifier: str, image: bool = False) -> dict | bytes:
    from .sim import ManipulationEnv
    arrays, record = store.load_episode(identifier)
    if record.get("seed") is None or "config" not in record:
        raise ValueError("Replay requires a simulation seed and environment configuration")
    env = ManipulationEnv(record["config"])
    try:
        if contract(env.metadata()) != record.get("contract_sha256"):
            raise ValueError("Replay model or observation/action contract has changed")
        observed = env.reset(seed=record["seed"])
        error = float(np.max(np.abs(observed - arrays["observations"][0])))
        reward_error, terminals_match = 0.0, True
        outcome = {}
        for index, action in enumerate(arrays["actions"]):
            observed, reward, terminal, outcome = env.step(action)
            error = max(error, float(np.max(np.abs(observed - arrays["observations"][index + 1]))))
            reward_error = max(reward_error, abs(float(reward) - float(arrays["rewards"][index])))
            terminals_match &= bool(terminal) == bool(arrays["terminated"][index])
            if terminal and index != len(arrays["actions"]) - 1:
                raise ValueError("Replay terminated before the recorded episode ended")
        matched = error <= 1e-5 and reward_error <= 1e-5 and terminals_match
        if image:
            if not matched:
                raise ValueError("Replay transitions do not match the recorded episode")
            import io
            from PIL import Image
            stream = io.BytesIO()
            Image.fromarray(env.render()).save(stream, format="PNG")
            return stream.getvalue()
        return {"episode_id": identifier, "steps": len(arrays["actions"]),
                "max_observation_error": error, "max_reward_error": reward_error,
                "terminals_match": terminals_match, "matched": matched,
                "outcome": outcome, "hardware": False}
    finally:
        env.close()


def benchmark(store: Store, seeds: list[int], conditions: list[dict], model_ids: list[str],
              progress=None, cancelled=None) -> dict:
    """Paired comparison: scripted baseline and checkpoints see identical trials."""
    if not conditions or len(conditions) > 12 or len(set(model_ids)) != len(model_ids):
        raise ValueError("Choose 1..12 conditions and distinct model IDs")
    if len(seeds) * len(conditions) * (len(model_ids) + 1) > 200:
        raise ValueError("A local benchmark is limited to 200 total rollouts; split larger studies")
    identifier, path = store.create("evaluations")
    rows = []
    total = len(conditions) * (len(model_ids) + 1)
    for condition_index, config in enumerate(conditions):
        for model_id in [None, *model_ids]:
            check_cancel(cancelled)
            result = evaluate(store, seeds, config, model_id, cancelled=cancelled,
                              experiment=f"benchmark_{identifier}")
            rows.append({"condition_index": condition_index, "config": result["config"],
                         "model_id": model_id, "evaluation_id": result["id"], "ledger_id": result["ledger_id"],
                         "successes": result["successes"], "trials": result["trials"],
                         "success_rate": result["success_rate"], "success_rate_ci95": result["success_rate_ci95"],
                         "failure_counts": result["failure_counts"]})
            if progress:
                progress({"completed": len(rows), "total": total, **rows[-1]})
    report = {"id": identifier, "kind": "paired_benchmark", "seeds": seeds, "comparisons": rows,
              "hardware": False, "safety_certification": False,
              "scope": "Same initial-state seeds per condition; simulation perturbations, not hardware robustness proof"}
    atomic_json(path / "metadata.json", report)
    return report
