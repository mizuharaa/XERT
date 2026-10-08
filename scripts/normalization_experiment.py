"""Controlled CPU ablation; canonical learners and reference artifacts stay untouched.

Run from the project root. Checkpoints and raw rollouts remain in a new OS temp
directory, whose path and source changes are recorded in the JSON report.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile
import time
import types

import numpy as np
import torch

from xle_workbench.pipeline import contract, rollout
from xle_workbench.sim import ManipulationEnv
from xle_workbench.storage import Store, atomic_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline_mlp", "act"], default="baseline_mlp")
    parser.add_argument("--output", type=Path, default=Path("docs/audit/NORMALIZATION_EXPERIMENT_MLP.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh report path; previous experiments are retained")
    torch.set_num_threads(1)
    reference_path = Path("docs/REFERENCE_EXPERIMENT.json")
    reference = json.loads(reference_path.read_text("utf-8"))
    proof = reference[args.model]
    data_store = Store("data")
    pairs = [data_store.load_episode(identifier) for identifier in proof["episode_ids"]]
    assert [row[1]["sha256"] for row in pairs] == proof["episode_sha256"]
    episodes = [pair[0] for pair in pairs]
    config = {**proof["metrics"]["config"], "device": "cpu"}
    seeds = [20000, 20001, 20002]
    source_path = Path("xle_workbench/learning.py")
    source_bytes = source_path.read_bytes()
    source = source_bytes.decode("utf-8")
    scale_line = "    scale = np.maximum(train_obs.std(axis=0, dtype=np.float64), 1e-4).astype(np.float32)"
    normalize_line = "    return normalized.astype(np.float32)"
    assert source.count(scale_line) == 1 and source.count(normalize_line) == 1
    variants = {
        "original": source,
        "floored_scale_one": source.replace(scale_line, scale_line + "\n    scale[scale <= np.float32(1e-4)] = 1.0"),
        "mask_floored_features": source.replace(normalize_line,
            "    normalized[..., scale <= np.float32(1e-4)] = 0.0\n" + normalize_line),
    }
    area = Path(tempfile.mkdtemp(prefix="xle-normalization-ablation-"))
    report = {
        "scope": "Controlled simulation-only normalization ablation, not hardware validation",
        "model_type": args.model, "started_at": time.time(), "temporary_artifacts_retained": str(area),
        "reference_sha256": sha256(reference_path), "learning_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "script_sha256": sha256(Path(__file__)), "config": config,
        "episode_ids": proof["episode_ids"], "episode_sha256": proof["episode_sha256"],
        "evaluation_seeds": seeds, "notes": [
            "Same training split, initialization seed, epochs, and architecture in every variant.",
            "Evaluation seeds were used in previous diagnostics: this is a development ablation, not a fresh final test.",
            "A unit scale for low-variance inputs is a sensitivity ablation, not calibrated per-sensor engineering units.",
            "Masking a feature hides actual deviations and is not automatically safe for deployment.",
            "Three deterministic simulation trials per variant cannot establish general task competence.",
            "Original checkpoint is reused after verifying saved config, dataset, and weight hashes; historical learning source hash was not recorded, so that provenance cannot be proven retrospectively.",
        ],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": torch.__version__, "cpu_threads": torch.get_num_threads()},
        "variants": [],
    }
    env = ManipulationEnv(proof["simulation"]["config"])
    try:
        if contract(env.metadata()) != proof["contract_sha256"]:
            raise ValueError("Simulator contract differs from reference; do not conflate normalization with environment changes")
        report["simulation"] = env.metadata()
    finally:
        env.close()
    atomic_json(args.output, report)
    train_ids = proof["metrics"]["training_episode_indices"]
    train_obs = np.concatenate([episodes[i]["observations"][:-1] for i in train_ids])
    original_scale = np.array(proof["metrics"]["observation_scale"])
    names = config["observation_names"]
    report["floored_features"] = [names[i] for i in np.where(original_scale <= float(np.float32(1e-4)))[0]]
    for label, modified in variants.items():
        print("Reuse original" if label == "original" else "Training", args.model, label, flush=True)
        variant_dir = area / label
        variant_dir.mkdir()
        module_path = variant_dir / "learning_snapshot.py"
        module_path.write_text(modified, encoding="utf-8")
        module = types.ModuleType("normalization_ablation_" + label)
        module.__file__ = str(module_path)
        exec(compile(modified, str(module_path), "exec"), module.__dict__)
        started = time.perf_counter()
        if label == "original":
            original_dir = data_store.path("models", proof["id"])
            assert sha256(original_dir / "model.pt") == proof["model_sha256"]
            assert sha256(original_dir / "metadata.json") == proof["metadata_sha256"]
            metrics = json.loads((original_dir / "metadata.json").read_text("utf-8"))
            assert metrics["config"] == config
            (variant_dir / "checkpoint").mkdir()
            for filename in ("model.pt", "metadata.json"):
                shutil.copy2(original_dir / filename, variant_dir / "checkpoint" / filename)
        else:
            metrics = module.train(episodes, variant_dir / "checkpoint", config)
        assert metrics["training_episode_indices"] == train_ids
        assert metrics["validation_episode_indices"] == proof["metrics"]["validation_episode_indices"]
        policy = module.load_policy(variant_dir / "checkpoint")
        row = {"variant": label, "reused_original_checkpoint": label == "original", "source_sha256": sha256(module_path),
               "source_diff": "".join(difflib.unified_diff(source.splitlines(True), modified.splitlines(True),
                                                        fromfile="canonical", tofile=label)),
               "metrics": metrics, "trials": []}
        mean = np.array(metrics["observation_mean"])
        scale = np.array(metrics["observation_scale"])
        train_z = np.stack([module._normalize(obs, mean, scale) for obs in train_obs])
        env = ManipulationEnv(proof["simulation"]["config"])
        try:
            assert contract(env.metadata()) == proof["contract_sha256"]
            for seed in seeds:
                arrays, outcome = rollout(env, seed, policy)
                payload = variant_dir / f"rollout-{seed}.npz"
                np.savez_compressed(payload, **arrays)
                normalized = np.stack([module._normalize(obs, mean, scale) for obs in arrays["observations"]])
                peaks = np.max(np.abs(normalized), axis=0)
                top = []
                for i in np.argsort(peaks)[-5:][::-1]:
                    step = int(np.argmax(np.abs(normalized[:, i])))
                    top.append({"feature": names[i], "max_abs_normalized": float(peaks[i]),
                                "step": step, "seconds": step / env.metadata()["control_hz"],
                                "training_max_abs_normalized": float(np.abs(train_z[:, i]).max())})
                row["trials"].append({"seed": seed, "outcome": outcome, "raw_rollout": str(payload),
                                      "raw_sha256": sha256(payload), "steps": len(arrays["actions"]),
                                      "action_saturation_fraction": float((np.abs(arrays["actions"]) > .99).mean()),
                                      "block_max_height": float(arrays["observations"][:, names.index("block.z")].max()),
                                      "initial_action": arrays["actions"][0].tolist(), "top_normalized_features": top})
        finally:
            env.close()
        row["successes"] = sum(trial["outcome"]["success"] for trial in row["trials"])
        row["elapsed_seconds"] = time.perf_counter() - started
        report["variants"].append(row)
        atomic_json(args.output, report)
        print(label, "successes", row["successes"], "/", len(seeds),
              "validation_mae", metrics["best_validation_mae"], flush=True)
    report["canonical_source_unchanged"] = source_path.read_bytes() == source_bytes
    report["finished_at"] = time.time()
    atomic_json(args.output, report)
    print("Saved", args.output, "artifacts", area, flush=True)


if __name__ == "__main__":
    main()
