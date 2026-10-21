"""Small CLI over the same backend functions used by the API."""
import argparse
import json
from pathlib import Path

from .storage import Store, atomic_json
from . import pipeline


def settings(text):
    return json.loads(Path(text[1:]).read_text("utf-8-sig") if text.startswith("@") else text)


def main():
    parser = argparse.ArgumentParser(description="Simulation-only XLeRobot workbench")
    parser.add_argument("--data", default="data", help="Local artifact directory")
    parser.add_argument("--output", type=Path, help="Also save result JSON without shell redirection")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-bench", help="Export the model XML and seeded block positions for reach_trials")
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--positions", type=Path, required=True)
    export.add_argument("--seeds", default="0..8", help="Inclusive nonnegative seed range, e.g. 0..8")
    serve = commands.add_parser("serve", help="Start localhost API and interactive /docs")
    serve.add_argument("--port", type=int, default=8765)
    collect = commands.add_parser("collect", help="Record scripted contact-simulation demonstrations")
    collect.add_argument("--episodes", type=int, default=10)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument('--action-noise-std', type=float, default=0.0,
                         help='Gaussian arm-target noise; save corrective expert labels separately')
    collect.add_argument("--config", default="{}", help="JSON environment config")
    train = commands.add_parser("train", help="Train on explicit episode IDs or a collection result JSON")
    train.add_argument("--episodes", nargs="+")
    train.add_argument("--collection", type=Path)
    train.add_argument("--config", default="{}", help="JSON training config")
    evaluate = commands.add_parser("evaluate", help="Evaluate saved policy or scripted baseline")
    evaluate.add_argument("--model", default=None)
    evaluate.add_argument("--episodes", type=int, default=10)
    evaluate.add_argument("--seed", type=int, default=10000)
    evaluate.add_argument("--config", default="{}")
    replay = commands.add_parser("replay", help="Re-run recorded actions and compare all observations")
    replay.add_argument("episode")
    replay.add_argument("--image", type=Path)
    smoke = commands.add_parser("smoke", help="Run collect -> train -> held-out evaluation -> replay")
    smoke.add_argument("--epochs", type=int, default=2)
    smoke.add_argument("--horizon", type=int, default=600)
    smoke.add_argument("--model-type", choices=["baseline_mlp", "act"], default="baseline_mlp")
    args = parser.parse_args()
    if args.command == "export-bench":
        import numpy as np
        from .sim import ManipulationEnv, model_xml
        try:
            first, last = map(int, args.seeds.split(".."))
            if not 0 <= first <= last:
                raise ValueError
        except ValueError:
            parser.error("--seeds must be an inclusive nonnegative range, e.g. 0..8")
        if args.model.resolve() == args.positions.resolve():
            parser.error("--model and --positions must be different files")
        env = ManipulationEnv()
        try:
            positions = []
            for seed in range(first, last + 1):
                env.reset(seed)
                positions.append(env.data.qpos[env.block_q:env.block_q+2].copy())
            args.model.parent.mkdir(parents=True, exist_ok=True)
            args.positions.parent.mkdir(parents=True, exist_ok=True)
            args.model.write_text(model_xml(), encoding="utf-8")
            np.savetxt(args.positions, positions, delimiter=",", fmt="%.17g")
        finally:
            env.close()
        print(json.dumps({"model": str(args.model), "positions": str(args.positions), "trials": len(positions)}))
        return
    if args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(args.data), host="127.0.0.1", port=args.port)
        return
    import torch
    torch.set_num_threads(1)
    store = Store(args.data)
    if args.command == "collect":
        result = pipeline.collect(store, list(range(args.seed, args.seed + args.episodes)), settings(args.config),
                                  action_noise_std=args.action_noise_std)
    elif args.command == "train":
        if bool(args.episodes) == bool(args.collection):
            parser.error("Specify exactly one of --episodes or --collection")
        identifiers = args.episodes or json.loads(args.collection.read_text("utf-8-sig"))["episode_ids"]
        result = pipeline.train_model(store, identifiers, settings(args.config))
    elif args.command == "evaluate":
        result = pipeline.evaluate(store, list(range(args.seed, args.seed + args.episodes)),
                                   settings(args.config), args.model)
    elif args.command == "replay":
        result = pipeline.replay(store, args.episode)
        if args.image:
            args.image.parent.mkdir(parents=True, exist_ok=True)
            args.image.write_bytes(pipeline.replay(store, args.episode, image=True))
            result["image"] = str(args.image)
    else:
        config = {"horizon": args.horizon}
        collection = pipeline.collect(store, [0, 1, 2, 3], config)
        model = pipeline.train_model(store, collection["episode_ids"], {
            "model_type": args.model_type, "epochs": args.epochs,
            "hidden_dim": 32, "batch_size": 64, "chunk_size": 8})
        expert = pipeline.evaluate(store, [10000, 10001], config)
        evaluation = pipeline.evaluate(store, [10000, 10001], config, model["id"])
        replay_result = pipeline.replay(store, collection["episode_ids"][0])
        result = {"scope": "pipeline smoke test, not a policy performance benchmark", "collection": collection,
                  "model_id": model["id"], "training_metrics": model["metrics"],
                  "expert_evaluation": expert, "learned_evaluation": evaluation, "replay": replay_result,
                  "hardware": False}
        atomic_json(store.root / "smoke-report.json", result)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
