"""G1: official LeRobot ACT on rendered `scene`/`wrist` images. Scaffolding only; no trained policy, no result.

Every training choice is a TODO(daniel) stub (image_act_settings, make_optimizer, training_step, select_epoch).
This module owns the plumbing around them: dataset checks, train-split-only normalization, the chunked data
loader, checkpoints with provenance, and the held-out bench evaluation.

Rollouts run in simulated time with no wall-clock deadline. Each tick renders both cameras from the live
simulator state and queries the policy between two bench steps; every bench step still advances physics by
exactly 1/25 s, so rendering cost never changes the 25 Hz control contract in sim time.
"""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from xert.data.cameras import CAMERAS, _recorded_robot
from xert.data.quality import HUMAN_SOURCES
from xert.eval import ledger
from xle_workbench.storage import atomic_json, sha256

IMAGE_KEYS = ("observation.images.scene", "observation.images.wrist")
SETTINGS_KEYS = {"act", "epochs", "batch_size", "image_transforms", "validation_episodes"}
MIN_EVAL_EPISODES = 30  # Same bar as P5.3 before a success rate becomes a ledger row.
METADATA = "xert_image_act.json"
REPO_ID = "xert/runtime"
SPECS = Path(__file__).resolve().parents[1] / "spec"
STD_FLOOR = 1e-4  # Same floor as P1.4: a constant feature (the static tray) must not divide by ~0.


# ---------------------------------------------------------------- TODO(daniel): every training choice lives here.

def image_act_settings(dataset_meta):
    """TODO(daniel): choose every image-ACT training setting for this dataset.

    Input: dataset_meta = {"robot", "fps", "episodes" (count), "frames" (count), "features" (the LeRobot dataset
    feature dict, images stored (H, W, 3))}.
    Output: a dict with exactly the keys in SETTINGS_KEYS:
      - "act": ACTConfig keyword arguments: chunk_size, n_action_steps, vision_backbone, pretrained_backbone_weights,
        dim_model, n_heads, dim_feedforward, n_encoder_layers, n_decoder_layers, n_vae_encoder_layers, latent_dim,
        dropout, kl_weight, temporal_ensemble_coeff, optimizer_lr (read it in make_optimizer),
        normalization_mapping, device. Never input_features/output_features: the plumbing derives those.
      - "epochs", "batch_size": positive ints.
      - "image_transforms": None, or ImageTransformsConfig fields (augmentation, applied to the TRAIN split only).
      - "validation_episodes": dataset episode indices held out of training (every other episode trains).
    Invariants: JSON-serializable (stored in the checkpoint); ACTConfig(**act) validates. Pre-register the
    performance target you will judge the evaluation by before the first evaluate; the plumbing has none.
    """
    raise NotImplementedError("image-ACT settings not chosen (TODO(daniel))")


def make_optimizer(policy):
    """TODO(daniel): the optimizer for `policy`, an ACTPolicy whose `config` holds settings["act"].

    Output: a torch.optim.Optimizer over the policy's parameters (e.g. a separate backbone learning rate).
    """
    raise NotImplementedError("image-ACT optimizer not chosen (TODO(daniel))")


def training_step(policy, batch, optimizer):
    """TODO(daniel): one update on one training batch; return the batch loss as a float.

    Input: `policy` in train mode; `batch` after the train-only normalizer: every input feature (images
    (B, 3, H, W) float), "action" (B, chunk_size, action_dim) and "action_is_pad" (B, chunk_size) bool, True
    past the episode end. Output: a finite float, logged per epoch as "train_loss".
    Invariants: the plumbing never steps the optimizer, so this owns the loss, backward and step; raise on a
    nonfinite loss.
    """
    raise NotImplementedError("image-ACT training step not written (TODO(daniel))")


def select_epoch(history):
    """TODO(daniel): the checkpoint selection and early-stopping rule. Called after every epoch.

    Input: history = [{"epoch": k, "train_loss": mean training_step loss, "validation_l1": mean |predicted -
    target| normalized action over valid chunk positions with ACT's prior (no VAE encoder), or None without
    validation episodes}], one entry per finished epoch, in order.
    Output: (selected_epoch, stop). selected_epoch must be the newest epoch (its weights replace the kept ones)
    or the previous pick (the kept weights stay); anything else is refused, because only one epoch's weights are
    ever on disk (the P1.4 keep-best pattern). stop=True ends training early. The checkpoint is the last pick.
    """
    raise NotImplementedError("image-ACT selection rule not chosen (TODO(daniel))")


# ---------------------------------------------------------------- Plumbing.

def dataset_roots(paths):
    """Each path is one LeRobot dataset, or a session directory of episode_* datasets."""
    roots = []
    for path in map(Path, paths):
        found = [path] if (path / "meta/info.json").exists() else sorted(path.glob("episode_*"))
        if not found:
            raise ValueError(f"{path} is neither a LeRobot dataset nor a session of episode_* datasets")
        roots += found
    return roots


def _provenance(root):
    """(source kinds, bench seeds) of a P4.1 recording or a P2.1 export."""
    if (root / "xert_recording.json").exists():
        import pyarrow.parquet as pq
        source = json.loads((root / "xert_recording.json").read_text("utf-8"))["source"]
        frames = {value for path in sorted(root.glob("data/**/*.parquet"))
                  for value in pq.read_table(path, columns=["xert.source"]).column("xert.source").to_pylist()} - {""}
        if frames - {source["kind"]}:  # The P5.1 check: per-frame sources must agree with the manifest.
            raise ValueError(f"{root}: frame sources {sorted(frames)} conflict with manifest {source['kind']!r}")
        return {source["kind"]}, {source.get("bench_seed")} - {None}
    records = json.loads((root / "xert_export.json").read_text("utf-8"))["source_episodes"]
    return {str(r.get("source")) for r in records}, {r["seed"] for r in records if isinstance(r.get("seed"), int)}


def describe(paths):
    """Check a render-cameras output and summarize it from its JSON metadata (no LeRobot import)."""
    roots = dataset_roots(paths)
    infos = [json.loads((root / "meta/info.json").read_text("utf-8")) for root in roots]
    for root, info in zip(roots, infos):
        missing = [key for key in IMAGE_KEYS if info["features"].get(key, {}).get("dtype") not in ("image", "video")]
        if missing:
            raise ValueError(f"{root} lacks camera images {missing}; run render-cameras first")
        if (info["features"], info["fps"]) != (infos[0]["features"], infos[0]["fps"]):
            raise ValueError(f"{root} declares different features or fps from {roots[0]}")
    robots = {_recorded_robot(root) for root in roots}
    if len(robots) != 1 or not robots <= set(CAMERAS):
        raise ValueError(f"Need one bench robot of {sorted(CAMERAS)}; datasets name {robots}")
    robot = robots.pop()
    if not all((root / "xert_cameras.json").exists() for root in roots):
        raise ValueError("Every dataset must be a render-cameras output (xert_cameras.json)")
    manifests = [json.loads((root / "xert_cameras.json").read_text("utf-8")) for root in roots]
    camera = {"model_sha256": manifests[0]["model_sha256"], "size": manifests[0]["cameras"]["scene"]["height"],
              "cameras": CAMERAS[robot]}
    for manifest in manifests:  # Evaluation renders with CameraRenderer; the training images must match it.
        if (manifest["model_sha256"] != camera["model_sha256"]
                or any(manifest["cameras"][name]["height"] != camera["size"]
                       or any(manifest["cameras"][name][key] != value for key, value in definition.items())
                       for name, definition in CAMERAS[robot].items())):
            raise ValueError("Datasets were rendered with other camera definitions, sizes or models than CAMERAS")
    from xle_workbench.sim import ROBOTS
    joints = ROBOTS[robot]["joints"]
    action_names = infos[0]["features"]["action"]["names"]
    if action_names == [f"{joint}.pos" for joint in joints]:
        action_units = "radians"  # P4.1 recordings: absolute joint targets.
    elif action_names == joints:
        action_units = "normalized"  # P2.1 exports: targets normalized to [-1, 1] by actuator_ctrlrange.
    else:
        raise ValueError(f"Actions {action_names} are not absolute bench joint targets")
    kinds, seeds = set(), set()
    for root in roots:
        k, s = _provenance(root)
        kinds |= k
        seeds |= s
    digest = hashlib.sha256()
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            digest.update(path.relative_to(root.parent).as_posix().encode() + b"\0" + sha256(path).encode())
    return {"roots": roots, "episodes": [(root, i) for root, info in zip(roots, infos)
                                         for i in range(info["total_episodes"])],
            "meta": {"robot": robot, "fps": infos[0]["fps"], "features": infos[0]["features"],
                     "episodes": sum(info["total_episodes"] for info in infos),
                     "frames": sum(info["total_frames"] for info in infos)},
            "camera": camera, "action_units": action_units, "source_kinds": sorted(kinds),
            "bench_seeds": sorted(seeds), "dataset_sha256": digest.hexdigest()}


def policy_config(features, settings):
    """Official ACTConfig: inputs and output from the dataset features, every other field from the settings."""
    from lerobot.configs.types import FeatureType
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.utils.feature_utils import dataset_to_policy_features
    if set(settings) != SETTINGS_KEYS:
        raise ValueError(f"Settings need exactly {sorted(SETTINGS_KEYS)}")
    if {"input_features", "output_features"} & set(settings["act"]):
        raise ValueError("Input and output features come from the dataset, not the settings")
    json.dumps(settings)  # Stored in the checkpoint metadata.
    policy = dataset_to_policy_features(features)
    return ACTConfig(input_features={key: f for key, f in policy.items() if f.type is not FeatureType.ACTION},
                     output_features={"action": policy["action"]}, **settings["act"])


def _datasets(episodes, indices, **options):
    """One LeRobotDataset per root holding the selected episodes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    chosen = {}
    for index in indices:
        root, episode = episodes[index]
        chosen.setdefault(root, []).append(episode)
    return [LeRobotDataset(REPO_ID, root=root, episodes=chosen[root], **options) for root in chosen]


def train_stats(datasets, keys):
    """mean/std/min/max per normalized feature over the given (TRAIN) frames; images per channel, (3, 1, 1)."""
    import torch
    sums = {}
    for dataset in datasets:
        for batch in torch.utils.data.DataLoader(dataset, batch_size=64):
            for key in keys:
                x = batch[key].double()
                dims = (0, 2, 3) if x.ndim == 4 else (0,)
                count = x.numel() // x.shape[1]
                n, total, square, low, high = sums.get(key, (0, 0., 0., math.inf, -math.inf))
                sums[key] = (n + count, total + x.sum(dims), square + (x * x).sum(dims),
                             torch.minimum(torch.as_tensor(low), x.amin(dims)),
                             torch.maximum(torch.as_tensor(high), x.amax(dims)))
    stats = {}
    for key, (n, total, square, low, high) in sums.items():
        mean = total / n
        std = (square / n - mean * mean).clamp_min(0).sqrt().clamp_min(STD_FLOOR)
        shape = (-1, 1, 1) if key.startswith("observation.images.") else (-1,)
        stats[key] = {name: value.float().reshape(shape) for name, value in
                      (("mean", mean), ("std", std), ("min", low), ("max", high))}
    return stats


def _image_transforms(settings):
    if settings is None:
        return None
    from lerobot.transforms.transforms import ImageTransformConfig, ImageTransforms, ImageTransformsConfig
    fields = dict(settings)
    if "tfs" in fields:
        fields["tfs"] = {name: ImageTransformConfig(**value) for name, value in fields["tfs"].items()}
    return ImageTransforms(ImageTransformsConfig(**fields))


def train(paths, output, seed=0):
    """Train one image-ACT checkpoint into a fresh `output` directory."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    data = describe(paths)
    settings = image_act_settings(data["meta"])
    config = policy_config(data["meta"]["features"], settings)
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    episodes, validation = data["episodes"], settings["validation_episodes"]
    if (any(type(i) is not int or not 0 <= i < len(episodes) for i in validation)
            or len(set(validation)) != len(validation) or len(validation) >= len(episodes)):
        raise ValueError(f"validation_episodes must be distinct indices < {len(episodes)} and leave training data")
    training_ids = [i for i in range(len(episodes)) if i not in validation]
    torch.manual_seed(seed)
    # Normalization statistics from the TRAIN split only, before any augmentation.
    stats = train_stats(_datasets(episodes, training_ids), [*config.input_features, "action"])
    delta = {"action": [i / data["meta"]["fps"] for i in range(config.chunk_size)]}  # Chunk targets, padded at ends.
    training = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(_datasets(episodes, training_ids, delta_timestamps=delta,
                                                 image_transforms=_image_transforms(settings["image_transforms"]))),
        batch_size=settings["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed))
    checking = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(_datasets(episodes, validation, delta_timestamps=delta)),
        batch_size=settings["batch_size"]) if validation else None
    policy = ACTPolicy(config)
    preprocessor, postprocessor = make_act_pre_post_processors(config, stats)
    optimizer = make_optimizer(policy)
    keys = [*config.input_features, "action", "action_is_pad"]
    output.mkdir(parents=True)
    history, selected, kept = [], None, output / "epochs" / "selected"
    try:
        for epoch in range(1, settings["epochs"] + 1):
            policy.train()
            losses = [training_step(policy, preprocessor({key: batch[key] for key in keys}), optimizer)
                      for batch in training]
            error = count = 0
            if checking:
                policy.eval()
                with torch.inference_mode():
                    for batch in checking:
                        batch = preprocessor({key: batch[key] for key in keys})
                        prediction = policy.predict_action_chunk({key: batch[key] for key in config.input_features})
                        valid = ~batch["action_is_pad"]
                        error += ((prediction - batch["action"]).abs() * valid.unsqueeze(-1)).sum().item()
                        count += valid.sum().item() * prediction.shape[-1]
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                            "validation_l1": error / count if count else None})
            if not all(math.isfinite(v) for v in history[-1].values() if v is not None):
                raise ValueError(f"Nonfinite training history: {history[-1]}")
            print(json.dumps({"phase": "training", "seed": seed, **history[-1]}), flush=True)
            choice, stop = select_epoch(history)
            if choice == epoch:
                policy.save_pretrained(kept)  # Overwrites the one kept epoch (P1.4 keep-best).
            elif choice is None or choice != selected:
                raise ValueError(f"select_epoch chose {choice} after epoch {epoch}; only the newest epoch or the "
                                 f"previous pick ({selected}) can be kept")
            selected = choice
            if stop:
                break
    except BaseException:
        shutil.rmtree(output / "epochs", ignore_errors=True)  # A failed run keeps its output, not stray weights.
        raise
    shutil.copytree(kept, output, dirs_exist_ok=True)
    shutil.rmtree(output / "epochs")
    preprocessor.save_pretrained(output)
    postprocessor.save_pretrained(output)
    atomic_json(output / METADATA, {
        "slice": "G1", "backend": "mujoco", "hardware": False, **ledger.source_provenance(),
        "robot": data["meta"]["robot"], "fps": data["meta"]["fps"], "action_units": data["action_units"],
        "datasets": [str(root) for root in data["roots"]], "dataset_sha256": data["dataset_sha256"],
        "source_kinds": data["source_kinds"], "bench_seeds": data["bench_seeds"],
        "episodes": [[str(root), i] for root, i in episodes], "training_episodes": training_ids,
        "validation_episodes": validation, "camera": data["camera"],
        "features": {key: data["meta"]["features"][key] for key in config.input_features},
        "settings": settings, "seed": seed, "history": history, "selected_epoch": selected,
        "normalization": "train-split mean/std/min/max (images per channel), std floored at 1e-4",
        "runtime": {key: importlib.metadata.version(key) for key in ("torch", "lerobot", "mujoco")}})
    return output


def live_frame(backend, renderer, features, native_names):
    """One unbatched policy input from the live bench: vectors by feature name, cameras rendered from the live qpos.

    Images go through the same CameraRenderer (camera definitions, hidden sites) as render-cameras, as float
    (3, H, W) in [0, 1], which is how LeRobotDataset returns the stored PNGs.
    """
    import torch
    from xert.eval.gate_experiment import bench_observation
    native = dict(zip(native_names, backend.env._obs()))  # ponytail: private accessor; the native env has no public one.
    images, _ = renderer.render(backend.read_state().q, {name: native[name] for name in renderer.object_names})
    return {key: torch.from_numpy(images[key]).permute(2, 0, 1).float().div(255) if key in images
            else torch.from_numpy(bench_observation(backend, feature["names"], native_names))
            for key, feature in features.items()}


def decode(action, units, low, high):
    """Policy action in dataset units -> absolute joint targets in radians (P4.1 radians, or P2.1 [-1, 1])."""
    if units == "radians":
        return action
    if units == "normalized":
        return low + (action + 1) / 2 * (high - low)
    raise ValueError(f"Unknown action units {units!r}")


def rollout(policy, preprocessor, postprocessor, spec, seed, metadata):
    """One held-out bench episode; success is the native core's contact rule. One renderer per rollout."""
    import torch
    from xert.backends.bench import BenchBackend
    from xert.data.cameras import CameraRenderer
    backend = BenchBackend(spec, seed)
    renderer = CameraRenderer(metadata["camera"]["size"], metadata["robot"])
    try:
        if renderer.model_sha256 != metadata["camera"]["model_sha256"]:
            raise ValueError("The bench camera model differs from the one the training images were rendered on")
        native_names, (low, high) = backend.env.metadata()["observation_names"], backend.env.limits.T
        policy.reset()
        render, infer = [], []
        while not backend.finished:
            start = time.perf_counter()
            frame = live_frame(backend, renderer, metadata["features"], native_names)
            middle = time.perf_counter()
            with torch.inference_mode():
                action = postprocessor(policy.select_action(preprocessor(frame)))[0].numpy().astype(np.float64)
            render.append(middle - start)
            infer.append(time.perf_counter() - middle)
            if action.shape != low.shape or not np.isfinite(action).all():
                raise ValueError("Image ACT emitted an invalid action")
            backend.send_targets(np.clip(decode(action, metadata["action_units"], low, high), low, high),
                                 None)  # Joint-limit clip only, as in P5.3.
            if backend.info["success"]:
                break
    finally:
        renderer.close()
    return {"seed": seed, "success": bool(backend.info["success"]), "steps": backend.info["steps"],
            "failure_reason": backend.info["failure_reason"]}, render, infer


def _ledger_reason(kinds, n):
    """Why an evaluation writes no ledger row, or None: rows need human-only training data and enough seeds."""
    if not kinds or set(kinds) - set(HUMAN_SOURCES):
        return f"no rows: training data is {list(kinds)}, not only {list(HUMAN_SOURCES)}"
    if n < MIN_EVAL_EPISODES:
        return f"no rows: {n} seeds < {MIN_EVAL_EPISODES}"
    return None


def evaluate(checkpoint, seeds, output, experiment="image_act_v1"):
    """Roll a checkpoint out on held-out bench seeds; a report file always, ledger rows only for human data."""
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from xert.spec import load_spec
    from xle_workbench.pipeline import wilson
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(output)
    metadata = json.loads((checkpoint / METADATA).read_text("utf-8"))
    if (not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("Evaluation seeds must be distinct nonnegative integers")
    if set(seeds) & set(metadata["bench_seeds"]):
        raise ValueError("Evaluation seeds overlap the training demonstrations' bench seeds; held-out states required")
    if CAMERAS[metadata["robot"]] != metadata["camera"]["cameras"]:
        raise ValueError("CAMERAS changed since training; live renders would differ from the training images")
    reason = _ledger_reason(metadata["source_kinds"], len(seeds))
    provenance, checkpoint_sha256 = ledger.source_provenance(), sha256(checkpoint / METADATA)
    if reason is None:  # Rows only (as in P5.3); a report on a dirty tree stays allowed.
        if provenance["git_dirty"] or metadata["git_dirty"]:
            raise ValueError("Repository or checkpoint has uncommitted changes; commit them so every row names "
                             "the exact source")
        if any(row["experiment"] == experiment and row.get("checkpoint_sha256") == checkpoint_sha256
               for row in ledger.read_rows()):
            raise ValueError(f"Ledger already has {experiment} rows for this checkpoint; re-running it on new seeds "
                             "is seed-shopping, so choose a new experiment tag")
    policy = ACTPolicy.from_pretrained(checkpoint, local_files_only=True)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    spec = load_spec(SPECS / f"{metadata['robot']}.yaml")
    torch.set_num_threads(1)
    trials, render, infer = [], [], []
    for seed in seeds:
        trial, r, i = rollout(policy, preprocessor, postprocessor, spec, seed, metadata)
        trials.append(trial)
        render += r
        infer += i
        print(json.dumps({"phase": "evaluation", **trial}), flush=True)
    successes = sum(trial["success"] for trial in trials)
    report = {
        "slice": "G1", "experiment": experiment, "checkpoint": str(checkpoint), "robot": metadata["robot"],
        "train_seed": metadata["seed"], "seeds": list(seeds), "trials": trials, "successes": successes,
        "n": len(trials), "success_rate": successes / len(trials), "ci95": wilson(successes, len(trials)),
        "source_kinds": metadata["source_kinds"],
        "ledger": reason or "rows appended",
        "timing_ms_per_tick": {"ticks": len(render), "render_mean": 1e3 * float(np.mean(render)),
                               "render_max": 1e3 * float(np.max(render)),
                               "inference_mean": 1e3 * float(np.mean(infer)),
                               "inference_max": 1e3 * float(np.max(infer))},
        "notes": "Simulated time: rendering and inference run between 1/25 s bench steps with no wall-clock "
                 "deadline. Images are live MuJoCo renders, not camera data. No hardware claim."}
    row = None
    if reason is None:
        row = ledger.new_row(**provenance, slice="G1", experiment=experiment, backend="mujoco",
                             embodiment=metadata["robot"], metric="success_rate", value=report["success_rate"],
                             ci95=report["ci95"], n=len(trials), seeds={"train": [metadata["seed"]], "eval": list(seeds)},
                             config_path=ledger.artifact_path(checkpoint / METADATA), source_kinds=metadata["source_kinds"],
                             dataset_sha256=metadata["dataset_sha256"], checkpoint_sha256=checkpoint_sha256,
                             notes="Official LeRobot ACT on rendered scene + wrist images and bench state, rolled "
                                   "out on held-out MuJoCo seeds with live renders. Simulation only; no hardware claim.")
        report["ledger_row"] = row["id"]
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    if row:
        artifacts = [*sorted(p for p in checkpoint.iterdir() if p.is_file()), output]
        row.update(artifact_paths=[ledger.artifact_path(p) for p in artifacts],
                   artifact_sha256={ledger.artifact_path(p): sha256(p) for p in artifacts})
        ledger.append_row(row)
    return report


def add_command(commands):
    parser = commands.add_parser("image-act", help="Train / evaluate official ACT on rendered scene + wrist images")
    operations = parser.add_subparsers(dest="operation", required=True)
    training = operations.add_parser("train", help="Needs the TODO(daniel) settings in xert/eval/image_act.py")
    training.add_argument("--dataset", type=Path, nargs="+", required=True,
                          help="render-cameras outputs: datasets or session directories of episode_* datasets")
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seed", type=int, default=0)
    evaluation = operations.add_parser("evaluate", help="Roll a checkpoint out on held-out bench seeds")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--seeds", type=int, nargs="+", required=True)
    evaluation.add_argument("--output", type=Path, required=True, help="Fresh report JSON path")
    evaluation.add_argument("--experiment", default="image_act_v1")


def execute(args):
    if args.operation == "train":
        print(json.dumps({"checkpoint": str(train(args.dataset, args.output, args.seed))}))
    else:
        report = evaluate(args.checkpoint, args.seeds, args.output, args.experiment)
        print(json.dumps({key: report[key] for key in ("successes", "n", "ci95", "ledger")}))
