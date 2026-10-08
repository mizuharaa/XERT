"""State-only imitation learning. See docs/LEARNING.md before changing the contracts."""

from collections import deque
from pathlib import Path
import hashlib
import json
import math
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULTS = dict(model_type="baseline_mlp", epochs=20, batch_size=64, chunk_size=8,
                hidden_dim=64, seed=0, val_fraction=0.2, lr=0.001,
                kl_weight=0.01, device="cpu")


def validate_config(config):
    unknown = set(config) - (set(DEFAULTS) | {"observation_names", "action_names"})
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **config}
    if cfg["model_type"] not in ("baseline_mlp", "act"):
        raise ValueError("model_type must be baseline_mlp or act")
    for key, low, high in (("epochs", 1, 2000), ("batch_size", 1, 1024),
                           ("chunk_size", 1, 128), ("hidden_dim", 16, 512),
                           ("seed", 0, 2**31 - 1)):
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError(f"{key} must be an integer in [{low}, {high}]")
    if cfg["hidden_dim"] % 4:
        raise ValueError("hidden_dim must be divisible by four attention heads")
    for key, low, high in (("val_fraction", 0.01, 0.5), ("lr", 1e-7, 0.1),
                           ("kl_weight", 0.0, 100.0)):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or not low <= cfg[key] <= high:
            raise ValueError(f"{key} must be finite and in [{low}, {high}]")
    if cfg["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; explicitly select cpu")
    if cfg["model_type"] == "baseline_mlp":
        cfg["chunk_size"] = 1
    return cfg


def validate_episodes(episodes):
    if not isinstance(episodes, (list, tuple)) or len(episodes) < 2:
        raise ValueError("At least two episodes are required for an episode-level split")
    result = []
    dimensions = None
    for i, episode in enumerate(episodes):
        obs = np.asarray(episode["observations"], dtype=np.float32)
        actions = np.asarray(episode["actions"], dtype=np.float32)
        if obs.ndim != 2 or actions.ndim != 2 or len(actions) < 1 or len(obs) != len(actions) + 1:
            raise ValueError(f"Episode {i}: expected observations (T+1,D) and actions (T,A), T >= 1")
        shape = (obs.shape[1], actions.shape[1])
        if not 1 <= shape[0] <= 512 or not 1 <= shape[1] <= 64 or (dimensions and shape != dimensions):
            raise ValueError(f"Episode {i}: incompatible or excessive dimensions")
        if not np.isfinite(obs).all() or not np.isfinite(actions).all() or np.abs(actions).max() > 1.000001:
            raise ValueError(f"Episode {i}: finite observations and actions in [-1,1] required")
        dimensions = shape
        result.append({"observations": obs, "actions": actions})
    return result


def episode_split(count, seed, val_fraction):
    """Split trajectories first, never neighboring frames from the same trajectory."""
    order = np.random.default_rng(seed).permutation(count).tolist()
    n_val = max(1, min(count - 1, math.ceil(count * val_fraction)))
    return order[n_val:], order[:n_val]


class ActionChunks(Dataset):
    def __init__(self, episodes, indices, chunk_size, mean, scale):
        self.episodes, self.chunk_size = episodes, chunk_size
        self.mean, self.scale = mean, scale
        self.index = [(i, t) for i in indices for t in range(len(episodes[i]["actions"]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, t = self.index[index]
        ep = self.episodes[i]
        obs = _normalize(ep["observations"][t], self.mean, self.scale)
        count = min(self.chunk_size, len(ep["actions"]) - t)
        target = np.zeros((self.chunk_size, ep["actions"].shape[1]), np.float32)
        target[:count] = ep["actions"][t:t + count]
        valid = np.arange(self.chunk_size) < count
        return torch.from_numpy(obs), torch.from_numpy(target), torch.from_numpy(valid)


class BaselineMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, chunk_size=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim), nn.Tanh())

    def forward(self, obs, actions=None, valid=None):
        return self.net(obs).unsqueeze(1), obs.new_zeros(())


class StateACT(nn.Module):
    """Compact CVAE action chunker, no vision backbone and no temporal ensemble."""
    def __init__(self, obs_dim, action_dim, hidden_dim, chunk_size):
        super().__init__()
        self.latent_dim = 16
        self.state = nn.Linear(obs_dim, hidden_dim)
        self.action = nn.Linear(action_dim, hidden_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.encoder_positions = nn.Parameter(torch.randn(1, chunk_size + 2, hidden_dim) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 4, dropout=0.0, batch_first=True),
            num_layers=1, enable_nested_tensor=False)
        self.posterior = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.latent = nn.Linear(self.latent_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(1, chunk_size, hidden_dim) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, 4, hidden_dim * 4, dropout=0.0, batch_first=True),
            num_layers=1)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, actions=None, valid=None):
        state = self.state(obs).unsqueeze(1)
        z = obs.new_zeros((len(obs), self.latent_dim))
        kl = obs.new_zeros(())
        if actions is not None:
            tokens = torch.cat((self.cls.expand(len(obs), -1, -1), state, self.action(actions)), dim=1)
            padding = torch.cat((torch.zeros((len(obs), 2), dtype=torch.bool, device=obs.device), ~valid), dim=1)
            encoded = self.encoder(tokens + self.encoder_positions, src_key_padding_mask=padding)[:, 0]
            mean, logvar = self.posterior(encoded).chunk(2, dim=-1)
            logvar = logvar.clamp(-10, 10)
            z = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
            kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
        memory = torch.cat((state, self.latent(z).unsqueeze(1)), dim=1)
        decoded = self.decoder(self.queries.expand(len(obs), -1, -1), memory)
        return self.output(decoded).tanh(), kl


def masked_loss(prediction, target, valid, squared=False):
    error = (prediction - target).square() if squared else (prediction - target).abs()
    return (error * valid.unsqueeze(-1)).sum() / (valid.sum() * target.shape[-1]).clamp_min(1)


def _model(cfg, obs_dim, action_dim):
    cls = StateACT if cfg["model_type"] == "act" else BaselineMLP
    return cls(obs_dim, action_dim, cfg["hidden_dim"], cfg["chunk_size"])


def _normalize(obs, mean, scale):
    # Subtract in float64: finite float32 endpoints can overflow their difference.
    normalized = (np.asarray(obs, dtype=np.float64) - mean) / scale
    if not np.isfinite(normalized).all() or (np.abs(normalized) > np.finfo(np.float32).max).any():
        raise ValueError("Normalized observation exceeds finite float32 range")
    return normalized.astype(np.float32)


def train(episodes, output_dir: Path, config: dict, progress=None, cancelled=None):
    """Fit a model; save best prior-inference validation checkpoint, never rollout success."""
    cfg = validate_config(config)
    episodes = validate_episodes(episodes)
    output_dir = Path(output_dir)
    if (output_dir / "metadata.json").exists() or (output_dir / "model.pt").exists():
        raise ValueError("Checkpoint directory already contains a model; use a fresh run directory")
    train_ids, val_ids = episode_split(len(episodes), cfg["seed"], cfg["val_fraction"])
    fingerprints = [hashlib.sha256(ep["observations"].tobytes() + ep["actions"].tobytes()).hexdigest() for ep in episodes]
    if {fingerprints[i] for i in train_ids} & {fingerprints[i] for i in val_ids}:
        raise ValueError("Identical episodes occur in training and validation; deduplicate the dataset")
    # Terminal observations have no target action and are deliberately excluded.
    train_obs = np.concatenate([episodes[i]["observations"][:-1] for i in train_ids])
    mean = train_obs.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(train_obs.std(axis=0, dtype=np.float64), 1e-4).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Observation magnitudes overflow normalization")
    obs_dim, action_dim = train_obs.shape[1], episodes[0]["actions"].shape[1]
    for key, dim in (("observation_names", obs_dim), ("action_names", action_dim)):
        if key in cfg and (not isinstance(cfg[key], list) or len(cfg[key]) != dim or
                           any(not isinstance(name, str) or not name for name in cfg[key]) or len(set(cfg[key])) != dim):
            raise ValueError(f"{key} must contain {dim} unique nonempty strings")
    torch.manual_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    model = _model(cfg, obs_dim, action_dim).to(device)
    generator = torch.Generator().manual_seed(cfg["seed"])
    training = DataLoader(ActionChunks(episodes, train_ids, cfg["chunk_size"], mean, scale),
                          batch_size=cfg["batch_size"], shuffle=True, generator=generator)
    validation = DataLoader(ActionChunks(episodes, val_ids, cfg["chunk_size"], mean, scale),
                            batch_size=cfg["batch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    history, best_weights = [], None
    best_mae, best_epoch = float("inf"), 0

    def check_cancelled():
        if cancelled and cancelled():
            raise InterruptedError("Training cancelled; no completed checkpoint written")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, samples = 0.0, 0
        for obs, actions, valid in training:
            check_cancelled()
            obs, actions, valid = obs.to(device), actions.to(device), valid.to(device)
            prediction, kl = model(obs, actions, valid)
            loss = masked_loss(prediction, actions, valid, squared=cfg["model_type"] == "baseline_mlp") + cfg["kl_weight"] * kl
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; checkpoint not written")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(obs)
            samples += len(obs)
        model.eval()
        absolute_error, count = 0.0, 0
        with torch.inference_mode():
            for obs, actions, valid in validation:
                check_cancelled()
                obs, actions, valid = obs.to(device), actions.to(device), valid.to(device)
                # Prior z=0; future held-out actions never enter the policy encoder.
                prediction, _ = model(obs)
                absolute_error += ((prediction - actions).abs() * valid.unsqueeze(-1)).sum().item()
                count += valid.sum().item() * action_dim
        val_mae = absolute_error / count
        if not math.isfinite(val_mae):
            raise ValueError("Non-finite validation error; checkpoint not written")
        if val_mae < best_mae:
            best_mae, best_epoch = val_mae, epoch
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        record = dict(epoch=epoch, train_loss=total_loss / samples, validation_mae=val_mae)
        history.append(record)
        if progress:
            progress(record)
    check_cancelled()
    metadata = dict(schema_version=1, config=cfg, obs_dim=obs_dim, action_dim=action_dim,
                    trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    runtime={"torch": torch.__version__, "numpy": np.__version__},
                    observation_mean=mean.tolist(), observation_scale=scale.tolist(),
                    training_episode_indices=train_ids, validation_episode_indices=val_ids,
                    episode_sha256=fingerprints, best_epoch=best_epoch,
                    best_validation_mae=best_mae, history=history,
                    execution="open_loop_action_chunks", observations="state_only",
                    rollout_success_measured=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = output_dir / "model.pt.tmp"
    torch.save(best_weights, tmp)
    os.replace(tmp, output_dir / "model.pt")
    metadata["weights_sha256"] = hashlib.sha256((output_dir / "model.pt").read_bytes()).hexdigest()
    tmp = output_dir / "metadata.json.tmp"
    tmp.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, output_dir / "metadata.json")
    return {**metadata, "checkpoint_dir": str(output_dir.resolve())}


class LoadedPolicy:
    def __init__(self, model, metadata):
        self.model, self.metadata = model.eval(), metadata
        self.mean = np.asarray(metadata["observation_mean"], np.float32)
        self.scale = np.asarray(metadata["observation_scale"], np.float32)
        self.pending = deque()

    def reset(self):
        self.pending.clear()

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.metadata["obs_dim"],) or not np.isfinite(obs).all():
            raise ValueError("Observation must match saved dimensions and contain only finite values")
        if not self.pending:
            x = torch.from_numpy(_normalize(obs, self.mean, self.scale)).unsqueeze(0)
            with torch.inference_mode():
                chunk, _ = self.model(x)
            self.pending.extend(chunk[0].numpy())
        action = np.clip(self.pending.popleft(), -1, 1).copy()
        if not np.isfinite(action).all():
            raise ValueError("Policy produced non-finite action")
        return action

    __call__ = act


def load_policy(path):
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    cfg = validate_config({**metadata["config"], "device": "cpu"})
    obs_dim, action_dim = metadata["obs_dim"], metadata["action_dim"]
    if type(obs_dim) is not int or type(action_dim) is not int or not 1 <= obs_dim <= 512 or not 1 <= action_dim <= 64:
        raise ValueError("Invalid checkpoint dimensions")
    mean, scale = np.asarray(metadata["observation_mean"]), np.asarray(metadata["observation_scale"])
    if mean.shape != (obs_dim,) or scale.shape != (obs_dim,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid checkpoint normalization")
    if (np.abs(mean) > np.finfo(np.float32).max).any() or (scale > np.finfo(np.float32).max).any() or (scale < float(np.float32(1e-4))).any():
        raise ValueError("Invalid checkpoint normalization: not representable or below training scale floor")
    weights_path = directory / "model.pt"
    if hashlib.sha256(weights_path.read_bytes()).hexdigest() != metadata["weights_sha256"]:
        raise ValueError("Checkpoint weights hash does not match metadata")
    model = _model(cfg, obs_dim, action_dim)
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                                            for value in weights.values()):
        raise ValueError("Checkpoint weights must contain only finite tensors")
    model.load_state_dict(weights)
    return LoadedPolicy(model, metadata)
