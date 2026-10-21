r"""Runnable: .venv\Scripts\python -m pytest tests/test_learning.py -q"""
import json
import hashlib

import numpy as np
import pytest
import torch

from xle_workbench.learning import ActionChunks, StateACT, episode_split, load_policy, masked_loss, train
from xle_workbench.learning import LoadedPolicy, validate_config


def episodes():
    rng = np.random.default_rng(42)
    result = []
    for _ in range(6):
        observations = rng.normal(size=(17, 3)).astype(np.float32)
        actions = np.tanh(observations[:-1, :2] * 0.5)
        result.append(dict(observations=observations, actions=actions))
    return result


def test_chunks_masks_and_no_terminal_targets():
    ep = dict(observations=np.arange(12, dtype=np.float32).reshape(4, 3),
              actions=np.array([[.1, .2], [.3, .4], [.5, .6]], dtype=np.float32))
    chunks = ActionChunks([ep], [0], 4, np.zeros(3, np.float32), np.ones(3, np.float32))
    obs, target, valid = chunks[2]
    np.testing.assert_array_equal(obs, ep["observations"][2])
    assert valid.tolist() == [True, False, False, False]
    prediction = target.clone()
    prediction[1:] = 1000
    assert masked_loss(prediction.unsqueeze(0), target.unsqueeze(0), valid.unsqueeze(0)).item() == 0


def test_training_round_trip_and_train_only_normalization(tmp_path):
    torch.set_num_threads(1)
    data = episodes()
    train_ids, val_ids = episode_split(len(data), 0, .2)
    # Strong terminal outliers must not contaminate normalizers either.
    for ep in data:
        ep["observations"][-1] = 10000
    output = train(data, tmp_path, {"epochs": 45, "hidden_dim": 32, "lr": .008})
    train_obs = np.concatenate([data[i]["observations"][:-1] for i in train_ids])
    np.testing.assert_allclose(output["observation_mean"], train_obs.mean(axis=0), atol=1e-6)
    assert output["training_episode_indices"] == train_ids
    assert output["validation_episode_indices"] == val_ids
    assert set(train_ids).isdisjoint(val_ids)
    assert output["best_validation_mae"] < output["history"][0]["validation_mae"] * .5
    policy = load_policy(tmp_path)
    first = policy.act(data[val_ids[0]]["observations"][0])
    policy.reset()
    np.testing.assert_array_equal(first, load_policy(tmp_path / "model.pt").act(data[val_ids[0]]["observations"][0]))
    assert first.shape == (2,) and np.isfinite(first).all() and np.abs(first).max() <= 1
    assert output["rollout_success_measured"] is False
    json.dumps(output, allow_nan=False)


def test_act_trains_and_resets_open_loop_chunk(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data = episodes()
    progress = []
    calls = {"posterior": 0, "prior": 0}
    original_forward = StateACT.forward

    def checked_forward(self, obs, actions=None, valid=None):
        if self.training:
            assert actions is not None and valid is not None
            calls["posterior"] += 1
        else:
            assert actions is None and valid is None
            calls["prior"] += 1
        return original_forward(self, obs, actions, valid)

    monkeypatch.setattr(StateACT, "forward", checked_forward)
    output = train(data, tmp_path, {"model_type": "act", "epochs": 2, "hidden_dim": 16,
                                   "chunk_size": 4, "batch_size": 16}, progress=progress.append)
    assert len(progress) == 2 and np.isfinite(output["best_validation_mae"])
    assert calls["posterior"] > 0 and calls["prior"] > 0
    policy = load_policy(tmp_path)
    obs = data[0]["observations"][0]
    first = policy.act(obs)
    assert len(policy.pending) == 3
    policy.act(obs + 10)
    assert len(policy.pending) == 2
    policy.reset()
    np.testing.assert_array_equal(first, policy.act(obs))
    with pytest.raises(ValueError, match="Observation"):
        policy.act([float("nan")] * 3)


def test_act_padded_posterior_and_deterministic_prior():
    torch.set_num_threads(1)
    model = StateACT(3, 2, 16, 4).eval()
    obs, actions = torch.randn(2, 3), torch.randn(2, 4, 2)
    valid = torch.tensor([[True, False, False, False], [True, True, False, False]])
    modified = actions.clone()
    modified[~valid] = 1000
    with torch.inference_mode():
        torch.manual_seed(7)
        prediction, kl = model(obs, actions, valid)
        torch.manual_seed(7)
        other_prediction, other_kl = model(obs, modified, valid)
        torch.testing.assert_close(prediction, other_prediction)
        torch.testing.assert_close(kl, other_kl)
        prior, prior_kl = model(obs)
        torch.manual_seed(999)
        other_prior, _ = model(obs)
        torch.testing.assert_close(prior, other_prior, rtol=0, atol=0)
        assert prior_kl.item() == 0


def test_temporal_ensemble_replans_aligns_and_resets():
    class ChunkModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, observation):
            self.calls += 1
            return torch.tensor([[[.1], [.2], [.3]]]) + (self.calls - 1) * .2, torch.tensor(0.)

    model = ChunkModel()
    policy = LoadedPolicy(model, dict(obs_dim=1, observation_mean=[0.], observation_scale=[1.],
                                     config={'temporal_ensemble': True, 'ensemble_decay': np.log(2)}))
    np.testing.assert_allclose(policy.act([0]), [.1])
    np.testing.assert_allclose(policy.act([1]), [(.2 * .5 + .3) / 1.5])
    np.testing.assert_allclose(policy.act([2]), [(.3 * .25 + .4 * .5 + .5) / 1.75])
    assert model.calls == 3 and len(policy.chunks) == 2
    policy.reset()
    assert not policy.chunks
    np.testing.assert_allclose(policy.act([3]), [.7])
    for invalid in ({'model_type': 'act', 'temporal_ensemble': 'yes'},
                    {'model_type': 'act', 'ensemble_decay': float('nan')},
                    {'temporal_ensemble': True}):
        with pytest.raises(ValueError):
            validate_config(invalid)


def test_checkpoint_rejects_nonfinite_weights_and_normalization(tmp_path):
    torch.set_num_threads(1)
    train(episodes(), tmp_path, {"epochs": 1})
    metadata_path = tmp_path / "metadata.json"
    original = json.loads(metadata_path.read_text())
    for value in (1e300, float("nan")):
        changed = json.loads(json.dumps(original))
        changed["observation_mean"][0] = value
        metadata_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="normalization"):
            load_policy(tmp_path)
    weights_path = tmp_path / "model.pt"
    weights = torch.load(weights_path, weights_only=True)
    next(iter(weights.values())).flatten()[0] = float("nan")
    torch.save(weights, weights_path)
    original["weights_sha256"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="finite tensors"):
        load_policy(tmp_path)


def test_constant_features_and_extreme_inference_values(tmp_path):
    torch.set_num_threads(1)
    data = episodes()
    train_ids, val_ids = episode_split(len(data), 0, .2)
    for index in train_ids:
        data[index]["observations"][:, 2] = 0
    for index in val_ids:
        data[index]["observations"][:, 2] = 1
    result = train(data, tmp_path, {"epochs": 1})
    assert result["observation_mean"][2] == 0
    assert result["observation_scale"][2] == float(np.float32(1e-4))
    policy = load_policy(tmp_path)
    assert np.isfinite(policy.act([0, 0, 0])).all()
    with pytest.raises(ValueError, match="Normalized observation"):
        policy.act([0, 0, np.finfo(np.float32).max])


def test_invalid_data_cancellation_and_tampered_checkpoint(tmp_path):
    torch.set_num_threads(1)
    data = episodes()
    with pytest.raises(InterruptedError):
        train(data, tmp_path, {"epochs": 1}, cancelled=lambda: True)
    assert not (tmp_path / "model.pt").exists()
    with pytest.raises(ValueError, match="At least two"):
        train(data[:1], tmp_path, {})
    with pytest.raises(ValueError, match="Identical"):
        train([data[0], data[0]], tmp_path, {})
    broken = episodes()
    broken[0]["actions"][0, 0] = 2
    with pytest.raises(ValueError, match="finite observations"):
        train(broken, tmp_path, {})
    train(data, tmp_path, {"epochs": 1})
    with (tmp_path / "model.pt").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_policy(tmp_path)
