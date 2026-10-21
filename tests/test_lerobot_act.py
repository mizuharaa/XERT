"""Optional real-package checks; no simulator or long training needed."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('lerobot')
import torch

from xert.eval.lerobot_act import RuntimePolicy, fit, inputs, normalization, policy_config
from xle_workbench.learning import ActionChunks


def test_complete_feature_partition_and_training_only_statistics():
    observation = torch.arange(26, dtype=torch.float32).unsqueeze(0)
    batch = inputs(observation)
    torch.testing.assert_close(torch.cat(list(batch.values()), dim=-1), observation)
    with pytest.raises(ValueError, match='complete legacy state'):
        inputs(observation[..., :6])
    episodes = [{'observations': np.arange(104, dtype=np.float32).reshape(4, 26),
                 'actions': np.zeros((3, 6), np.float32)},
                {'observations': np.full((3, 26), 10000, np.float32),
                 'actions': np.zeros((2, 6), np.float32)}]
    source = episodes[0]['observations'][:-1]
    metadata = {'training_episode_indices': [0], 'validation_episode_indices': [1],
                'observation_mean': source.mean(axis=0, dtype=np.float64).astype(np.float32).tolist(),
                'observation_scale': source.std(axis=0, dtype=np.float64).astype(np.float32).tolist()}
    mean, scale, _, _ = normalization(episodes, metadata)
    episodes[1]['observations'][:] *= 10
    np.testing.assert_array_equal(normalization(episodes, metadata)[0], mean)
    chunks = ActionChunks(episodes, [0], 3, mean, scale)
    _, actions, valid = chunks[2]
    assert valid.tolist() == [True, False, False]
    assert actions.shape == (3, 6)
    with pytest.raises(ValueError, match='normalization differs'):
        normalization(episodes, {**metadata, 'observation_mean': [0] * 26})
    with pytest.raises(ValueError, match='does not partition'):
        normalization(episodes, {**metadata, 'validation_episode_indices': [0]})


def test_real_official_act_checkpoint_and_episode_reset(tmp_path):
    from lerobot.policies.act.modeling_act import ACTPolicy
    torch.set_num_threads(1)
    settings = json.loads(Path('configs/lerobot_act_experiment.json').read_text())
    # Synthetic one-step optimizer test only; this is not the measured experiment budget.
    settings.update(epochs=1, chunk_size=3, batch_size=2)
    rng = np.random.default_rng(7)
    episodes = [{'observations': rng.normal(size=(4, 26)).astype(np.float32),
                 'actions': rng.uniform(-1, 1, size=(3, 6)).astype(np.float32)} for _ in range(2)]
    observations = episodes[0]['observations'][:-1]
    metadata = {'training_episode_indices': [0], 'validation_episode_indices': [1],
                'observation_mean': observations.mean(axis=0, dtype=np.float64).astype(np.float32).tolist(),
                'observation_scale': np.maximum(observations.std(axis=0, dtype=np.float64), 1e-4).astype(np.float32).tolist()}
    runtime, stats = fit(episodes, settings, 0, metadata, tmp_path)
    assert isinstance(runtime.policy, ACTPolicy)
    assert np.isfinite(stats['best_validation_mae'])
    runtime.reset()
    first = runtime.act(episodes[0]['observations'][0])
    runtime.act(episodes[0]['observations'][1])
    runtime.reset()
    np.testing.assert_array_equal(runtime.act(episodes[0]['observations'][0]), first)
    assert first.shape == (6,) and np.max(np.abs(first)) <= 1
    config = policy_config(settings)
    assert config.temporal_ensemble_coeff == .01 and config.n_action_steps == 1
    assert not config.image_features
    assert config.env_state_feature.shape == (14,)


def test_runtime_rejects_nonfinite_actions():
    class BrokenPolicy:
        def select_action(self, batch):
            return torch.full((1, 6), float('nan'))
    runtime = RuntimePolicy(BrokenPolicy(), np.zeros(26), np.ones(26))
    with pytest.raises(ValueError, match='invalid action'):
        runtime.act(np.zeros(26, np.float32))
