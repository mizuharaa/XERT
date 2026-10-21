"""Executed actions replay exactly; expert labels are computed at the actual state."""
import hashlib
import numpy as np
import pytest
import torch

from xle_workbench import pipeline
from xle_workbench import storage
from xle_workbench.sim import ManipulationEnv
from xle_workbench.storage import Store, atomic_json, sha256


def test_recovery_labels_replay_and_training(tmp_path):
    torch.set_num_threads(1)
    store = Store(tmp_path)
    result = pipeline.collect(store, [100, 101], {'horizon': 12}, action_noise_std=.005)
    fingerprints = []
    for identifier in result['episode_ids']:
        executed, record = store.load_episode(identifier)
        training, _ = store.load_training_episode(identifier)
        assert not np.array_equal(training['actions'], executed['actions'])
        np.testing.assert_array_equal(training['actions'][:, -1], executed['actions'][:, -1])
        env = ManipulationEnv(record['config'])
        env.reset(record['seed'])
        for label, action in zip(training['actions'], executed['actions']):
            np.testing.assert_array_equal(label, np.asarray(env.expert_action(), np.float32))
            env.step(action)
        env.close()
        assert pipeline.replay(store, identifier)['matched']
        fingerprints.append(hashlib.sha256(training['observations'].tobytes() + training['actions'].tobytes()).hexdigest())
    model = pipeline.train_model(store, result['episode_ids'], {'epochs': 1, 'hidden_dim': 16})
    assert model['metrics']['episode_sha256'] == fingerprints
    assert model['training_action_sources'] == ['expert_correction', 'expert_correction']
    identifier = result['episode_ids'][0]
    path = store.path('episodes', identifier)
    payload = path / 'corrective.npz'
    raw = payload.read_bytes()
    payload.unlink()
    with pytest.raises(ValueError, match='corrective labels'):
        store.load_training_episode(identifier)
    payload.write_bytes(raw + b'corrupt')
    with pytest.raises(ValueError, match='corrective labels'):
        store.load_training_episode(identifier)
    payload.write_bytes(raw)
    arrays, metadata = store.load_training_episode(identifier)
    arrays['observations'] = arrays['observations'].copy()
    arrays['observations'][0, 0] += 1
    np.savez_compressed(payload, **arrays)
    metadata['corrective_sha256'] = sha256(payload)
    atomic_json(path / 'metadata.json', metadata)
    with pytest.raises(ValueError, match='aligned'):
        store.load_training_episode(identifier)


def test_recovery_noise_validation(tmp_path):
    for value in (-.01, .051, float('nan'), True):
        with pytest.raises(ValueError, match='action_noise_std'):
            pipeline.collect(Store(tmp_path), [100], action_noise_std=value)


@pytest.mark.parametrize('filename', ['episode.npz', 'corrective.npz'])
def test_archive_hash_and_decode_use_same_snapshot(tmp_path, monkeypatch, filename):
    store = Store(tmp_path)
    identifier = pipeline.collect(store, [100], {'horizon': 2}, action_noise_std=.005)['episode_ids'][0]
    path = store.path('episodes', identifier) / filename
    original = storage.decode_npz(path.read_bytes())
    forged = {**original, 'actions': np.full_like(original['actions'], .75)}
    real_hash = storage.sha256

    def replace_after_hash(candidate):
        digest = real_hash(candidate)
        if candidate == path:
            np.savez_compressed(path, **forged)
        return digest

    monkeypatch.setattr(storage, 'sha256', replace_after_hash)
    loaded, _ = (store.load_episode(identifier) if filename == 'episode.npz'
                 else store.load_training_episode(identifier))
    np.testing.assert_array_equal(loaded['actions'], original['actions'])
