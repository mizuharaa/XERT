"""Run with the optional LeRobot interpreter; no native core or motors needed."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('lerobot', reason='Install the isolated LeRobot environment for the real v3 round trip')
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from xert.data import lerobot_export as export
from xle_workbench.pipeline import contract
from xle_workbench.storage import Store, sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/p1.2/recovery_integrated'
IDS = json.loads((SOURCE / 'collection.json').read_text())['episode_ids'][:2]


def test_real_legacy_roundtrip_and_no_overwrite(tmp_path):
    store = Store(SOURCE / 'dataset')
    destination = tmp_path / 'export'
    before = {path: sha256(path) for identifier in IDS for path in store.path('episodes', identifier).iterdir()}
    result = export.export_episodes(store, IDS, destination, task='Move the block to the tray')
    assert result['episodes'] == 2 and result['max_abs_error'] == 0
    dataset = LeRobotDataset('xert/local-export', root=destination, video_backend='pyav')
    assert dataset.meta.info.codebase_version == 'v3.0'
    assert dataset.meta.robot_type == 'xlerobot_left' and dataset.fps == 25
    assert dataset.features['observation.state']['shape'] == (26,)
    arrays, _ = store.load_episode(IDS[0])
    labels, _ = store.load_training_episode(IDS[0])
    np.testing.assert_array_equal(dataset[0]['action'], labels['actions'][0])
    np.testing.assert_array_equal(dataset[0]['xert.executed_action'], arrays['actions'][0])
    assert not np.array_equal(labels['actions'], arrays['actions'])
    manifest = json.loads((destination / 'xert_export.json').read_text())
    assert manifest['timestamp_semantics'].startswith('derived simulation time')
    hashes = {path: sha256(path) for path in destination.rglob('*') if path.is_file()}
    with pytest.raises(FileExistsError):
        export.export_episodes(store, IDS, destination, task='replacement')
    assert hashes == {path: sha256(path) for path in destination.rglob('*') if path.is_file()}
    assert before == {path: sha256(path) for path in before}


def test_failure_cleans_staging_and_profile_rejects_relabeling(tmp_path, monkeypatch):
    store = Store(SOURCE / 'dataset')
    output = tmp_path / 'failed'
    with pytest.raises(ValueError, match='outside'):
        export.export_episodes(store, IDS, store.root / 'new-export', task='test')
    with pytest.raises(ValueError, match='profile'):
        export.export_episodes(store, IDS, output, task='test', robot_type='so101')
    assert not output.exists()
    def fail(*args, **kwargs):
        raise ValueError('injected verification failure')
    monkeypatch.setattr(export, 'verify_roundtrip', fail)
    with pytest.raises(ValueError, match='injected'):
        export.export_episodes(store, IDS[:1], output, task='test')
    assert list(tmp_path.iterdir()) == []


def test_atomic_publication_preserves_existing_empty_directory(tmp_path):
    staging, destination = tmp_path / 'staging', tmp_path / 'existing'
    staging.mkdir()
    destination.mkdir()
    (staging / 'payload').write_text('preserve')
    with pytest.raises(OSError):
        export._publish(staging, destination)
    assert destination.is_dir() and list(destination.iterdir()) == []
    assert (staging / 'payload').read_text() == 'preserve'


def test_so101_six_joint_schema_fixture(tmp_path):
    # Synthetic schema fixture, not a recorded demonstration or hardware evidence.
    store = Store(tmp_path / 'source')
    source_path, names = export.PROFILES['so101']
    simulation = {
        'simulation_only': True, 'source': json.loads((ROOT / source_path).read_text()),
        'observation_names': [f'joint_position.{name}' for name in names],
        'action_names': names, 'observation_dim': 6, 'action_dim': 6,
        'control_hz': 25, 'horizon': 3,
        'observation_convention': 'joint position radians',
        'action_convention': 'absolute joint position targets normalized to [-1,1] by actuator_ctrlrange',
        'action_joint_ranges_radians': json.loads((ROOT / 'xert/spec/so101.yaml').read_text())['position_limits'],
    }
    arrays = {'observations': np.arange(24, dtype=np.float32).reshape(4, 6) / 100,
              'actions': np.arange(18, dtype=np.float32).reshape(3, 6) / 100,
              'rewards': np.array([0, 0, 1], np.float32), 'terminated': np.array([False, False, True])}
    record = store.save_episode(arrays, {'source': 'test_fixture', 'simulation': simulation,
                                        'contract_sha256': contract(simulation), 'hardware': False})
    destination = tmp_path / 'so101'
    result = export.export_episodes(store, [record['id']], destination, task='schema fixture', robot_type='so101')
    dataset = LeRobotDataset('xert/local-export', root=destination, video_backend='pyav')
    expected = [f'{name}.pos' for name in names]
    assert dataset.features['observation.state']['names'] == expected
    assert dataset.features['action']['names'] == expected
    assert result == {'frames': 3, 'episodes': 1, 'max_abs_error': 0.0}
