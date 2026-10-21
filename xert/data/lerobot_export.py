"""Lossless, simulation-only NPZ transitions in the LeRobot v3 container."""
import argparse
import ctypes
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from xle_workbench.pipeline import contract
from xle_workbench.storage import Store, atomic_json, sha256

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    'xlerobot_left': ('assets/xlerobot/SOURCE.json',
                     ['Rotation_L', 'Pitch_L', 'Elbow_L', 'Wrist_Pitch_L', 'Wrist_Roll_L', 'Jaw_L']),
    'so101': ('assets/menagerie/robotstudio_so101/SOURCE.json',
              ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']),
}


def _schema(metadata, robot_type):
    simulation = metadata['simulation']
    source_path, joints = PROFILES[robot_type]
    if (metadata.get('hardware') is not False or simulation.get('simulation_only') is not True
            or simulation.get('source') != json.loads((ROOT / source_path).read_text('utf-8'))
            or metadata.get('contract_sha256') != contract(simulation)
            or simulation.get('action_names') != joints):
        raise ValueError('Source identity, simulation provenance, or joint contract does not match profile')
    names = simulation['observation_names']
    expected_positions = [f'joint_position.{name}' for name in joints]
    if (not isinstance(names, list) or len(names) != metadata['observation_dim']
            or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names)
            or names[:6] != expected_positions or metadata['action_dim'] != 6
            or simulation.get('observation_dim') != metadata['observation_dim']
            or simulation.get('action_dim') != metadata['action_dim']
            or len(names) != (26 if robot_type == 'xlerobot_left' else 6)):
        raise ValueError('Observation names/dimensions do not match the declared profile')
    fps = simulation['control_hz']
    if type(fps) not in (int, float) or not np.isfinite(fps) or fps <= 0 or int(fps) != fps:
        raise ValueError('LeRobot export requires a positive integer control rate')
    if not isinstance(simulation.get('observation_convention'), str):
        raise ValueError('Observation units/convention must be explicit')
    if robot_type == 'so101' and simulation['observation_convention'] != 'joint position radians':
        raise ValueError('SO-101 simulation state must explicitly use joint position radians')
    ranges = np.asarray(simulation.get('action_joint_ranges_radians'), dtype=float)
    if ranges.shape != (6, 2) or not np.isfinite(ranges).all() or np.any(ranges[:, 0] >= ranges[:, 1]):
        raise ValueError('Normalized action decoding ranges must be finite and ordered')
    if simulation.get('action_convention') != 'absolute joint position targets normalized to [-1,1] by actuator_ctrlrange':
        raise ValueError('Unsupported action convention')
    state_names = names if robot_type == 'xlerobot_left' else [f'{name}.pos' for name in joints]
    action_names = joints if robot_type == 'xlerobot_left' else state_names
    def vector(names):
        return {'dtype': 'float32', 'shape': (len(names),), 'names': names}
    return int(fps), {
        'observation.state': vector(state_names),
        'action': vector(action_names),
        'xert.executed_action': vector(action_names),
        'xert.next_observation': vector(state_names),
        'next.reward': vector(['reward']),
        'next.done': {'dtype': 'bool', 'shape': (1,), 'names': ['terminated']},
    }


def _publish(source, destination):
    """Atomic directory publication without replacing even an empty destination."""
    if sys.platform == 'win32':
        os.rename(source, destination)
    elif sys.platform.startswith('linux'):
        # Linux rename() replaces empty directories; renameat2 NOREPLACE does not.
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))
    else:
        raise RuntimeError('Atomic no-replace export is supported on Windows and Linux')


def verify_roundtrip(store, episode_ids, root, repo_id, task):
    """Load through the real LeRobot API and compare every source transition."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend='pyav', download_videos=False)
    offset, maximum_error = 0, 0.0
    try:
        for episode_index, identifier in enumerate(episode_ids):
            executed, _ = store.load_episode(identifier)
            training, _ = store.load_training_episode(identifier)
            for step in range(len(executed['actions'])):
                frame = dataset[offset + step]
                expected = {
                    'observation.state': executed['observations'][step],
                    'action': training['actions'][step],
                    'xert.executed_action': executed['actions'][step],
                    'xert.next_observation': executed['observations'][step + 1],
                    # LeRobot returns scalar features as scalar tensors.
                    'next.reward': np.asarray(executed['rewards'][step]),
                    'next.done': np.asarray(executed['terminated'][step]),
                }
                for key, value in expected.items():
                    actual = frame[key].numpy()
                    if actual.shape != value.shape or not np.array_equal(actual, value):
                        raise ValueError(f'Round-trip mismatch: {identifier}/{step}/{key}')
                    if value.dtype != np.bool_:
                        maximum_error = max(maximum_error, float(np.max(np.abs(actual - value))))
                if (frame['episode_index'].item() != episode_index or frame['frame_index'].item() != step
                        or frame['index'].item() != offset + step or frame['task'] != task
                        or not np.isclose(frame['timestamp'].item(), step / dataset.fps, atol=1e-6, rtol=1e-6)):
                    raise ValueError('Round-trip frame indexing, task, or derived simulation time mismatch')
            offset += len(executed['actions'])
        if len(dataset) != offset or dataset.num_episodes != len(episode_ids):
            raise ValueError('Round-trip frame/episode count mismatch')
    finally:
        del dataset
        gc.collect()  # Release Arrow memory maps before Windows directory publication.
    return {'frames': offset, 'episodes': len(episode_ids), 'max_abs_error': maximum_error}


def export_episodes(store, episode_ids, output, *, task, robot_type='xlerobot_left', repo_id='xert/local-export'):
    """Write a fresh local dataset. Existing source and destination are never overwritten."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if robot_type not in PROFILES or not isinstance(task, str) or not task.strip():
        raise ValueError('Choose a supported profile and nonempty task')
    if (not isinstance(episode_ids, list) or not episode_ids
            or any(not isinstance(identifier, str) for identifier in episode_ids)
            or len(set(episode_ids)) != len(episode_ids)):
        raise ValueError('Choose distinct source episode IDs')
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(store.root):
        raise ValueError('Export destination must be outside the immutable source store')
    records = []
    schema = None
    for identifier in episode_ids:
        _, metadata = store.load_training_episode(identifier)
        current = _schema(metadata, robot_type)
        if schema is not None and (schema != current or records[0]['contract_sha256'] != metadata['contract_sha256']):
            raise ValueError('Cannot combine differing source schemas or simulation contracts')
        schema = current
        records.append(metadata)
    fps, features = schema
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.xert-export-', dir=output.parent) as temporary:
        staging = Path(temporary) / 'dataset'
        dataset = LeRobotDataset.create(repo_id=repo_id, root=staging, robot_type=robot_type,
                                       fps=fps, features=features, use_videos=False, video_backend='pyav')
        try:
            for identifier, original in zip(episode_ids, records, strict=True):
                executed, metadata = store.load_episode(identifier)
                training, label_metadata = store.load_training_episode(identifier)
                if metadata != original or label_metadata != original:
                    raise ValueError('Source metadata changed during export')
                for step in range(len(executed['actions'])):
                    dataset.add_frame({
                        'observation.state': executed['observations'][step],
                        'action': training['actions'][step],
                        'xert.executed_action': executed['actions'][step],
                        'xert.next_observation': executed['observations'][step + 1],
                        'next.reward': executed['rewards'][step:step + 1],
                        'next.done': executed['terminated'][step:step + 1],
                        'task': task,
                    })
                dataset.save_episode(parallel_encoding=False)
        finally:
            dataset.finalize()
        del dataset
        gc.collect()
        verification = verify_roundtrip(store, episode_ids, staging, repo_id, task)
        atomic_json(staging / 'xert_export.json', {
            'schema_version': 1, 'backend': 'mujoco', 'robot_type': robot_type, 'task': task,
            'lerobot_version': importlib.metadata.version('lerobot'), 'fps': fps,
            'timestamp_semantics': 'derived simulation time: frame_index / fps; not measured capture time',
            'action_semantics': 'verified corrective labels when present, otherwise executed normalized targets',
            'executed_action_feature': 'xert.executed_action', 'terminal_observation_feature': 'xert.next_observation',
            'source_episodes': records,
        })
        atomic_json(staging / 'roundtrip.json', verification)
        _publish(staging, output)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--robot-type', choices=PROFILES, default='xlerobot_left')
    args = parser.parse_args()
    from xert.eval import ledger
    provenance = ledger.source_provenance()
    ids = json.loads(args.collection.read_text('utf-8'))['episode_ids']
    result = export_episodes(Store(args.source), ids, args.output, task=args.task, robot_type=args.robot_type)
    files = sorted(path for path in args.output.rglob('*') if path.is_file())
    row = ledger.new_row(**provenance, slice='P2.1', experiment='lerobot_export_' + args.output.name,
                         backend='mujoco', embodiment=args.robot_type, metric='roundtrip_max_abs_error',
                         value=result['max_abs_error'], ci95=None, n=result['frames'], seeds={'train': [], 'eval': []},
                         config_path=ledger.artifact_path(args.output / 'xert_export.json'),
                         artifact_paths=[ledger.artifact_path(path) for path in files],
                         artifact_sha256={ledger.artifact_path(path): sha256(path) for path in files},
                         notes='Exact float32 NPZ/LeRobot reconstruction, including terminal observations and both action streams. '
                               'Simulation-time timestamps are derived, not capture latency; not hardware or a learning result.')
    ledger.append_row(row)
    print(json.dumps({**result, 'ledger_id': row['id']}))


if __name__ == '__main__':
    main()
