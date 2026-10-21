"""Bounded official ACT reference; preserves the legacy task and its held-out split."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import numpy as np
import torch
from torch.utils.data import DataLoader

from xert.eval import ledger
from xle_workbench import pipeline
from xle_workbench.learning import ActionChunks, _normalize
from xle_workbench.storage import Store, atomic_json, sha256


def policy_config(settings):
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    return ACTConfig(
        input_features={
            'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(12,)),
            'observation.environment_state': PolicyFeature(type=FeatureType.ENV, shape=(14,)),
        },
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        # Explicit normalization below exactly matches the old train-only statistics.
        normalization_mapping={key: NormalizationMode.IDENTITY for key in ('STATE', 'ENV', 'ACTION')},
        device='cpu', pretrained_backbone_weights=None, n_action_steps=1,
        **{key: settings[key] for key in (
            'chunk_size', 'dim_model', 'n_heads', 'dim_feedforward', 'n_encoder_layers',
            'n_decoder_layers', 'n_vae_encoder_layers', 'latent_dim', 'dropout',
            'kl_weight', 'temporal_ensemble_coeff')})


def inputs(observation):
    """Partition all normalized source features without dropping or duplicating any."""
    if observation.shape[-1] != 26:
        raise ValueError('Official ACT adapter requires the complete legacy state')
    return {'observation.state': observation[..., :12],
            'observation.environment_state': observation[..., 12:]}


def load_export(settings):
    """Read the real LeRobot dataset and prove identity against both immutable sources."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if importlib.metadata.version('lerobot') != '0.6.1':
        raise ValueError('Use the pinned optional ACT environment')
    root = Path(settings['export_path'])
    row = next(row for row in ledger.read_rows() if row['id'] == settings['export_ledger_id'])
    if row['slice'] != 'P2.1' or row['metric'] != 'roundtrip_max_abs_error' or row['value'] != 0:
        raise ValueError('Require a verified lossless export ledger row')
    expected_files = {Path(path).resolve() for path in row['artifact_sha256']}
    if {path.resolve() for path in root.rglob('*') if path.is_file()} != expected_files:
        raise ValueError('Export file inventory differs from the ledger')
    for path, expected in row['artifact_sha256'].items():
        if sha256(Path(path)) != expected:
            raise ValueError('Export checksum mismatch')
    manifest = json.loads((root / 'xert_export.json').read_text('utf-8'))
    if manifest['robot_type'] != 'xlerobot_left' or manifest['backend'] != 'mujoco':
        raise ValueError('This reference protocol is specific to the existing simulated task')
    dataset = LeRobotDataset('xert/local-export', root=root, video_backend='pyav', download_videos=False)
    # LeRobot's own transform avoids the generic HF formatter's removed VideoReader import.
    columns = {key: np.asarray(value) for key, value in dataset.hf_dataset[:].items()
               if key in ('episode_index', 'observation.state', 'action', 'xert.next_observation')}
    episodes, records = [], manifest['source_episodes']
    store = Store(Path(settings['source_store']))
    for index, record in enumerate(records):
        source, current = store.load_training_episode(record['id'])
        if current != record:
            raise ValueError('Source episode metadata differs from the export manifest')
        mask = columns['episode_index'] == index
        observation = np.asarray(columns['observation.state'][mask], dtype=np.float32)
        action = np.asarray(columns['action'][mask], dtype=np.float32)
        successor = np.asarray(columns['xert.next_observation'][mask], dtype=np.float32)
        if not (np.array_equal(observation, source['observations'][:-1])
                and np.array_equal(successor, source['observations'][1:])
                and np.array_equal(action, source['actions'])):
            raise ValueError('LeRobot training arrays differ from verified corrective demonstrations')
        episodes.append({'observations': np.concatenate((observation, successor[-1:])), 'actions': action})
    if sum(len(ep['actions']) for ep in episodes) != len(dataset):
        raise ValueError('Unexpected extra LeRobot rows')
    names = records[0]['simulation']['observation_names']
    if (len(names) != 26 or not all(name.startswith('joint_position.') for name in names[:6])
            or not all(name.startswith('joint_velocity.') for name in names[6:12])):
        raise ValueError('The state/environment partition no longer matches source feature names')
    return episodes, records


def reference(settings, seed):
    root = Path(settings['reference_path'])
    outcomes = json.loads((root / 'outcomes.json').read_text('utf-8'))
    if not outcomes['complete'] or outcomes['chunk_size'] != settings['chunk_size']:
        raise ValueError('Reference experiment is incomplete or uses a different chunk length')
    result = next(item for item in outcomes['outcomes'] if item['train_seed'] == seed)
    model_path = root / 'dataset/models' / result['model_id']
    proof = json.loads((model_path / 'pipeline.json').read_text('utf-8'))
    if sha256(model_path / 'metadata.json') != proof['metadata_sha256']:
        raise ValueError('Reference model metadata checksum mismatch')
    metadata = json.loads((model_path / 'metadata.json').read_text('utf-8'))
    evaluation = json.loads((root / 'dataset/evaluations' / result['evaluation_id'] / 'metadata.json').read_text('utf-8'))
    if evaluation['seeds'] != settings['evaluation_seeds']:
        raise ValueError('Evaluation seed order differs from the reference')
    return result, proof, metadata, evaluation


def normalization(episodes, metadata):
    train, validation = metadata['training_episode_indices'], metadata['validation_episode_indices']
    if set(train) & set(validation) or sorted(train + validation) != list(range(len(episodes))):
        raise ValueError('Reference split does not partition the demonstrations')
    observations = np.concatenate([episodes[index]['observations'][:-1] for index in train])
    mean = observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(observations.std(axis=0, dtype=np.float64), 1e-4).astype(np.float32)
    if not (np.array_equal(mean, np.asarray(metadata['observation_mean'], np.float32))
            and np.array_equal(scale, np.asarray(metadata['observation_scale'], np.float32))):
        raise ValueError('Train-only normalization differs from the reference')
    return mean, scale, train, validation


class RuntimePolicy:
    def __init__(self, policy, mean, scale):
        self.policy, self.mean, self.scale = policy, mean, scale

    def reset(self):
        self.policy.reset()

    def act(self, observation):
        normalized = torch.from_numpy(_normalize(observation, self.mean, self.scale)).unsqueeze(0)
        action = self.policy.select_action(inputs(normalized))[0].detach().numpy()
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError('Official ACT emitted an invalid action')
        return np.clip(action, -1, 1).astype(np.float32)


def fit(episodes, settings, seed, metadata, output):
    from lerobot.policies.act.modeling_act import ACTPolicy
    mean, scale, train_ids, validation_ids = normalization(episodes, metadata)
    torch.manual_seed(seed)
    policy = ACTPolicy(policy_config(settings))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    training = DataLoader(ActionChunks(episodes, train_ids, settings['chunk_size'], mean, scale),
                          batch_size=settings['batch_size'], shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
    validation = DataLoader(ActionChunks(episodes, validation_ids, settings['chunk_size'], mean, scale),
                            batch_size=settings['batch_size'])
    history, best, weights = [], math.inf, None
    for epoch in range(1, settings['epochs'] + 1):
        policy.train()
        for observation, action, valid in training:
            loss, _ = policy({**inputs(observation), 'action': action, 'action_is_pad': ~valid})
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite official ACT training loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        error, count = 0.0, 0
        with torch.inference_mode():
            for observation, action, valid in validation:
                # Official prior inference excludes future actions from the VAE encoder.
                prediction = policy.predict_action_chunk(inputs(observation))
                error += ((prediction - action).abs() * valid.unsqueeze(-1)).sum().item()
                count += valid.sum().item() * action.shape[-1]
        mae = error / count
        if not math.isfinite(mae):
            raise ValueError('Nonfinite official ACT validation error')
        if mae < best:
            best, best_epoch = mae, epoch
            weights = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
        history.append({'epoch': epoch, 'validation_mae': mae})
        print(json.dumps({'phase': 'training', 'seed': seed, **history[-1]}), flush=True)
    policy.load_state_dict(weights)
    policy.eval()
    policy.save_pretrained(output)
    stats = {'observation_mean': mean.tolist(), 'observation_scale': scale.tolist(),
             'training_episode_indices': train_ids, 'validation_episode_indices': validation_ids,
             'best_validation_mae': best, 'best_epoch': best_epoch, 'history': history}
    atomic_json(output / 'training.json', stats)
    # Exercise the same saved official checkpoint that evaluation will load.
    loaded = ACTPolicy.from_pretrained(output, local_files_only=True)
    return RuntimePolicy(loaded, mean, scale), stats


def run(config_path, output):
    from xle_workbench.sim import ManipulationEnv
    settings = json.loads(config_path.read_text('utf-8'))
    seeds = settings['training_seeds']
    if seeds != [0, 1, 2] or settings['epochs'] != 20 or settings['batch_size'] != 128:
        raise ValueError('This runner implements the predeclared P1.4 budget only')
    provenance = ledger.source_provenance()
    episodes, records = load_export(settings)
    fingerprints = [hashlib.sha256(ep['observations'].tobytes() + ep['actions'].tobytes()).hexdigest()
                    for ep in episodes]
    references = {seed: reference(settings, seed) for seed in seeds}
    env = ManipulationEnv(records[0]['config'])
    try:
        if pipeline.contract(env.metadata()) != records[0]['contract_sha256']:
            raise ValueError('Native observation/action contract differs from the demonstrations')
        reset_states = {seed: pipeline.initial_state_hash(env.reset(seed=seed)) for seed in settings['evaluation_seeds']}
        demo_states = {pipeline.initial_state_hash(ep['observations'][0]) for ep in episodes}
        if len(set(reset_states.values())) != len(reset_states) or set(reset_states.values()) & demo_states:
            raise ValueError('Evaluation reset states are repeated or overlap demonstrations')
        for _, proof, metadata, evaluation in references.values():
            if (proof['episode_ids'] != [record['id'] for record in records]
                    or metadata['episode_sha256'] != fingerprints
                    or {row['seed']: row['initial_state_sha256'] for row in evaluation['episodes']} != reset_states):
                raise ValueError('Data or actual reset-state hashes differ from P1.3')
            normalization(episodes, metadata)
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / 'config.json', {**settings, **provenance})
        atomic_json(output / 'provenance.json', {
            'backend': 'mujoco', 'embodiment': 'xlerobot_left', 'simulation': env.metadata(),
            'source_sha256': sha256(Path(__file__)), 'episode_fingerprints': fingerprints,
            'source_episode_ids': [record['id'] for record in records],
            'training_payload_sha256': [record.get('corrective_sha256', record['sha256']) for record in records],
            'reset_state_sha256': reset_states, 'hardware': False,
            'runtime': {key: importlib.metadata.version(key) for key in ('torch', 'numpy', 'lerobot', 'mujoco')},
            'python': platform.python_version(),
            'feature_mapping': {'observation.state': records[0]['simulation']['observation_names'][:12],
                                'observation.environment_state': records[0]['simulation']['observation_names'][12:]},
            'normalization': 'train-only observation mean/std matching P1.3; action identity',
        })
        torch.set_num_threads(1)
        store, outcomes = Store(output / 'dataset'), []
        for seed in seeds:
            original, _, metadata, _ = references[seed]
            model_dir = output / 'models' / str(seed)
            model_dir.mkdir(parents=True)
            policy, stats = fit(episodes, settings, seed, metadata, model_dir)
            model_files = sorted(path for path in model_dir.rglob('*') if path.is_file())
            common = dict(**provenance, slice='P1.4', experiment='p1.4_' + output.name,
                          backend='mujoco', embodiment='xlerobot_left',
                          config_path=ledger.artifact_path(output / 'config.json'))
            ledger.append_row(ledger.new_row(
                **common, metric='validation_action_mae', value=stats['best_validation_mae'], ci95=None,
                n=sum(len(episodes[i]['actions']) for i in stats['validation_episode_indices']),
                seeds={'train': [seed], 'eval': []}, artifact_paths=[ledger.artifact_path(path) for path in model_files],
                notes='Official ACT prior-inference validation MAE over valid chunk positions; not rollout success.'))
            trials = []
            for eval_seed in settings['evaluation_seeds']:
                arrays, outcome = pipeline.rollout(env, eval_seed, policy)
                initial = pipeline.initial_state_hash(arrays['observations'][0])
                if initial != reset_states[eval_seed]:
                    raise ValueError('Evaluation reset state changed after preflight')
                record = store.save_episode(arrays, {
                    'source': 'lerobot_act_policy_evaluation', 'seed': eval_seed, 'config': env.config,
                    'simulation': env.metadata(), 'contract_sha256': pipeline.contract(env.metadata()),
                    'success': bool(outcome.get('success', False)), 'outcome': outcome,
                    'training_seed': seed, 'hardware': False})
                trials.append({'seed': eval_seed, 'episode_id': record['id'], 'success': record['success'],
                               'initial_state_sha256': initial, 'sha256': record['sha256']})
                print(json.dumps({'phase': 'evaluation', 'training_seed': seed, **trials[-1]}), flush=True)
            successes = sum(row['success'] for row in trials)
            evaluation_path = model_dir / 'evaluation.json'
            atomic_json(evaluation_path, {'trials': trials, 'successes': successes,
                                         'held_out': True, 'reference_ledger_id': original['ledger_id']})
            artifacts = [*model_files, evaluation_path, output / 'config.json', output / 'provenance.json']
            artifacts.extend(store.path('episodes', trial['episode_id']) / name
                             for trial in trials for name in ('metadata.json', 'episode.npz'))
            row = ledger.new_row(
                **common, metric='success_rate', value=successes / len(trials),
                ci95=pipeline.wilson(successes, len(trials)), n=len(trials),
                seeds={'train': [seed], 'eval': settings['evaluation_seeds']},
                artifact_paths=[ledger.artifact_path(path) for path in artifacts],
                artifact_sha256={ledger.artifact_path(path): sha256(path) for path in artifacts},
                notes='Official LeRobot ACT on scripted recovery data and privileged simulator state. '
                      'Same actual held-out states and training split as P1.3; different architecture, ensemble weighting '
                      'and dependency stack. Per-model Wilson interval; shared trials are not pooled. No hardware or causal ablation claim.')
            ledger.append_row(row)
            outcomes.append({'train_seed': seed, 'successes': successes, 'ledger_id': row['id'],
                             'reference_ledger_id': original['ledger_id']})
            atomic_json(output / 'outcomes.json', {'outcomes': outcomes, 'complete': len(outcomes) == len(seeds)})
        rows = {row['id']: row for row in ledger.read_rows()}
        lines = ['# Official ACT reference comparison (MuJoCo only)', '',
                 '| Model | Training seed | Successes / trials | Wilson 95% interval | Ledger row |',
                 '|---|---|---|---|---|']
        for outcome in outcomes:
            for label, key in (('XERT ACT-style', 'reference_ledger_id'), ('LeRobot ACT', 'ledger_id')):
                row = rows[outcome[key]]
                lines.append(f"| {label} | {outcome['train_seed']} | {round(row['value'] * row['n'])} / {row['n']} | "
                             f"[{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] | {row['id']} |")
        lines += ['', settings['notes'], '', 'The same evaluation states recur across models; do not pool trials. '
                  'This comparison does not establish optimal training or isolate one architectural change.']
        (output / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/lerobot_act_experiment.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
