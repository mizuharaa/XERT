"""P1.3: fixed-budget validation chunk sweep, then independent held-out evaluations."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from xert.eval import ledger
from xle_workbench import pipeline
from xle_workbench.learning import validate_config
from xle_workbench.storage import Store, atomic_json


def run(config_path, output):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    collection = json.loads(Path(config['collection_path']).read_text(encoding='utf-8'))
    eval_seeds, train_seeds = config['evaluation_seeds'], config['training_seeds']
    if (not 30 <= len(set(eval_seeds)) <= 100 or len(set(train_seeds)) < 3
            or len(set(eval_seeds)) != len(eval_seeds) or len(set(train_seeds)) != len(train_seeds)
            or any(type(seed) is not int or seed < 0 for seed in [*eval_seeds, *train_seeds])
            or set(eval_seeds) & set(collection['demonstration_seeds'])):
        raise ValueError('Require distinct training seeds and at least 30 disjoint held-out initial-state seeds')
    if not config['chunk_sizes'] or len(set(config['chunk_sizes'])) != len(config['chunk_sizes']):
        raise ValueError('Chunk sweep requires distinct lengths')
    for chunk in config['chunk_sizes']:
        validate_config({**config['training'], 'epochs': config['sweep_epochs'],
                         'seed': config['sweep_seed'], 'chunk_size': chunk})
    for seed in train_seeds:
        validate_config({**config['training'], 'seed': seed})
    provenance = ledger.source_provenance()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'config.json', {**config, **provenance})
    # Keep each experiment self-contained; never add models to or mutate source demos.
    store = Store(output / 'dataset')
    shutil.copytree(Path(config['dataset_path']) / 'episodes', store.root / 'episodes')
    ids = collection['episode_ids']
    torch.set_num_threads(1)
    experiment = 'p1.3_' + output.name

    def train(settings, phase):
        proof = pipeline.train_model(store, ids, settings,
                                     progress=lambda record: print(json.dumps(dict(phase=phase, **record)), flush=True))
        model_path = store.path('models', proof['id'])
        metrics = proof['metrics']
        n = sum(store.read('episodes', ids[index])['steps'] for index in metrics['validation_episode_indices'])
        ledger.append_row(ledger.new_row(
            **provenance, slice='P1.3', experiment=experiment, backend='mujoco',
            embodiment='xlerobot_left', metric='validation_action_mae', value=metrics['best_validation_mae'],
            ci95=None, n=n, seeds={'train': [settings['seed']], 'eval': []},
            config_path=ledger.artifact_path(output / 'config.json'),
            artifact_paths=[ledger.artifact_path(model_path / name) for name in ('metadata.json', 'pipeline.json', 'model.pt')],
            phase=phase, chunk_size=settings['chunk_size'], model_id=proof['id'],
            notes='Validation action error, not a held-out task-success estimate; overlapping chunk targets are correlated.'))
        return proof

    sweep = []
    for chunk in config['chunk_sizes']:
        settings = {**config['training'], 'epochs': config['sweep_epochs'],
                    'seed': config['sweep_seed'], 'chunk_size': chunk}
        proof = train(settings, f'sweep_chunk_{chunk}')
        sweep.append(dict(chunk_size=chunk, model_id=proof['id'], validation_mae=proof['metrics']['best_validation_mae']))
    selected = min(sweep, key=lambda row: (row['validation_mae'], row['chunk_size']))['chunk_size']
    atomic_json(output / 'selection.json', dict(chunk_size=selected, selection='lowest validation MAE only', sweep=sweep))
    outcomes = []
    for seed in train_seeds:
        proof = train({**config['training'], 'seed': seed, 'chunk_size': selected}, f'train_seed_{seed}')
        report = pipeline.evaluate(store, eval_seeds, proof['simulation']['config'], proof['id'],
                                   experiment=experiment, slice_id='P1.3',
                                   progress=lambda row: print(json.dumps(dict(phase=f'eval_seed_{seed}', **row)), flush=True))
        if report['held_out'] is not True or report['success_rate_ci95'] is None:
            raise ValueError('Held-out initial-state independence was not established')
        outcomes.append(dict(train_seed=seed, model_id=proof['id'], evaluation_id=report['id'],
                             ledger_id=report['ledger_id'], successes=report['successes'], n=report['trials'],
                             success_rate=report['success_rate'], ci95=report['success_rate_ci95']))
        atomic_json(output / 'outcomes.json', dict(chunk_size=selected, outcomes=outcomes, complete=len(outcomes) == len(train_seeds)))
    rows = {row['id']: row for row in ledger.read_rows()}
    lines = ['# P1.3 held-out MuJoCo evaluation', '',
             'Independent per-training-seed Wilson intervals; the same evaluation seeds are not pooled across models.', '',
             '| Training seed | Successes / trials | Success rate | Wilson 95% interval | Ledger row |',
             '|---|---|---|---|---|']
    for result in outcomes:
        row = rows[result['ledger_id']]
        lines.append(f"| {result['train_seed']} | {result['successes']} / {row['n']} | {row['value']:.4f} | [{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] | {row['id']} |")
    lines.extend(['', 'Scripted noisy demonstrations, privileged simulator state, pinned XLeRobot left arm. '
                  'This is not SO-101 hardware, visual perception, or a human-data quality-gate experiment.', '',
                  'The budget and chunk selection rule were fixed before held-out evaluation; '
                  'these results do not establish convergence or optimal hyperparameters.'])
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(complete=True, chunk_size=selected, outcomes=outcomes)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/act_experiment.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
