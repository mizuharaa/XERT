"""P1.2: collect configured recovery demonstrations and append measured dataset evidence."""
import argparse
import json
from pathlib import Path

from xert.eval import ledger
from xle_workbench.pipeline import collect, wilson
from xle_workbench.storage import Store, atomic_json


def run(config_path, output):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if type(config.get('action_noise_std')) not in (int, float) or not 0 < config['action_noise_std'] <= .05:
        raise ValueError('Recovery collection requires positive noise up to .05')
    seeds = list(range(config['seed_start'], config['seed_start'] + config['episodes']))
    held_out = set(range(16)) | set(range(10000, 10030))
    held_out.update(s for lo, hi in config['held_out_seed_ranges'] for s in range(lo, hi + 1))
    if not 100 <= len(seeds) <= 200 or set(seeds) & held_out:
        raise ValueError('Collect 100..200 demonstrations disjoint from the declared held-out seeds')
    provenance = ledger.source_provenance()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'config.json', {**config, **provenance})
    store = Store(output / 'dataset')
    identifiers = []
    for start in range(0, len(seeds), 100):
        result = collect(store, seeds[start:start+100], config['environment'],
                         action_noise_std=config['action_noise_std'],
                         progress=lambda item: print(json.dumps(item), flush=True))
        identifiers.extend(result['episode_ids'])
    records = [store.read('episodes', identifier) for identifier in identifiers]
    summary = dict(episode_ids=identifiers, demonstration_seeds=seeds, episodes=len(records),
                   successes=sum(record['success'] for record in records),
                   frames=sum(record['steps'] for record in records),
                   hardware=False, source='scripted_simulation_noise', **provenance)
    atomic_json(output / 'collection.json', summary)
    artifacts = [output / 'collection.json']
    artifacts.extend(store.path('episodes', identifier) / name for identifier in identifiers
                     for name in ('metadata.json', 'episode.npz', 'corrective.npz'))
    common = dict(**provenance, slice='P1.2', experiment=output.name, backend='mujoco',
                  embodiment='xlerobot_left', seeds={'train': [], 'eval': []},
                  demonstration_seeds=seeds, config_path=ledger.artifact_path(output / 'config.json'),
                  artifact_paths=[ledger.artifact_path(path) for path in artifacts],
                  notes='Scripted noisy MuJoCo expert collection, all outcomes retained. Not human demonstrations or a learned-policy evaluation.')
    for metric, value, n, ci in [('dataset_episodes', len(records), len(records), None),
                                ('dataset_frames', summary['frames'], len(records), None),
                                ('success_rate', summary['successes'] / len(records), len(records),
                                 wilson(summary['successes'], len(records)))]:
        ledger.append_row(ledger.new_row(**common, metric=metric, value=value, n=n, ci95=ci))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/recovery_demos.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
