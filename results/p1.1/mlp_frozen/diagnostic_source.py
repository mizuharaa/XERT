"""P1.1: in-sample overfit, teacher-forced actions, and actual simulation rollouts.

Run with python -m scripts.diagnose_learning --output results/p1.1/mlp.
This intentionally uses one episode, outside the production held-out trainer.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from xert.eval import ledger
from xle_workbench import learning
from xle_workbench.pipeline import rollout
from xle_workbench.sim import JOINTS, ManipulationEnv


def plot_errors(path, errors):
    """One SVG panel per joint; values are normalized target errors, not radians."""
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="780">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="40" y="24">Teacher-forced predicted minus recorded action (normalized target units)</text>']
    bound = max(float(np.abs(errors).max()), 1e-6)
    for j, name in enumerate(JOINTS):
        y = 90 + 115 * j
        points = ' '.join(f'{60 + t * 900 / max(len(errors)-1, 1):.2f},{y - value * 42 / bound:.2f}'
                          for t, value in enumerate(errors[:, j]))
        parts.extend([f'<text x="60" y="{y-48}">{name}; shared y range +/-{bound:.6g}</text>',
                      f'<path d="M60 {y}H960" stroke="#aaa"/>',
                      f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="1"/>'])
    parts.append('<text x="60" y="775">x: control tick; simulation only, one training episode</text></svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def run(args):
    if args.epochs < 1 or args.epochs > 10000 or args.seed < 0:
        raise ValueError('Positive epochs up to 10000 and nonnegative seed required')
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = ledger.source_provenance()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    env = ManipulationEnv()
    episode, expert_info = rollout(env, args.seed)
    assert expert_info['success'], 'Diagnostic requires a successful expert episode'
    observations, actions = episode['observations'][:-1], episode['actions']
    mean = observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(observations.std(axis=0, dtype=np.float64), args.scale_floor).astype(np.float32)
    cfg = learning.validate_config(dict(model_type=args.model, chunk_size=args.chunk_size,
                                       epochs=min(args.epochs, 2000), hidden_dim=64, seed=args.seed))
    config = dict(**vars(args), environment=env.config, training=cfg,
                  in_sample=True, held_out=False, normalization_floor=args.scale_floor,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **provenance)
    config['output'] = str(args.output)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    dataset = learning.ActionChunks([episode], [0], cfg['chunk_size'], mean, scale)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
                                        generator=torch.Generator().manual_seed(args.seed))
    model = learning._model(cfg, observations.shape[1], actions.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for epoch in range(args.epochs):
        model.train()
        for obs, targets, valid in loader:
            prediction, kl = model(obs, targets, valid)
            loss = learning.masked_loss(prediction, targets, valid,
                                        squared=args.model == 'baseline_mlp') + cfg['kl_weight'] * kl
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        if (epoch+1) % 100 == 0:
            print(f'epoch={epoch+1} last_batch_loss={loss.item():.6g}', flush=True)
    metadata = dict(obs_dim=observations.shape[1], observation_mean=mean.tolist(),
                    observation_scale=scale.tolist(), config=cfg)
    class DiagnosticPolicy(learning.LoadedPolicy):
        def act(self, observation):
            observed = np.asarray(observation).copy()
            if args.freeze_constant:
                observed[scale <= args.scale_floor] = mean[scale <= args.scale_floor]
            return super().act(observed)

    policy = DiagnosticPolicy(model, metadata)
    teacher = np.asarray([policy.act(obs) for obs in observations])
    errors = teacher - actions
    closed, closed_info = rollout(env, args.seed, policy)
    env.reset(args.seed)
    replay_info = {}
    replay_states = []
    for action in teacher:
        obs, _, terminal, replay_info = env.step(action)
        replay_states.append(obs)
        if terminal:
            break
    normalized_closed = learning._normalize(closed['observations'][:-1], mean, scale)
    summary = dict(expert_success=bool(expert_info['success']),
                   closed_loop_success=bool(closed_info['success']),
                   closed_loop_outcome=closed_info, predicted_replay_success=bool(replay_info['success']),
                   predicted_replay_outcome=replay_info, action_mae=float(np.abs(errors).mean()),
                   per_joint_mae=np.abs(errors).mean(axis=0).tolist(),
                   max_normalized_closed_observation=float(np.abs(normalized_closed).max()),
                   max_normalized_feature=env.metadata()['observation_names'][np.unravel_index(np.abs(normalized_closed).argmax(), normalized_closed.shape)[1]],
                   gripper_recorded_values=np.unique(actions[:, -1]).tolist(),
                   gripper_mae=float(np.abs(errors[:, -1]).mean()),
                   time_feature='step / configured horizon; identical horizon in all diagnostic rollouts',
                   training_frames=len(actions), closed_loop_frames=len(closed['actions']),
                   in_sample=True, held_out=False, hardware=False)
    np.savez_compressed(args.output / 'traces.npz', **episode, teacher_actions=teacher,
                        closed_observations=closed['observations'], closed_actions=closed['actions'],
                        predicted_replay_observations=np.asarray(replay_states), mean=mean, scale=scale)
    torch.save(model.state_dict(), args.output / 'model.pt')
    plot_errors(args.output / 'joint_errors.svg', errors)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    common = dict(**provenance, slice='P1.1', experiment=args.output.name, backend='mujoco',
                  embodiment='xlerobot_left', ci95=None, seeds={'train':[args.seed], 'eval':[args.seed]},
                  config_path=ledger.artifact_path(args.output / 'config.json'),
                  artifact_paths=[ledger.artifact_path(args.output / name) for name in ('summary.json', 'traces.npz', 'joint_errors.svg', 'model.pt')],
                  notes='Single training episode diagnostic, not held-out success or hardware evidence. Training and evaluation seed intentionally overlap.')
    for metric, value, n in [('success_rate', float(summary['closed_loop_success']), 1),
                             ('predicted_action_replay_success', float(summary['predicted_replay_success']), 1),
                             ('action_mae', summary['action_mae'], len(actions)),
                             ('max_normalized_closed_observation', summary['max_normalized_closed_observation'], len(closed['actions']))]:
        ledger.append_row(ledger.new_row(**common, metric=metric, value=value, n=n))
    print(json.dumps(summary, indent=2), flush=True)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['baseline_mlp', 'act'], default='baseline_mlp')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--chunk-size', type=int, default=8)
    parser.add_argument('--scale-floor', type=float, default=1e-4)
    parser.add_argument('--freeze-constant', action='store_true',
                        help='Controlled probe: ignore features at the training scale floor')
    args = parser.parse_args()
    if not np.isfinite(args.scale_floor) or args.scale_floor < 1e-4:
        parser.error('scale floor must be finite and at least 1e-4')
    run(args)
