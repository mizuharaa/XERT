"""Append explicit completion corrections for the historical P1.1 truncated replays."""
import json
from xert.eval import ledger


def main():
    rows = ledger.read_rows()
    corrected = {row.get('superseded_id') for row in rows}
    provenance = ledger.source_provenance()
    for old in rows:
        if (old['slice'] != 'P1.1' or old['metric'] != 'predicted_action_replay_success'
                or old['id'] in corrected):
            continue
        summary_path = next(path for path in old['artifact_paths'] if path.endswith('/summary.json'))
        summary = json.loads((ledger.REPO_ROOT / summary_path).read_text(encoding='utf-8'))
        outcome = summary['predicted_replay_outcome']
        if outcome['success'] or outcome['failure_reason'] is not None:
            continue
        fields = {key: value for key, value in old.items()
                  if key not in ('id', 'time_utc', 'git_commit', 'git_dirty', 'metric', 'value', 'notes')}
        fields['artifact_paths'] = [*fields['artifact_paths'], 'scripts/clarify_replay_evidence.py']
        ledger.append_row(ledger.new_row(
            **fields, **provenance, metric='replay_completed', value=0,
            superseded_id=old['id'], measurement_git_commit=old['git_commit'],
            notes='Completion clarification from the original saved outcome; no new simulation rollout. '
                  'Replay stopped at demonstration length without success or a terminal failure reason. '
                  'The superseded zero was not a completed-trial failure or a valid success-rate estimate.'))


if __name__ == '__main__':
    main()
