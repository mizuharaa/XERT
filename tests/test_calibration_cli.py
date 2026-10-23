"""Synthetic inputs only: never opens a camera or appends measurement ledger rows."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_runtime import frame
from xert.backends import MuJoCoBackend
from xert.retarget.calibrate import create_calibration, load_profile
from xert.runtime import cli
from xert.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


def setup_cli(tmp_path, monkeypatch, command='latency'):
    spec = tmp_path / 'robot.yaml'
    raw = json.loads((ROOT / 'xert/spec/so101.yaml').read_text())
    raw['model_path'] = str(load_spec(ROOT / 'xert/spec/so101.yaml').model_path)
    spec.write_text(json.dumps(raw))
    profile = tmp_path / 'profile.json'
    profile.write_bytes((ROOT / 'configs/teleop/so101.json').read_bytes())
    model = tmp_path / 'model.task'
    model.write_bytes(b'mock')
    model_hash = hashlib.sha256(b'mock').hexdigest()
    manifest = tmp_path / 'assets/mediapipe/SOURCE.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'sha256': model_hash}))
    backend = MuJoCoBackend(load_spec(spec))
    try:
        calibration = create_calibration(spec, profile, frame(), backend)
    finally:
        backend.close()
    calibration.update(source='webcam', camera=0, model_sha256=model_hash, mirrored=False)
    calibration_path = tmp_path / 'calibration.json'
    calibration_path.write_text(json.dumps(calibration))
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli.time, 'sleep', lambda _: None)
    monkeypatch.setattr(cli.time, 'perf_counter', lambda: 100.)
    return SimpleNamespace(command=command, spec=spec, profile=profile, calibration=calibration_path,
                           model=model, camera=0, output=tmp_path / 'result', duration=.04, timeout=1.)


@pytest.mark.parametrize('key,value', [('gains', ['.3', .3, .15]), ('gains', [True, .3, .15]),
                                     ('axes', [['0', 0, 1], [-1, 0, 0], [0, -1, 0]]),
                                     ('pinch_range', ['.15', .8]), ('hands', ['right_wrist'])])
def test_profile_rejects_values_that_would_fail_after_capture(tmp_path, key, value):
    raw = json.loads((ROOT / 'configs/teleop/so101.json').read_text())
    raw[key] = value
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_profile(path, load_spec(ROOT / 'xert/spec/so101.yaml'))


@pytest.mark.parametrize('size', [None, [], [640], [640, 0], [True, 480], [640., 480]])
def test_bad_calibration_dimensions_fail_before_camera(tmp_path, monkeypatch, size):
    from xert.sources import webcam_mediapipe
    args = setup_cli(tmp_path, monkeypatch)
    raw = json.loads(args.calibration.read_text())
    raw['image_size'] = size
    args.calibration.write_text(json.dumps(raw))
    monkeypatch.setattr(webcam_mediapipe, 'WebcamSource', lambda *a, **k: pytest.fail('Camera opened'))
    with pytest.raises(ValueError, match='image_size'):
        cli.execute(args)
    assert not args.output.exists()


@pytest.mark.parametrize('command', ['calibrate', 'latency'])
def test_cli_uses_and_snapshots_captured_input_bytes(tmp_path, monkeypatch, command):
    from xert import backends
    from xert.eval import ledger
    from xert.runtime import loop, latency
    from xert.sources import webcam_mediapipe
    args = setup_cli(tmp_path, monkeypatch, command)
    spec_bytes, profile_bytes = args.spec.read_bytes(), args.profile.read_bytes()

    def backend(spec):
        # The caller already parsed its inputs; changing paths must not change mapping or evidence.
        args.spec.write_text('changed after initial read')
        args.profile.write_text('changed after initial read')
        return MuJoCoBackend(spec)

    monkeypatch.setattr(backends, 'MuJoCoBackend', backend)
    monkeypatch.setattr(webcam_mediapipe, 'WebcamSource', lambda *a, **k: SimpleNamespace(
        read=lambda: replace(frame(), source='webcam'), close=lambda: None,
        captured=1, inferred=1, overwritten=0))
    monkeypatch.setattr(ledger, 'artifact_path', lambda path: str(path))
    monkeypatch.setattr(ledger, 'source_provenance', lambda: {})
    monkeypatch.setattr(ledger, 'new_row', lambda **kwargs: kwargs)
    rows = []
    monkeypatch.setattr(ledger, 'append_row', rows.append)

    def run(source, backend, mapper, duration, **kwargs):
        assert mapper.profile == json.loads(profile_bytes)
        assert backend.spec.control_hz == json.loads(spec_bytes)['control_hz']
        kwargs['on_step']({'synthetic_test': True})
        return [{}]

    monkeypatch.setattr(loop, 'run_loop', run)
    monkeypatch.setattr(latency, 'summarize', lambda records: {'n': 1, 'p50_ms': 0, 'p95_ms': 0})
    cli.execute(args)
    if command == 'calibrate':
        saved = json.loads(args.output.read_text())
        assert saved['spec_sha256'] == hashlib.sha256(spec_bytes).hexdigest()
        assert saved['profile_sha256'] == hashlib.sha256(profile_bytes).hexdigest()
        assert not rows
    else:
        config = json.loads((args.output / 'config.json').read_text())
        assert Path(config['spec_snapshot']).read_bytes() == spec_bytes
        assert Path(config['profile_snapshot']).read_bytes() == profile_bytes
        assert config['spec_sha256'] == hashlib.sha256(spec_bytes).hexdigest()
        assert config['profile_sha256'] == hashlib.sha256(profile_bytes).hexdigest()
        assert len(rows) == 2
        assert str(args.spec) not in rows[0]['artifact_paths']
        assert str(args.profile) not in rows[0]['artifact_paths']
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                   for path, digest in rows[0]['artifact_sha256'].items())


@pytest.mark.parametrize('command', ['calibrate', 'latency'])
def test_cli_keeps_primary_failure_when_close_also_fails(tmp_path, monkeypatch, command):
    from xert.eval import ledger
    from xert.runtime import loop
    from xert.sources import webcam_mediapipe
    args = setup_cli(tmp_path, monkeypatch, command)
    primary = ValueError('primary operation failure')

    def fail(*args, **kwargs):
        raise primary

    def close():
        raise RuntimeError('worker did not stop')

    monkeypatch.setattr(webcam_mediapipe, 'WebcamSource', lambda *a, **k: SimpleNamespace(read=fail, close=close))
    monkeypatch.setattr(loop, 'run_loop', fail)
    monkeypatch.setattr(ledger, 'artifact_path', lambda path: str(path))
    monkeypatch.setattr(ledger, 'source_provenance', lambda: {})
    with pytest.raises(ValueError, match='primary operation failure') as caught:
        cli.execute(args)
    assert caught.value is primary
    assert any('worker did not stop' in note for note in primary.__notes__)


def test_close_failure_without_primary_is_not_suppressed():
    def close():
        raise RuntimeError('cleanup failed')
    with pytest.raises(RuntimeError, match='cleanup failed'):
        cli._close(SimpleNamespace(close=close))
