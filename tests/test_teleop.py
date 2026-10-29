"""Scripted hands, scripted clock and scripted keys only: no camera, no human data, no ledger rows."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.record_scripted_runtime import ScriptedClock, ScriptedSource, scripted_frame
from xert.backends import MuJoCoBackend
from xert.retarget.calibrate import CalibratedMapper, create_calibration
from xert.runtime import teleop
from xert.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "xert/spec/xlerobot_left.yaml"
PROFILE = ROOT / "configs/teleop/so101.json"
SCRIPTED = {"kind": "scripted", "human_demonstrations": False, "generator": "tests/test_teleop.py"}


def test_key_mapping():
    assert [teleop.key_action(k, recording=False) for k in (" ", "d", "\x1b", "x", None)] == \
        ["start", None, "quit", None, None]
    assert [teleop.key_action(k, recording=True) for k in (" ", "d", "D", "\x1b", "x", None)] == \
        ["stop", "discard", "discard", "quit", None, None]


def calibration(clock):
    backend = MuJoCoBackend(load_spec(SPEC))  # What `xert calibrate --spec xlerobot_left.yaml` uses.
    try:
        return create_calibration(SPEC, PROFILE, scripted_frame(0, clock()), backend)
    finally:
        backend.close()


def session(tmp_path, keys, make_mapper=None):
    clock = ScriptedClock()
    neutral = calibration(clock)
    make_mapper = make_mapper or (lambda backend: CalibratedMapper(SPEC, PROFILE, neutral, backend))
    pending = iter(keys)
    kept = teleop.run_session(ScriptedSource(clock), SPEC, tmp_path / "session", make_mapper, provenance=SCRIPTED,
                              keys=lambda: next(pending, "\x1b"), clock=clock, sleep=clock.sleep)
    return kept, tmp_path / "session"


def test_discard_writes_nothing_and_estop_never_publishes(tmp_path):
    pytest.importorskip("lerobot")
    kept, output = session(tmp_path, [" ", None, "d", " ", None, "\x1b"])
    assert kept == []
    staging = [path for path in output.iterdir()]
    assert len(staging) == 1 and staging[0].name.startswith(".xert-recording-")  # Escape: recoverable journal.
    assert json.loads((staging[0] / "dataset/xert_recording.json").read_text())["status"] == "aborted"


def test_stop_finalises_episode_that_round_trips(tmp_path):
    pytest.importorskip("lerobot")
    from xert.data.recorder import load_runtime_dataset, verify_recording
    kept, output = session(tmp_path, [" ", None, None, " ", " ", " "])
    assert [path.name for path in kept] == ["episode_000000", "episode_000001"]
    assert sorted(path.name for path in output.iterdir()) == ["episode_000000", "episode_000001"]
    assert verify_recording(kept[0])["frames"] == 3 and verify_recording(kept[1])["frames"] == 1
    manifest = json.loads((kept[0] / "xert_recording.json").read_text())
    assert manifest["source"]["kind"] == "scripted" and manifest["source"]["bench_seed"] == 0
    assert manifest["task_outcome"] == {"success": False, "failure_reason": None, "ended_by": "operator", "steps": 3}
    dataset = load_runtime_dataset(kept[0])
    assert dataset.features["next.success"]["dtype"] == "bool" and len(dataset) == 3
    assert not any(dataset[i]["next.success"].item() for i in range(3))
    del dataset


def test_bench_contact_success_reaches_next_success(tmp_path):
    pytest.importorskip("lerobot")
    from xert.data.recorder import load_runtime_dataset

    def expert(backend):
        # Scripted IK expert of the bench, still passed through the runtime safety clamps.
        limits = backend.env.limits
        return SimpleNamespace(map=lambda frame: limits[:, 0] + (np.asarray(backend.env.expert_action()) + 1) / 2 *
                               (limits[:, 1] - limits[:, 0]))

    kept, _ = session(tmp_path, [" "] + [None] * 600, expert)
    manifest = json.loads((kept[0] / "xert_recording.json").read_text())
    assert manifest["task_outcome"]["success"] and manifest["task_outcome"]["ended_by"] == "bench"
    dataset = load_runtime_dataset(kept[0])
    flags = [dataset[i]["next.success"].item() for i in range(len(dataset))]
    assert flags[-1] and not any(flags[:-1]) and len(flags) == manifest["task_outcome"]["steps"]
    del dataset


def test_cli_checks_inputs_before_camera_and_labels_webcam(tmp_path, monkeypatch):
    from xert.runtime import cli
    from xert.sources import webcam_mediapipe
    model = tmp_path / "model.task"
    model.write_bytes(b"mock")
    manifest = tmp_path / "assets/mediapipe/SOURCE.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"sha256": hashlib.sha256(b"mock").hexdigest()}))
    neutral = calibration(ScriptedClock())
    neutral.update(source="webcam", camera=0, model_sha256=hashlib.sha256(b"mock").hexdigest(), mirrored=True)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(neutral))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    args = SimpleNamespace(command="teleop", spec=SPEC, profile=PROFILE, model=model, camera=0,
                           calibration=path, output=tmp_path / "out")
    monkeypatch.setattr(webcam_mediapipe, "WebcamSource", lambda *a, **k: pytest.fail("Camera opened"))
    with pytest.raises(ValueError, match="unmirrored"):
        cli.execute(args)
    path.write_text(json.dumps({**neutral, "mirrored": False}))
    closed, seen = [], {}
    monkeypatch.setattr(webcam_mediapipe, "WebcamSource", lambda *a, **k: SimpleNamespace(
        read=lambda: replace(scripted_frame(1, 100.), source="webcam"), close=lambda: closed.append(True)))
    monkeypatch.setattr(teleop, "run_session", lambda source, spec, output, make_mapper, **kw: seen.update(kw))
    cli.execute(args)
    assert closed and seen["provenance"]["kind"] == "webcam" and seen["provenance"]["mirrored"] is False
    assert seen["provenance"]["calibration_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
