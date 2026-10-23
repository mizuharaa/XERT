"""Scripted landmarks exercise software contracts; these are not human demonstrations."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xert.backends import MuJoCoBackend
from xert.retarget.calibrate import CalibratedMapper, create_calibration, hand_features
from xert.runtime.latency import summarize
from xert.runtime.loop import run_loop
from xert.safety import SafetyController
from xert.sources import HandFrame, MotionFrame
from xert.sources.webcam_mediapipe import decode_result
from xert.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


def scripted_hand(shift=0, curl=0):
    points = np.zeros((21, 3))
    for start, x in ((1, .045), (5, .025), (9, 0), (13, -.02), (17, -.04)):
        for j in range(4):
            points[start + j] = [x, .04 + j * .025, curl * max(0, j - 1) * .02]
    image = points * 2 + [.5 + shift, .3, 0]
    return HandFrame(image, points, .9)


def frame(t=100., sequence=0, **kwargs):
    return MotionFrame(sequence, t, t, {"left": scripted_hand(**kwargs), "right": scripted_hand(**kwargs)},
                       "scripted", (640, 480))


@pytest.mark.parametrize("robot", ["so101", "g1_upper", "leap_hand"])
def test_calibration_and_data_driven_mapping(robot):
    spec_path = ROOT / f"xert/spec/{robot}.yaml"
    profile = ROOT / f"configs/teleop/{robot}.json"
    backend = MuJoCoBackend(load_spec(spec_path))
    try:
        neutral = create_calibration(spec_path, profile, frame(), backend)
        mapper = CalibratedMapper(spec_path, profile, neutral, backend)
        initial = mapper.map(frame())
        moved = mapper.map(frame(shift=.1, curl=.5))
        assert np.isfinite(moved).all()
        assert not np.allclose(initial, moved)
        assert mapper.map(replace(frame(), hands={})) is None
        with pytest.raises(ValueError, match="dimensions"):
            mapper.map(replace(frame(), image_size=(320, 240)))
        with pytest.raises(ValueError, match="checksum"):
            CalibratedMapper(spec_path, profile, {**neutral, "spec_sha256": "bad"}, backend)
    finally:
        backend.close()


def test_hand_geometry_and_confidence_semantics():
    hand = scripted_hand()
    features = hand_features(hand)
    assert np.allclose(features["basis"].T @ features["basis"], np.eye(3))
    invalid = HandFrame(np.zeros((21, 3)), np.zeros((21, 3)), .9)
    with pytest.raises(ValueError, match="Degenerate"):
        hand_features(invalid)
    landmarks = [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in hand.image]
    categories = [SimpleNamespace(category_name="Right", score=.8)]
    result = SimpleNamespace(handedness=[categories], hand_landmarks=[landmarks], hand_world_landmarks=[landmarks])
    assert decode_result(result)["right"].handedness_score == .8
    result.handedness *= 2
    result.hand_landmarks *= 2
    result.hand_world_landmarks *= 2
    assert decode_result(result) == {}  # ambiguous identity never drives a robot
    with pytest.raises(ValueError):
        replace(frame(), t_landmarks=99)


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class Source:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.closed = False

    def read(self):
        return next(self.frames, None)

    def close(self):
        self.closed = True


def test_real_watchdog_clock_ages_without_changing_nominal_dt():
    spec = load_spec(ROOT / "xert/spec/so101.yaml")
    safety = SafetyController(spec)
    safety.step(spec.rest, t_now=0, t_capture=100, watchdog_now=100)
    result = safety.step(None, t_now=.04, watchdog_now=101)
    assert result.phase == "return"  # not fresh merely because only one nominal tick passed
    with pytest.raises(ValueError, match="monotonic"):
        safety.step(None, t_now=.08, watchdog_now=100.9)
    with pytest.raises(ValueError, match="fixed-rate"):
        safety.step(None, t_now=.09, watchdog_now=102)


def test_runtime_unique_samples_loss_and_poststep_state():
    backend = MuJoCoBackend(load_spec(ROOT / "xert/spec/so101.yaml"))
    source = Source([frame(), None, replace(frame(100.08, 1), hands={})])
    mapper = SimpleNamespace(map=lambda f: backend.spec.rest.copy() if f.hands else None)
    clock, callbacks = Clock(), []
    records = run_loop(source, backend, mapper, .4, clock=clock, sleep=clock.sleep, on_step=callbacks.append)
    assert source.closed and backend.closed
    assert records == callbacks
    assert len([r for r in records if r["latency_sample"]]) == 1
    assert records[-1]["phase"] == "hold"
    assert records[-1]["tracking_confidence"] is None
    assert records[0]["t_sim"] > records[0]["t_control"]
    assert all(not row["valid_motion"] for row in records[2:])
    assert summarize(records, require_webcam=False)["n"] == 1
    with pytest.raises(ValueError, match="webcam"):
        summarize(records)
    with pytest.raises(ValueError, match="No valid"):
        summarize(records[2:])


def test_overrun_and_bad_source_close_without_late_command():
    for future in (False, True):
        backend = MuJoCoBackend(load_spec(ROOT / "xert/spec/so101.yaml"))
        source = Source([frame(101 if future else 100)])
        clock = Clock()
        def map_frame(_):
            clock.now += .08
            return backend.spec.rest.copy()
        with pytest.raises((RuntimeError, ValueError)):
            run_loop(source, backend, SimpleNamespace(map=map_frame), .1, clock=clock, sleep=clock.sleep)
        assert source.closed and backend.closed
        assert backend.data.time == 0


def test_estop_callback_latches_hold():
    backend = MuJoCoBackend(load_spec(ROOT / "xert/spec/so101.yaml"))
    source, clock = Source([frame()]), Clock()
    records = run_loop(source, backend, SimpleNamespace(map=lambda _: backend.spec.rest + .1), .4,
                       clock=clock, sleep=clock.sleep, stop_requested=lambda: True)
    assert len(records) == 1
    assert records[0]["phase"] == "estop" and not records[0]["latency_sample"]


@pytest.mark.parametrize("bad_duration", [False, True])
def test_cleanup_preserves_first_failure_and_attempts_both_resources(bad_duration):
    backend = MuJoCoBackend(load_spec(ROOT / "xert/spec/so101.yaml"))
    source, clock = Source([]), Clock()
    close_backend = backend.close
    def bad_source_close():
        source.closed = True
        raise RuntimeError("source cleanup failed")
    def bad_backend_close():
        close_backend()
        raise RuntimeError("backend cleanup failed")
    source.close, backend.close = bad_source_close, bad_backend_close
    expected, message = (ValueError, "Duration") if bad_duration else (RuntimeError, "source cleanup")
    with pytest.raises(expected, match=message) as caught:
        run_loop(source, backend, None, -1 if bad_duration else .04, clock=clock, sleep=clock.sleep)
    assert source.closed and backend.closed
    assert any("backend cleanup failed" in note for note in caught.value.__notes__)
    if bad_duration:
        assert any("source cleanup failed" in note for note in caught.value.__notes__)


def test_latency_snapshots_same_calibration_bytes_that_mapper_received(tmp_path, monkeypatch):
    from xert.eval import ledger
    from xert.retarget import calibrate
    from xert.runtime import cli, loop
    from xert.sources import webcam_mediapipe
    spec_path = ROOT / "xert/spec/so101.yaml"
    profile = ROOT / "configs/teleop/so101.json"
    model_path = tmp_path / "model.task"
    model_path.write_bytes(b"mock")
    model_hash = hashlib.sha256(b"mock").hexdigest()
    manifest = tmp_path / "assets/mediapipe/SOURCE.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"sha256": model_hash}))
    backend = MuJoCoBackend(load_spec(spec_path))
    neutral = create_calibration(spec_path, profile, frame(), backend)
    backend.close()
    neutral.update(source="webcam", camera=0, model_sha256=model_hash, mirrored=False)
    calibration_path = tmp_path / "neutral.json"
    original = json.dumps(neutral, indent=3).encode()
    calibration_path.write_bytes(original)
    def mapper(spec, profile, calibration, backend):
        assert calibration == neutral
        calibration_path.write_bytes(b"changed after the single read")
        return None
    monkeypatch.setattr(calibrate, "CalibratedMapper", mapper)
    monkeypatch.setattr(webcam_mediapipe, "WebcamSource", lambda *a, **k: Source([]))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(ledger, "artifact_path", lambda p: str(p))
    monkeypatch.setattr(ledger, "source_provenance", lambda: {})
    monkeypatch.setattr(loop, "run_loop", lambda *a, **k: [])
    args = SimpleNamespace(command="latency", spec=spec_path, profile=profile, model=model_path,
                           camera=0, calibration=calibration_path, output=tmp_path / "run", duration=.04)
    with pytest.raises(RuntimeError, match="stopped early"):
        cli.execute(args)
    config = json.loads((args.output / "config.json").read_text())
    assert (args.output / "calibration.json").read_bytes() == original
    assert config["calibration_sha256"] == hashlib.sha256(original).hexdigest()
    assert Path(config["calibration"]).read_bytes() == original
