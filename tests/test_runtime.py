"""Scripted landmarks exercise software contracts; these are not human demonstrations."""
from dataclasses import replace
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
