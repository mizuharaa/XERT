"""SO-101 dry-run backend: no bus, no motors. Numeric joint maps here are made-up TEST FIXTURES, not measurements."""
import json
from pathlib import Path

import numpy as np
import pytest

from xert.backends.so101_hw_backend import TBD, CommandedState, SO101HardwareBackend, load_joint_map
from xert.safety import SafetyController
from xert.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/hardware/so101_joint_map.yaml"
SPEC = load_spec(ROOT / "xert/spec/so101.yaml")


def fixture_map(tmp_path, **override):
    """TEST FIXTURE: made-up numbers so the conversion can be checked. Not measured on any arm."""
    raw = json.loads(TEMPLATE.read_text())
    for entry in raw["joints"]:
        entry.update({"sign": -1, "units_per_rad": 50.0, "zero_offset": 10.0, "min": -40.0, "max": 60.0,
                      "max_speed": 5.0, "max_accel": 100.0, **override})
    raw["joints"][-1].update({k: TBD for k in ("sign", "zero_offset")})  # gripper stays unmeasured
    path = tmp_path / "fixture_map.json"
    path.write_text(json.dumps(raw))
    return path


def test_motion_mode_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="owner-supervised"):
        SO101HardwareBackend(SPEC, fixture_map(tmp_path), dry_run=False)


def test_tbd_map_is_unmapped_never_guessed():
    with pytest.raises(ValueError, match="TBD"):
        load_joint_map(TEMPLATE, SPEC, dry_run=False)
    lines = []
    backend = SO101HardwareBackend(SPEC, TEMPLATE, echo=lines.append)
    q = np.array([.1, -.2, .3, 0, .05, .7])
    backend.send_targets(q, 0.)
    assert [r["status"] for r in backend.last_sent] == ["unmapped"] * 6
    assert [r["sim_target"] for r in backend.last_sent] == q.tolist()
    assert all("motor_target" not in r for r in backend.last_sent)
    assert "gripper: unmapped (sim +0.7000 rad)" in lines[0] and "dry-run, no hardware" in lines[0]


def test_sign_offset_limits_and_not_measured_state(tmp_path):
    backend = SO101HardwareBackend(SPEC, fixture_map(tmp_path, max_speed=1e6, max_accel=1e9), echo=lambda _: None)
    state = backend.read_state()
    assert isinstance(state, CommandedState) and state.measured is False
    backend.send_targets([.2, -1.2, 0, 0, 0, .5], 0.)
    sent = backend.last_sent
    assert np.isclose(sent[0]["motor_target"], -1 * 50 * .2 + 10) and not sent[0]["clamped"]
    assert sent[1]["motor_target"] == 60.0 and sent[1]["clamped"]  # 70 wanted, above max 60
    assert sent[5]["status"] == "unmapped"
    state = backend.read_state()
    assert state.measured is False and np.isclose(state.q[0], .2) and state.q[5] == .5
    backend.send_targets([1.5, 0, 0, 0, 0, 0], .04)  # -65 wanted, below min -40
    assert backend.last_sent[0]["motor_target"] == -40.0 and backend.last_sent[0]["clamped"]
    assert np.isclose(backend.read_state().q[0], (-40 - 10) / -50)


def test_hardware_clamp_applies_on_top_of_safety_controller(tmp_path):
    backend = SO101HardwareBackend(SPEC, fixture_map(tmp_path), echo=lambda _: None)
    safety = SafetyController(SPEC)
    target = SPEC.rest + .5
    previous = [10.0] * 5  # rest 0 rad -> motor 10 in the fixture
    for tick in range(10):
        t = tick / SPEC.control_hz
        command = safety.step(target, t_now=t, t_capture=t)
        backend.send_targets(command.q, t)
        motors = [r["motor_target"] for r in backend.last_sent[:5]]
        assert all(abs(m - p) <= 5.0 / SPEC.control_hz + 1e-9 for m, p in zip(motors, previous))
        previous = motors
    # The safety layer alone moved further than the hardware speed limit allows.
    assert abs(-50 * command.q[0]) > abs(previous[0] - 10) + 1e-6
    assert backend.last_sent[0]["clamped"]

