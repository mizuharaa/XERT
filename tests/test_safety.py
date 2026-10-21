from pathlib import Path

import numpy as np
import pytest

from xert.safety import SafetyController
from xert.spec import load_spec


def spec():
    return load_spec(Path(__file__).resolve().parents[1] / "xert/spec/so101.yaml")


def test_limits_including_braking_at_position_boundaries():
    embodiment = spec()
    controller = SafetyController(embodiment)
    previous, velocity = controller.q.copy(), controller.velocity.copy()
    for tick in range(400):
        target = np.full(len(previous), 100 if tick < 200 else -100)
        result = controller.step(target, t_now=tick * controller.dt, t_capture=tick * controller.dt)
        new_velocity = (result.q - previous) / controller.dt
        assert np.all(result.q >= embodiment.position_limits[:, 0] - 1e-10)
        assert np.all(result.q <= embodiment.position_limits[:, 1] + 1e-10)
        assert np.all(np.abs(new_velocity) <= embodiment.velocity_limits + 1e-9)
        assert np.all(np.abs(new_velocity - velocity) <= embodiment.acceleration_limits * controller.dt + 1e-9)
        assert result.clamped and any(e["kind"] == "clamp" for e in result.events)
        previous, velocity = result.q, new_velocity


def test_watchdog_hold_then_rest_and_estop_latched():
    embodiment = spec()
    initial = np.full(len(embodiment.rest), .15)
    controller = SafetyController(embodiment, initial, stale_s=.08, hold_s=.12, return_s=.4)
    phases = []
    for tick in range(100):
        result = controller.step(initial if tick == 0 else None, t_now=tick * controller.dt,
                                 t_capture=0 if tick == 0 else None)
        phases.append(result.phase)
    assert {"live", "hold", "return", "rest"} <= set(phases)
    np.testing.assert_allclose(result.q, embodiment.rest, atol=1e-5)
    stopped = controller.step(initial, t_now=100 * controller.dt, t_capture=100 * controller.dt, estop=True)
    again = controller.step(initial, t_now=101 * controller.dt, t_capture=101 * controller.dt)
    assert stopped.phase == again.phase == "estop"
    np.testing.assert_array_equal(stopped.q, again.q)
    key_controller = SafetyController(embodiment)
    assert not key_controller.handle_key("x")
    assert key_controller.handle_key("Escape")
    assert key_controller.step(initial, t_now=0, t_capture=0).phase == "estop"


def test_reject_bad_stream_without_mutating_commands():
    controller = SafetyController(spec())
    original = controller.q.copy()
    with pytest.raises(ValueError):
        controller.step([float("nan")] * len(original), t_now=0, t_capture=0)
    with pytest.raises(ValueError):
        controller.step(original, t_now=0, t_capture=1)
    controller.step(original, t_now=0, t_capture=0)
    with pytest.raises(ValueError, match="clock"):
        controller.step(original, t_now=.1, t_capture=.1)
    np.testing.assert_array_equal(controller.q, original)
