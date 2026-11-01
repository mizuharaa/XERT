"""SO-101 dry-run backend: no bus, no motors. Numeric joint maps here are made-up TEST FIXTURES, not measurements."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_motion_file import scripted_clip
from test_runtime import frame
from xert.backends.so101_hw_backend import TBD, CommandedState, SO101HardwareBackend, load_joint_map
from xert.eval.ledger import ledger_path
from xert.runtime.hw_dry_run import SimClock
from xert.runtime.latency import summarize
from xert.runtime.loop import run_loop
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


# TEST FIXTURE (the QA probe's made-up values): 57.3 units/rad, 30 units/s, 60 units/s^2, limits +/-200.
QA = {"sign": 1, "units_per_rad": 57.3, "zero_offset": 0.0, "min": -200.0, "max": 200.0,
      "max_speed": 30.0, "max_accel": 60.0}


def drive(backend, q, ticks, start=0):
    motors, velocities = [], []
    for k in range(start, start + ticks):
        backend.send_targets(q, k / SPEC.control_hz)
        motors.append(backend.last_sent[0]["motor_target"])
        velocities.append(backend.motor_velocity[0])
    return np.array(motors), np.array(velocities)


class FreshFrames:
    """A new scripted frame every read, stamped on the loop's simulated clock."""

    def __init__(self, clock):
        self.clock, self.n = clock, 0

    def read(self):
        self.n += 1
        return frame(self.clock(), self.n)

    def close(self):
        pass


def loop_run(backend, target, stop_after=None):
    """run_loop through SafetyController in simulated time; returns records and joint-0 motor targets per tick."""
    clock, sent = SimClock(), []
    records = run_loop(FreshFrames(clock), backend, SimpleNamespace(map=lambda f: target), 3, clock=clock,
                       sleep=clock.sleep, on_step=lambda r: sent.append(backend.last_sent[0].get("motor_target")),
                       stop_requested=lambda: len(sent) == stop_after)
    return records, sent


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
    assert np.isclose(sent[1]["motor_target"], 60.0) and sent[1]["clamped"]  # 70 wanted, above max 60
    assert sent[5]["status"] == "unmapped"
    state = backend.read_state()
    assert state.measured is False and np.isclose(state.q[0], .2) and state.q[5] == .5
    backend.send_targets([1.5, 0, 0, 0, 0, 0], .04)  # -65 wanted, below min -40
    assert np.isclose(backend.last_sent[0]["motor_target"], -40.0) and backend.last_sent[0]["clamped"]
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


def test_cli_writes_trace_and_no_ledger_row(tmp_path, monkeypatch, capsys):
    from xert.__main__ import main
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["xert", "hw-dry-run", "--spec", str(ROOT / "xert/spec/so101.yaml"),
                                      "--map", str(TEMPLATE), "--clip",
                                      str(scripted_clip(tmp_path, [(.004 * k, .05 * k) for k in range(1, 9)])),
                                      "--output", str(output)])
    main()
    assert "no ledger rows appended" in capsys.readouterr().out
    assert not ledger_path().exists()
    rows = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    assert rows and all(r["mode"] == "dry-run, no hardware" and r["state_measured"] is False for r in rows)
    assert any(r["phase"] == "live" for r in rows)
    for path in output.iterdir():
        assert "so101_hw" not in path.read_text()


def test_dry_run_never_imports_hardware_modules(tmp_path):
    clip = scripted_clip(tmp_path, [(.01, .1)] * 3)
    code = f"""
import sys
from xert.backends.so101_hw_backend import SO101HardwareBackend
sys.argv = ["xert", "hw-dry-run", "--spec", r"{ROOT / 'xert/spec/so101.yaml'}", "--map", r"{TEMPLATE}",
            "--clip", r"{clip}", "--output", r"{tmp_path / 'run'}"]
from xert.__main__ import main
main()
bad = [m for m in sys.modules if m == "serial" or m.startswith(("serial.", "lerobot.robots", "lerobot.motors"))]
assert not bad, bad
"""
    child = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120,
                           env={**os.environ, "XERT_LEDGER_PATH": str(tmp_path / "ledger.jsonl")})
    assert child.returncode == 0, child.stderr
    assert not (tmp_path / "ledger.jsonl").exists()


def test_step_target_ramps_up_then_brakes_to_target_without_overshoot(tmp_path):
    backend = SO101HardwareBackend(SPEC, fixture_map(tmp_path, **QA), echo=lambda _: None)
    q = SPEC.rest.copy()
    q[0] = 1.0  # 57.3 motor units
    motors, velocities = drive(backend, q, 100)
    assert 0 < velocities[0] < velocities[1] < velocities[5] and np.isclose(velocities.max(), 30)
    assert motors.max() <= 57.3 + 1e-9 and np.all(np.diff(motors) >= -1e-12)
    assert np.isclose(motors[-1], 57.3) and velocities[-1] == 0


def test_speed_change_never_exceeds_accel_limit(tmp_path):
    backend = SO101HardwareBackend(SPEC, fixture_map(tmp_path, **QA), echo=lambda _: None)
    q = SPEC.rest.copy()
    q[0] = 1.0
    out, back = drive(backend, q, 20), drive(backend, -q, 100, start=20)  # reverse while still moving
    motors, velocities = np.r_[out[0], back[0]], np.r_[out[1], back[1]]
    assert np.all(np.abs(np.diff(np.r_[0, velocities])) <= 60 / SPEC.control_hz + 1e-9)
    assert np.all(np.abs(velocities) <= 30 + 1e-9) and motors.min() >= -57.3 - 1e-9


@pytest.mark.xfail(strict=True, reason="P6.2 BLOCKER (a): the backend never learns of the e-stop, so it keeps "
                                       "moving toward SafetyController's held q instead of holding its last value")
def test_estop_holds_last_hardware_command(tmp_path):
    backend = SO101HardwareBackend(SPEC, fixture_map(tmp_path, **QA), echo=lambda _: None)
    target = SPEC.rest.copy()
    target[0] = 1.5
    records, sent = loop_run(backend, target, stop_after=30)
    assert records[-1]["phase"] == "estop" and len(sent) == 31
    assert sent[-1] == sent[-2] and backend.motor_velocity[0] == 0


def test_joint_map_ids_and_motor_names_are_validated():
    for bad, message in (({"id": 0}, "1..253"), ({"id": 254}, "1..253"), ({"id": True}, "1..253"),
                         ({"id": "3"}, "1..253"), ({"motor": ""}, "nonempty"), ({"id": 1}, "distinct")):
        raw = json.loads(TEMPLATE.read_text())
        raw["joints"][2].update(bad)
        with pytest.raises(ValueError, match=message):
            load_joint_map(TEMPLATE, SPEC, dry_run=True, contents=json.dumps(raw))


def test_recorder_and_latency_refuse_commanded_state(tmp_path):
    from xert.data.recorder import RuntimeRecorder
    backend = SO101HardwareBackend(SPEC, TEMPLATE, echo=lambda _: None)
    with pytest.raises(ValueError, match="not measured"):
        RuntimeRecorder(ROOT / "xert/spec/so101.yaml", backend.read_state(), tmp_path / "out", task="x",
                        source={"kind": "scripted"})
    records, _ = loop_run(backend, SPEC.rest + .1)
    assert any(r["latency_sample"] for r in records) and all(r["state_measured"] is False for r in records)
    with pytest.raises(ValueError, match="measured state"):
        summarize(records, require_webcam=False)
