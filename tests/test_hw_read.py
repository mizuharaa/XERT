"""`xert hw-read` against a FAKE bus that records every call. No port is opened, no hardware was read.

Ticks, calibration ranges and joint maps here are made-up TEST FIXTURES, not measurements.
"""
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from xert.backends.so101_hw_backend import MOTOR_TABLE
from xert.eval.ledger import ledger_path
from xert.runtime import hw_read

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/hardware/so101_joint_map.yaml"
NAMES = [m for m, _, _ in MOTOR_TABLE]
# Body ticks around mid 2000 of [1000, 3000]; gripper 1500 of [1000, 2000] reads 50.
TICKS = {"shoulder_pan": 2100, "shoulder_lift": 1900, "elbow_flex": 2200, "wrist_flex": 2000, "wrist_roll": 2050,
         "gripper": 1500}


CALIBRATION = {m: {"id": i, "drive_mode": 0, "homing_offset": 0, "range_min": 1000,
                   "range_max": 2000 if m == "gripper" else 3000} for m, i, _ in MOTOR_TABLE}


def calibration_file(tmp_path):
    """FIXTURE LeRobot calibration file named <robot.id>.json (made-up ranges)."""
    path = tmp_path / "fixture_follower.json"
    path.write_text(json.dumps(CALIBRATION))
    return path


def joint_map(tmp_path, name="map.json", **override):
    """FIXTURE kinematic-complete map: degrees -> rad for the body, 60 units/rad for the gripper."""
    raw = json.loads(TEMPLATE.read_text())
    for entry in raw["joints"]:
        gripper = entry["units"] == "range_0_100"
        entry.update({"sign": 1, "units_per_rad": 60.0 if gripper else 180 / np.pi, "zero_offset": 0.0,
                      "min": 0.0 if gripper else -180.0, "max": 100.0 if gripper else 180.0, "max_speed": 90.0,
                      "max_accel": 900.0, **override.get(entry["joint"], {})})
    raw.update(override.get("top", {}))
    path = tmp_path / name
    path.write_text(json.dumps(raw))
    return path


def lerobot_normalize(calibration, ids_values):
    """LeRobot 0.6.1 `MotorsBus._normalize` for drive_mode 0 (motors_bus.py:854-881); range modes clamp first."""
    out = {}
    for motor, id_, units in MOTOR_TABLE:
        if id_ in ids_values:
            lo, hi, value = calibration[motor]["range_min"], calibration[motor]["range_max"], ids_values[id_]
            out[id_] = ((value - (lo + hi) / 2) * 360 / 4095 if units == "degrees" else
                        (min(hi, max(lo, value)) - lo) / (hi - lo) * 100)
    return out


class FakeBus:
    """Records every method call; any other attribute (write, enable_torque, ...) fails the test."""

    def __init__(self, port, calibration, ticks):
        self.port, self.calibration, self.ticks, self.calls = port, calibration, ticks, []

    def connect(self, *args, **kwargs):
        self.calls.append(("connect", args, kwargs))

    def sync_read(self, *args, **kwargs):
        self.calls.append(("sync_read", args, kwargs))
        return dict(self.ticks)

    def _normalize(self, ids_values):
        self.calls.append(("_normalize", (ids_values,), {}))
        return lerobot_normalize(self.calibration, ids_values)

    def disconnect(self, *args, **kwargs):
        self.calls.append(("disconnect", args, kwargs))

    def __getattr__(self, name):
        raise AssertionError(f"hw-read touched bus.{name}")


@pytest.fixture
def buses(monkeypatch):
    made, ticks = [], dict(TICKS)

    def factory(port, calibration):
        made.append(FakeBus(port, calibration, ticks))
        return made[-1]
    monkeypatch.setattr(hw_read, "_bus", factory)
    return SimpleNamespace(made=made, ticks=ticks)


def read(tmp_path, joint_map_path=None, name="read.json"):
    args = SimpleNamespace(port="COM7", calibration=calibration_file(tmp_path), output=tmp_path / name,
                           joint_map=joint_map_path)
    return hw_read.execute(args)


EXPECTED_CALLS = [("connect", (), {}), ("sync_read", ("Present_Position",), {"normalize": False}),
                  ("_normalize", ({i: TICKS[m] for m, i, _ in MOTOR_TABLE},), {}),
                  ("disconnect", (), {"disable_torque": False})]


def test_cli_reads_once_through_exact_calls_and_writes_no_ledger_row(tmp_path, buses, monkeypatch, capsys):
    from xert.__main__ import main
    output = tmp_path / "read.json"
    monkeypatch.setattr(sys, "argv", ["xert", "hw-read", "--port", "COM7", "--calibration",
                                      str(calibration_file(tmp_path)), "--output", str(output),
                                      "--joint-map", str(joint_map(tmp_path))])
    main()
    assert "no motor writes; no ledger rows appended" in capsys.readouterr().out
    assert not ledger_path().exists()
    (bus,) = buses.made
    assert bus.port == "COM7" and bus.calls == EXPECTED_CALLS  # One bus read; normalization is on the same ticks.
    record = json.loads(output.read_text())
    assert record["calibration"]["robot_id"] == "fixture_follower" and len(record["calibration"]["sha256"]) == 64
    assert record["motor_table"] == [{"joint": m, "motor": m, "id": i, "units": u} for m, i, u in MOTOR_TABLE]
    assert record["raw_ticks"] == TICKS and record["saturated"] == [] and record["valid_for_seeding"] is True
    assert record["normalized"]["gripper"] == 50.0 and record["inside_spec_limits"] is True
    assert record["joint_map"]["sha256"] and list(record["radians"]) == NAMES
    np.testing.assert_allclose(record["radians"]["gripper"], 50 / 60, rtol=0, atol=1e-15)


def test_without_a_map_there_are_no_radians(tmp_path, buses):
    record = read(tmp_path)
    assert {"radians", "joint_map", "inside_spec_limits"}.isdisjoint(record) and record["normalized"]


def test_conversion_equals_import_lerobot_on_the_same_values(tmp_path, buses, monkeypatch):
    from xert.data import lerobot_io
    path = joint_map(tmp_path, shoulder_pan={"sign": -1, "zero_offset": 2.5}, gripper={"zero_offset": 9.0})
    record = read(tmp_path, path)
    native = np.array([list(record["normalized"].values())])
    monkeypatch.setattr(lerobot_io, "_columns", lambda raw: {"observation.state": native, "action": native})
    motors = [f"{m}.pos" for m in NAMES]
    info = {"robot_type": "so_follower", "features": {k: {"names": motors} for k in ("observation.state", "action")}}
    _, converted, meta = lerobot_io._hardware(None, info, path, False, "degrees")
    assert meta["units"] == "radians" and converted["state"][0].tolist() == list(record["radians"].values())


def test_out_of_limit_radians_are_written_unclipped(tmp_path, buses):
    record = read(tmp_path, joint_map(tmp_path, shoulder_pan={"zero_offset": -120.0}))
    assert record["inside_spec_limits"] is False and record["valid_for_seeding"] is True
    np.testing.assert_allclose(record["radians"]["shoulder_pan"], (TICKS["shoulder_pan"] - 2000) * 360 / 4095
                               * np.pi / 180 + 120 * np.pi / 180, rtol=0, atol=1e-12)


def test_saturated_gripper_is_not_valid_for_seeding(tmp_path, buses):
    buses.ticks["gripper"] = 2200  # Past range_max 2000: LeRobot reads 100, whose radians would look fine.
    record = read(tmp_path, joint_map(tmp_path))
    assert record["normalized"]["gripper"] == 100.0 and record["inside_spec_limits"] is True
    assert record["saturated"] == ["gripper"] and record["valid_for_seeding"] is False
    assert "gripper" in record["invalid_reason"] and record["raw_ticks"]["gripper"] == 2200


def test_tbd_and_incompatible_maps_are_distinct_refusals_before_the_port_opens(tmp_path, buses):
    with pytest.raises(ValueError, match="TBD"):
        read(tmp_path, TEMPLATE)
    body = {m: {"units": "range_m100_100"} for m in NAMES[:5]}  # Complete, but LeRobot's non-degrees body mode.
    for bad in (body, {"elbow_flex": {"id": 7}}, {"wrist_roll": {"motor": "wrist"}}):
        with pytest.raises(ValueError, match="incompatible"):
            read(tmp_path, joint_map(tmp_path, **bad))
    with pytest.raises(ValueError, match="different calibration"):
        read(tmp_path, joint_map(tmp_path, top={"provenance": {"calibration": {"sha256": "0" * 64}}}))
    assert buses.made == [] and not (tmp_path / "read.json").exists()


def test_motor_set_mismatch_is_refused_and_the_port_still_closes_without_torque_writes(tmp_path, buses):
    del buses.ticks["gripper"]
    with pytest.raises(ValueError, match="Bus returned motors"):
        read(tmp_path)
    assert [c[0] for c in buses.made[0].calls] == ["connect", "sync_read", "disconnect"]
    assert buses.made[0].calls[-1] == ("disconnect", (), {"disable_torque": False})
    assert not (tmp_path / "read.json").exists()


def test_installed_lerobot_bus_matches_the_calls_and_table_hw_read_uses():
    """LeRobot 0.6.1 only (skipped in the base env). Builds the real bus WITHOUT connecting: no port is opened."""
    pytest.importorskip("lerobot.motors.feetech")
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.robots.so_follower import so_follower
    parameters = {name: inspect.signature(getattr(FeetechMotorsBus, name)).parameters
                  for name in ("connect", "sync_read", "disconnect", "_normalize")}
    assert parameters["connect"]["handshake"].default is True
    assert parameters["sync_read"]["normalize"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["disconnect"]["disable_torque"].default is True  # Hence hw-read always passes False.
    assert list(parameters["_normalize"]) == ["self", "ids_values"]
    source = inspect.getsource(so_follower.SOFollower.__init__)
    for motor, id_, units in MOTOR_TABLE:
        mode = "norm_mode_body" if units == "degrees" else "MotorNormMode.RANGE_0_100"
        assert f'"{motor}": Motor({id_}, "sts3215", {mode})' in source
    bus = hw_read._bus("COM_NEVER_OPENED", CALIBRATION)
    assert not bus.is_connected
    for ticks in (TICKS, {**TICKS, "gripper": 2200, "shoulder_pan": 3300}):
        ids = {i: ticks[m] for m, i, _ in MOTOR_TABLE}
        assert bus._normalize(ids) == lerobot_normalize(CALIBRATION, ids)  # The fake mirrors the real formula.


# ---- hw-dry-run --seed-from: the start pose comes from a read, recomputed with the dry-run's own map ----

def seed(read_record, path):
    from xert.runtime.hw_dry_run import seed_from
    from xert.spec import load_spec
    return seed_from(json.dumps(read_record).encode(), load_spec(ROOT / "xert/spec/so101.yaml"), path,
                     path.read_bytes())


def test_cli_dry_run_starts_from_the_recomputed_seed(tmp_path, buses, monkeypatch):
    from test_motion_file import scripted_clip
    from xert.__main__ import main
    read(tmp_path)
    path, output = joint_map(tmp_path), tmp_path / "run"
    expected = seed(json.loads((tmp_path / "read.json").read_text()), path)
    assert not np.allclose(expected, 0)  # Not the rest pose.
    monkeypatch.setattr(sys, "argv", ["xert", "hw-dry-run", "--spec", str(ROOT / "xert/spec/so101.yaml"),
                                      "--map", str(path), "--clip", str(scripted_clip(tmp_path, [(.01, .1)] * 3)),
                                      "--output", str(output), "--seed-from", str(tmp_path / "read.json")])
    main()
    config = json.loads((output / "config.json").read_text())
    assert config["start_pose"]["radians"] == expected.tolist()
    first = json.loads((output / "trace.jsonl").read_text().splitlines()[0])
    assert np.all(np.abs(np.array(first["q"]) - expected) <= 1.0 / 25 + 1e-12)  # One tick from the seed.
    assert not ledger_path().exists()


def test_two_maps_same_calibration_give_different_seeds_and_stale_radians_are_refused(tmp_path, buses):
    a, b = joint_map(tmp_path, "a.json"), joint_map(tmp_path, "b.json", shoulder_pan={"zero_offset": 5.0})
    plain = read(tmp_path, name="plain.json")  # No map: native values only.
    seed_a, seed_b = seed(plain, a), seed(plain, b)
    np.testing.assert_allclose(seed_a[0] - seed_b[0], 5.0 * np.pi / 180, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(seed_a[1:], seed_b[1:])
    with_a = read(tmp_path, a, name="with_a.json")
    with pytest.raises(ValueError, match="different joint map"):
        seed(with_a, b)  # Its radians were computed with map a; never reused under map b.
    tampered = {**with_a, "radians": {m: 0.0 for m in NAMES}}
    np.testing.assert_array_equal(seed(tampered, a), seed_a)  # Saved radians are ignored, always recomputed.


def test_seed_refusals(tmp_path, buses):
    good = read(tmp_path, name="good.json")
    path = joint_map(tmp_path)
    bad_table = {**good, "motor_table": [{**row, "id": row["id"] + 10} for row in good["motor_table"]]}
    with pytest.raises(ValueError, match="different motor table"):
        seed(bad_table, path)
    with pytest.raises(ValueError, match="TBD"):
        seed(good, TEMPLATE)
    with pytest.raises(ValueError, match="different calibration"):
        seed(good, joint_map(tmp_path, "fitted.json", top={"provenance": {"calibration": {"sha256": "0" * 64}}}))
    with pytest.raises(ValueError, match="outside the spec position limits"):
        seed(good, joint_map(tmp_path, "far.json", shoulder_pan={"zero_offset": -120.0}))
    buses.ticks["gripper"] = 2200
    with pytest.raises(ValueError, match="not valid for seeding.*gripper"):
        seed(read(tmp_path, name="saturated.json"), path)


def test_backend_start_pose_keeps_the_motor_limit_check(tmp_path):
    from xert.backends.so101_hw_backend import SO101HardwareBackend
    from xert.spec import load_spec
    spec = load_spec(ROOT / "xert/spec/so101.yaml")
    start = spec.rest + .1
    backend = SO101HardwareBackend(spec, joint_map(tmp_path), initial=start, echo=lambda _: None)
    np.testing.assert_array_equal(backend.read_state().q, start)
    with pytest.raises(ValueError, match="motor limits"):
        SO101HardwareBackend(spec, joint_map(tmp_path, gripper={"min": 50.0}), initial=start, echo=lambda _: None)
