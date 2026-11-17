"""`xert hw-read`: ONE read-only SO-101 position read through LeRobot 0.6.1's bare `FeetechMotorsBus`.

Bus calls, exactly: `connect()` (opens the port; the Feetech handshake pings every id and reads firmware
versions, motors_bus.py:514-541, feetech.py:145-157), `sync_read("Present_Position", normalize=False)` and
`disconnect(disable_torque=False)` (closes the port, no torque write, motors_bus.py:547-562). Never `SOFollower`,
`configure`, `write` or `enable_torque`: nothing here can energize or move a motor. Writes read.json only;
no ledger row. Imported by the hw-read command only.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUS_CALLS = ["connect()", 'sync_read("Present_Position", normalize=False)', "disconnect(disable_torque=False)"]


def _bus(port, calibration):
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    from xert.backends.so101_hw_backend import MOTOR_TABLE
    return FeetechMotorsBus(port, {m: Motor(i, "sts3215", MotorNormMode(u)) for m, i, u in MOTOR_TABLE},
                            calibration={m: MotorCalibration(**v) for m, v in calibration.items()})


def execute(args):
    from xert.backends.so101_hw_backend import MOTOR_TABLE, compatible_map, to_radians
    from xert.data.lerobot_io import _calibration
    from xert.spec import load_spec
    if args.output.exists():
        raise ValueError("Output already exists; choose a fresh output path")
    provenance = _calibration(args.calibration)  # Refuses anything but a LeRobot so101 follower calibration file.
    calibration_bytes = args.calibration.read_bytes()
    if hashlib.sha256(calibration_bytes).hexdigest() != provenance["sha256"]:
        raise ValueError("Calibration file changed while it was being read")
    calibration = json.loads(calibration_bytes)
    spec = load_spec(ROOT / "xert/spec/so101.yaml")
    # The map is checked before the port opens: a refused map never touches the bus.
    entries, map_sha = (None, None) if args.joint_map is None else compatible_map(
        args.joint_map, spec, provenance["sha256"])
    names = [m for m, _, _ in MOTOR_TABLE]
    bus = _bus(args.port, calibration)
    bus.connect()
    try:
        ticks = bus.sync_read("Present_Position", normalize=False)
        if sorted(ticks) != sorted(names):
            raise ValueError(f"Bus returned motors {sorted(ticks)}, expected {names}")
        # LeRobot's own normalization (motors_bus.py:854-881) on the SAME ticks: one acquisition, no second read.
        normalized = bus._normalize({i: ticks[m] for m, i, _ in MOTOR_TABLE})
    finally:
        bus.disconnect(disable_torque=False)
    # LeRobot clamps range-mode readings to the calibration interval before normalizing (motors_bus.py:867-873),
    # so a tick past either bound reads as a plausible value that misrepresents the pose. At the bound the clamp
    # changes nothing (owner default: strictly beyond counts; `<=`/`>=` would also refuse a gripper resting there).
    saturated = [m for m in names if ticks[m] < calibration[m]["range_min"] or ticks[m] > calibration[m]["range_max"]]
    record = {"version": 1, "mode": "read-only position read", "port": args.port,
              "time_utc": datetime.now(timezone.utc).isoformat(), "register": "Present_Position",
              "bus_calls": BUS_CALLS, "motor_writes": 0, "ledger_rows": 0, "calibration": provenance,
              "motor_table": [{"joint": m, "motor": m, "id": i, "units": u} for m, i, u in MOTOR_TABLE],
              "raw_ticks": {m: int(ticks[m]) for m in names},
              "normalized": {m: float(normalized[i]) for m, i, _ in MOTOR_TABLE},
              "saturated": saturated, "valid_for_seeding": not saturated,
              "invalid_reason": None if not saturated else
              f"raw ticks beyond the calibration range_min/range_max: {saturated}"}
    if entries is not None:
        radians = to_radians(entries, list(record["normalized"].values()))
        limits = spec.position_limits
        record.update(joint_map={"path": str(args.joint_map.absolute()), "sha256": map_sha},
                      radians=dict(zip(spec.joint_names, radians.tolist())),  # Never clipped.
                      inside_spec_limits=bool(np.all((radians >= limits[:, 0]) & (radians <= limits[:, 1]))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(f"[read-only] {args.port}: Present_Position of {len(names)} motors to {args.output}; "
          f"saturated: {saturated or 'none'}; no motor writes; no ledger rows appended.")
    return record
