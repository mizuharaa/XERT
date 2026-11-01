"""SO-101 hardware backend, dry-run only: converts safe sim targets to motor space and prints them.

Nothing here opens a port, imports LeRobot or sends a byte. The documented future path (P6.2, owner-supervised)
is LeRobot 0.6.1 `lerobot.motors.feetech.FeetechMotorsBus` (read-only `connect()` + `sync_read("Present_Position")`
first), never `lerobot.robots.so_follower.SO101Follower.connect()`, which writes configuration and enables torque.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .base import JointState

TBD = "TBD (not measured)"
DOCUMENTED = ("joint", "motor", "id", "units")
MEASURED = ("sign", "units_per_rad", "zero_offset", "min", "max", "max_speed", "max_accel")


@dataclass(frozen=True)
class CommandedState(JointState):
    """Last commanded targets echoed back. NOT a hardware measurement."""
    measured: bool = False


def load_joint_map(path, spec, *, dry_run, contents=None):
    """Return one dict per spec joint, or None where any measured value is TBD (dry-run only)."""
    raw = json.loads(Path(path).read_bytes() if contents is None else contents)
    joints = raw.get("joints") if isinstance(raw, dict) else None
    if not isinstance(joints, list) or [j.get("joint") if isinstance(j, dict) else None for j in joints] != list(spec.joint_names):
        raise ValueError("Joint map needs one entry per spec joint, in spec order")
    mapped = []
    for entry in joints:
        if set(entry) != set(DOCUMENTED + MEASURED):
            raise ValueError(f"Joint map fields must be exactly {DOCUMENTED + MEASURED}: {entry['joint']}")
        if any(entry[k] == TBD for k in MEASURED):
            if not dry_run:
                raise ValueError(f"{entry['joint']}: TBD joint-map values are refused outside dry-run; measure them (P6.1)")
            mapped.append(None)  # Never guessed: an unmeasured joint stays unmapped.
            continue
        if any(type(entry[k]) not in (int, float) or not np.isfinite(entry[k]) for k in MEASURED):
            raise ValueError(f"{entry['joint']}: measured values must be finite numbers or {TBD!r}")
        if (entry["sign"] not in (-1, 1) or entry["units_per_rad"] <= 0 or entry["min"] >= entry["max"] or
                entry["max_speed"] <= 0 or entry["max_accel"] <= 0):
            raise ValueError(f"{entry['joint']}: sign must be +/-1; scale, speed, accel positive; min < max")
        mapped.append(entry)
    return mapped


class SO101HardwareBackend:
    """Backend protocol. Hardware limits apply AFTER run_loop's SafetyController, on top of it, in motor space."""

    def __init__(self, spec, joint_map_path, *, dry_run=True, echo=print):
        if dry_run is not True:
            raise RuntimeError("SO-101 motion needs owner-supervised bring-up (P6.1/P6.2); only dry_run=True exists")
        self.spec, self.echo = spec, echo
        self.map = load_joint_map(joint_map_path, spec, dry_run=True)
        self.dt = 1 / spec.control_hz
        # ponytail: the start pose is the spec rest pose, ASSUMED; real bring-up seeds it from a read-only position read.
        self.q = spec.rest.astype(float)
        self.qd = np.zeros_like(self.q)
        self.motor = [None if m is None else self._to_motor(m, r) for m, r in zip(self.map, self.q)]
        if any(m is not None and not e["min"] <= m <= e["max"] for m, e in zip(self.motor, self.map)):
            raise ValueError("Rest pose maps outside the joint map's motor limits")
        self.motor_velocity = [0.0] * len(self.q)
        self.t = 0.0
        self.last_command = None
        self.last_sent = []
        self.closed = False

    @staticmethod
    def _to_motor(entry, q):
        return entry["sign"] * entry["units_per_rad"] * q + entry["zero_offset"]

    def read_state(self):
        if self.closed:
            raise RuntimeError("Backend is closed")
        return CommandedState(self.q.copy(), self.qd.copy(), self.t)

    def send_targets(self, q, t_cmd):
        """Dry-run: convert, clamp to hardware limits, print and record. Nothing is sent."""
        if self.closed:
            raise RuntimeError("Backend is closed")
        q = np.asarray(q, dtype=float)
        if q.shape != self.q.shape or not np.isfinite(q).all():
            raise ValueError("Targets must be a finite ordered joint vector")
        if not np.isfinite(t_cmd) or (self.last_command is not None and t_cmd <= self.last_command):
            raise ValueError("Command timestamps must be finite and strictly increasing")
        applied, records = q.copy(), []
        for i, (name, entry) in enumerate(zip(self.spec.joint_names, self.map)):
            if entry is None:
                records.append({"joint": name, "status": "unmapped", "sim_target": float(q[i])})
                continue
            wanted = self._to_motor(entry, q[i])
            last, v0 = self.motor[i], self.motor_velocity[i]
            target = min(max(wanted, entry["min"]), entry["max"])
            step = entry["max_accel"] * self.dt
            v = np.clip((target - last) / self.dt, v0 - step, v0 + step)
            v = float(np.clip(v, -entry["max_speed"], entry["max_speed"]))
            motor = min(max(last + v * self.dt, entry["min"]), entry["max"])
            self.motor[i], self.motor_velocity[i] = motor, (motor - last) / self.dt
            applied[i] = (motor - entry["zero_offset"]) / (entry["sign"] * entry["units_per_rad"])
            records.append({"joint": name, "status": "mapped", "motor": entry["motor"], "id": entry["id"],
                            "units": entry["units"], "sim_target": float(q[i]), "motor_target": motor,
                            "clamped": not np.isclose(motor, wanted, rtol=0, atol=1e-9)})
        self.qd, self.q = (applied - self.q) / self.dt, applied
        self.t = self.last_command = float(t_cmd)
        self.last_sent = records
        self.echo(f"[dry-run, no hardware] t={t_cmd:.3f} " + " | ".join(
            f"{r['joint']}: unmapped (sim {r['sim_target']:+.4f} rad)" if r["status"] == "unmapped" else
            f"{r['joint']}: id {r['id']} -> {r['motor_target']:+.3f} {r['units']}{' clamped' if r['clamped'] else ''}"
            for r in records))

    def close(self):
        self.closed = True
