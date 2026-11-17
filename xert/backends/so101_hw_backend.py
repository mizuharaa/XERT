"""SO-101 hardware backend, dry-run only: converts safe sim targets to motor space and prints them.

Nothing here opens a port, imports LeRobot or sends a byte. The documented future path (P6.2, owner-supervised)
is LeRobot 0.6.1 `lerobot.motors.feetech.FeetechMotorsBus` (read-only `connect()` + `sync_read("Present_Position")`
first), never `lerobot.robots.so_follower.SO101Follower.connect()`, which writes configuration and enables torque.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from xert.safety.limits import braking_speed
from .base import JointState

TBD = "TBD (not measured)"
DOCUMENTED = ("joint", "motor", "id", "units")
UNITS = ("degrees", "range_0_100", "range_m100_100")  # lerobot/motors/motors_bus.py:169-172 MotorNormMode.
MEASURED = ("sign","units_per_rad", "zero_offset", "min", "max", "max_speed", "max_accel")
# Completeness tiers: frame conversion (import-lerobot, replay-real) needs only the kinematic fields; motion needs all.
TIERS = {"kinematic": MEASURED[:5], "motion": MEASURED}
# LeRobot 0.6.1 SO-101 follower motors (so_follower.py:50-60, use_degrees=True), all model sts3215: (motor, id, units).
MOTOR_TABLE = (("shoulder_pan", 1, "degrees"), ("shoulder_lift", 2, "degrees"), ("elbow_flex", 3, "degrees"),
               ("wrist_flex", 4, "degrees"), ("wrist_roll", 5, "degrees"), ("gripper", 6, "range_0_100"))


@dataclass(frozen=True)
class CommandedState(JointState):
    """Last commanded targets echoed back. NOT a hardware measurement."""
    measured: bool = False


def load_joint_map(path, spec, *, dry_run, contents=None, require="motion"):
    """Return one dict per spec joint, or None where any measured value is TBD (dry-run only).

    Outside dry-run every field of the `require` tier must be measured (not TBD).
    """
    raw = json.loads(Path(path).read_bytes() if contents is None else contents)
    joints = raw.get("joints") if isinstance(raw, dict) else None
    if not isinstance(joints, list) or [j.get("joint") if isinstance(j, dict) else None for j in joints] != list(spec.joint_names):
        raise ValueError("Joint map needs one entry per spec joint, in spec order")
    mapped = []
    for entry in joints:
        if set(entry) != set(DOCUMENTED + MEASURED):
            raise ValueError(f"Joint map fields must be exactly {DOCUMENTED + MEASURED}: {entry['joint']}")
        if type(entry["id"]) is not int or not 1 <= entry["id"] <= 253 or not isinstance(entry["motor"], str) or not entry["motor"]:
            raise ValueError(f"{entry['joint']}: motor id must be an integer 1..253 and the motor name nonempty")
        if entry["units"] not in UNITS:
            raise ValueError(f"{entry['joint']}: units must be a LeRobot 0.6.1 MotorNormMode value {UNITS}")
        tbd = [k for k in MEASURED if entry[k] == TBD]
        if tbd and dry_run:
            mapped.append(None)  # Never guessed: an unmeasured joint stays unmapped.
            continue
        if set(tbd) & set(TIERS[require]):
            raise ValueError(f"{entry['joint']}: TBD joint-map values {tbd} are refused outside dry-run for "
                             f"{require}; measure them (P6.1)")
        present = [k for k in MEASURED if k not in tbd]
        if any(type(entry[k]) not in (int, float) or not np.isfinite(entry[k]) for k in present):
            raise ValueError(f"{entry['joint']}: measured values must be finite numbers or {TBD!r}")
        if (entry["sign"] not in (-1, 1) or entry["units_per_rad"] <= 0 or entry["min"] >= entry["max"] or
                any(entry[k] <= 0 for k in ("max_speed", "max_accel") if k in present)):
            raise ValueError(f"{entry['joint']}: sign must be +/-1; scale, speed, accel positive; min < max")
        mapped.append(entry)
    if len({j["id"] for j in joints}) != len(joints):
        raise ValueError("Motor ids must be distinct")
    return mapped


def compatible_map(path, spec, calibration_sha256, *, contents=None):
    """(entries, sha256) of a joint map that can turn LeRobot SO-101 readings into radians, else ValueError.

    Two distinct refusals: any TBD kinematic field (load_joint_map, "TBD ..."), and a complete map whose
    (motor, id, units) differ from MOTOR_TABLE ("incompatible"). A map fitted on another calibration file is
    refused too (import-lerobot's rule, lerobot_io.py:201-204).
    """
    contents = Path(path).read_bytes() if contents is None else contents
    entries = load_joint_map(path, spec, dry_run=False, contents=contents, require="kinematic")
    if [(e["motor"], e["id"], e["units"]) for e in entries] != list(MOTOR_TABLE):
        raise ValueError(f"Joint map is incompatible with the SO-101 motor table (motor, id, units) {MOTOR_TABLE}")
    fitted = (json.loads(contents).get("provenance") or {}).get("calibration", {}).get("sha256")
    if fitted is not None and fitted != calibration_sha256:
        raise ValueError("Joint map was fitted on a different calibration file than this read")
    return entries, hashlib.sha256(contents).hexdigest()


def to_radians(entries, motor):
    """Inverse of `motor = sign * units_per_rad * rad + zero_offset`, float64: import-lerobot's conversion."""
    sign, scale, offset = (np.array([e[k] for e in entries], dtype=np.float64)
                           for k in ("sign", "units_per_rad", "zero_offset"))
    return (np.asarray(motor, dtype=np.float64) - offset) / (sign * scale)


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
            lo, hi, vmax, amax, dt = entry["min"], entry["max"], entry["max_speed"], entry["max_accel"], self.dt

            def brake(distance):  # The SafetyController rule, in this joint's motor units.
                return float(braking_speed(np.array(distance), vmax, amax, dt))
            # Same law as SafetyController.step: approach the bounded target, never faster than can still stop
            # there, and keep room to stop before either motor limit; acceleration bounds the speed change.
            error = min(max(wanted, lo), hi) - last
            v = np.sign(error) * min(abs(error) / dt, brake(abs(error)))
            low, high = max(v0 - amax * dt, -brake(last - lo)), min(v0 + amax * dt, brake(hi - last))
            if low > high + 1e-9:
                raise ValueError("Command history cannot satisfy the joint map's braking limits")
            motor = last + float(np.clip(v, low, max(low, high))) * dt
            self.motor[i], self.motor_velocity[i] = motor, (motor - last) / self.dt
            applied[i] = (motor - entry["zero_offset"]) / (entry["sign"] * entry["units_per_rad"])
            records.append({"joint": name, "status": "mapped", "motor": entry["motor"], "id": entry["id"],
                            "units": entry["units"], "sim_target": float(q[i]), "motor_target": motor,
                            "clamped": not np.isclose(motor, wanted, rtol=0, atol=1e-9)})
        self.qd, self.q = (applied - self.q) / self.dt, applied
        self.t = self.last_command = float(t_cmd)
        self._print(t_cmd, records)

    def hold(self, t_cmd):
        """E-stop: re-send the last motor targets at zero velocity, never the controller's frozen q.

        run_loop calls it only after a send; with nothing sent there is no hardware command to hold.
        """
        if self.closed:
            raise RuntimeError("Backend is closed")
        if self.last_command is None:
            raise RuntimeError("hold() before any send_targets: there is no last hardware command to hold")
        if not np.isfinite(t_cmd) or t_cmd <= self.last_command:
            raise ValueError("Command timestamps must be finite and strictly increasing")
        self.motor_velocity = [0.0] * len(self.q)
        self.qd = np.zeros_like(self.q)
        self.t = self.last_command = float(t_cmd)
        self._print(t_cmd, [
            {"joint": name, "status": "unmapped", "sim_target": float(q)} if entry is None else
            {"joint": name, "status": "hold", "motor": entry["motor"], "id": entry["id"], "units": entry["units"],
             "sim_target": float(q), "motor_target": motor, "clamped": False}
            for name, entry, q, motor in zip(self.spec.joint_names, self.map, self.q, self.motor)])

    def _print(self, t_cmd, records):
        self.last_sent = records
        self.echo(f"[dry-run, no hardware] t={t_cmd:.3f} " + " | ".join(
            f"{r['joint']}: unmapped (sim {r['sim_target']:+.4f} rad)" if r["status"] == "unmapped" else
            f"{r['joint']}: id {r['id']} -> {r['motor_target']:+.3f} {r['units']}"
            f"{' clamped' if r['clamped'] else ''}{' hold' if r['status'] == 'hold' else ''}"
            for r in records))

    def close(self):
        self.closed = True
