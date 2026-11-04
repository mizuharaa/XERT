"""`xert map-from-calibration`: the SO-101 joint map's kinematic fields from evidence FILES (slice F1).

Inputs are a LeRobot 0.6.1 follower calibration file (lerobot-calibrate) and labelled static poses: lerobot-record
episodes where the arm was held still, each with the sim joint vector `sim_q` (radians, spec order) that Daniel
asserts the arm was in. This module reads files only: it never imports lerobot.robots, lerobot.motors or serial,
and never opens a bus.

Per joint, in the map's units (`motor = sign * units_per_rad * sim_rad + zero_offset`):
  min/max         the calibrated tick range through LeRobot's normalization (TICK_TO_UNIT).
  sign, units_per_rad, zero_offset
                  >= 2 distinct sim_q values: least squares `motor = a*sim_q + b`, sign = sign(a), scale = |a|, offset b.
                  1 distinct value, degree joint: offset only; sign +1 and 180/pi (direct drive) are ASSUMED and
                  listed in provenance.assumed, which load_joint_map refuses outside dry-run.
                  1 distinct value, gripper: left TBD; a 0-100 scale cannot be assumed.
  max_speed/max_accel stay TBD: they are Daniel's safety choice, not something these files measure.
"""
import json
import math
from pathlib import Path

import numpy as np

from xert.backends.so101_hw_backend import DOCUMENTED, MEASURED, TBD
from xert.data.lerobot_io import JOINT_MAP, ROOT, _calibration, _columns, _fresh, _hashes, _names, source_kind
from xle_workbench.storage import atomic_json

# sts3215: MODEL_RESOLUTION 4096, LeRobot divides by resolution - 1 (feetech/tables.py:186-190, motors_bus.py:877).
MAX_RES = 4095
# so_follower.py:50-60 with use_degrees=True: body joints DEGREES, gripper RANGE_0_100.
NORM_MODES = ["degrees"] * 5 + ["range_0_100"]
TICK_TO_UNIT = {
    "degrees": "deg = (ticks - mid) * 360 / 4095, mid = (range_min + range_max) / 2 (LeRobot 0.6.1 motors_bus.py:876-879; "
               "not clamped, drive_mode unused); the zero is the calibrated range midpoint, not the sim zero",
    "range_0_100": "norm = (clamp(ticks, range_min, range_max) - range_min) / (range_max - range_min) * 100, "
                   "100 - norm when drive_mode (LeRobot 0.6.1 motors_bus.py:873-875)"}
# Slice D rule: a still arm reads +-1 encoder step, so a pose whose max - min exceeds 2 steps on any joint moved.
STILL_STEPS = 2
# ponytail: tooling choice, not measured. 1 motor unit (1 deg, 1% of gripper travel) allows for hand-posing error
# against the asserted sim_q; tighten once real poses show their scatter.
RESIDUAL_TOLERANCE = 1.0


def _step(units, cal):
    """One encoder tick in motor units."""
    return 360 / MAX_RES if units == "degrees" else 100 / (cal["range_max"] - cal["range_min"])


def _range(units, cal):
    if units == "range_0_100":
        return 0.0, 100.0
    mid = (cal["range_min"] + cal["range_max"]) / 2
    return (cal["range_min"] - mid) * 360 / MAX_RES, (cal["range_max"] - mid) * 360 / MAX_RES


def _pose(entry, base, motors, limits, steps):
    """(median <motor>.pos over the episode, provenance); refused if the arm moved or sim_q is not a sim pose."""
    if not isinstance(entry, dict) or set(entry) != {"label", "dataset", "episode", "sim_q"} or             type(entry["episode"]) is not int:
        raise ValueError("Each pose needs exactly label, dataset, episode (int), sim_q")
    sim_q = np.asarray(entry["sim_q"], dtype=np.float64)
    if sim_q.shape != (len(motors),) or not np.isfinite(sim_q).all():
        raise ValueError(f"{entry['label']}: sim_q must be {len(motors)} finite radians in spec order")
    if np.any(sim_q < limits[:, 0]) or np.any(sim_q > limits[:, 1]):
        raise ValueError(f"{entry['label']}: sim_q is outside the so101 spec position limits")
    dataset = (base / entry["dataset"]).absolute()
    if source_kind(dataset) != "so101_follower":
        raise ValueError(f"{entry['label']}: {dataset} is not a LeRobot so101 follower recording")
    if _names(json.loads((dataset / "meta/info.json").read_text("utf-8")), "observation.state") != motors:
        raise ValueError(f"{entry['label']}: observation.state names must be exactly {motors}")
    columns = _columns(dataset)
    state = columns["observation.state"][columns["episode_index"] == entry["episode"]].astype(np.float64)
    if not len(state):
        raise ValueError(f"{entry['label']}: episode {entry['episode']} has no frames")
    moved = state.max(axis=0) - state.min(axis=0)
    for name, spread, step in zip(motors, moved, steps):
        if spread > STILL_STEPS * step:
            raise ValueError(f"{entry['label']}: {name} moved {spread:.4f} > {STILL_STEPS} steps ({STILL_STEPS * step:.4f}); "
                             "a static pose must be held still")
    median = np.median(state, axis=0)
    return median, sim_q, {"label": entry["label"], "dataset": str(dataset), "dataset_sha256": _hashes(dataset),
                           "episode": entry["episode"], "frames": len(state), "sim_q": sim_q.tolist(),
                           "median_motor": median.tolist(), "max_minus_min": moved.tolist()}


def map_from_calibration(calibration, poses, output):
    """Write a fresh joint map; never the real configs/hardware/so101_joint_map.yaml (it exists, so _fresh refuses)."""
    from xert.spec import load_spec
    output = _fresh(output)
    spec = load_spec(ROOT / "xert/spec/so101.yaml")
    template = json.loads(JOINT_MAP.read_text("utf-8"))
    # Only the documented fields come from the template; every measured field starts TBD and is filled below.
    template["joints"] = [{**{k: j[k] for k in DOCUMENTED}, **dict.fromkeys(MEASURED, TBD)} for j in template["joints"]]
    if [j["units"] for j in template["joints"]] != NORM_MODES:
        raise ValueError(f"Template units differ from LeRobot's so_follower norm modes {NORM_MODES}")
    record = _calibration(calibration)  # Refuses anything but an so101 follower calibration file.
    cal = json.loads(Path(calibration).read_bytes())
    for entry in template["joints"]:
        c = cal[entry["motor"]]
        if c["id"] != entry["id"] or c["range_min"] >= c["range_max"]:
            raise ValueError(f"{entry['joint']}: calibration id {c['id']} differs from the map id {entry['id']} "
                             "or range_min >= range_max")
    poses = Path(poses)
    labelled = json.loads(poses.read_text("utf-8"))
    if not isinstance(labelled, list) or not labelled:
        raise ValueError("poses.json must be a nonempty list of poses")
    motors = [f"{name}.pos" for name in spec.joint_names]
    steps = [_step(e["units"], cal[e["motor"]]) for e in template["joints"]]
    readings = [_pose(p, poses.parent, motors, spec.position_limits, steps) for p in labelled]
    medians, sim = np.array([r[0] for r in readings]), np.array([r[1] for r in readings])
    fits, assumed, limits_in_units = {}, {}, {}
    for j, entry in enumerate(template["joints"]):
        name, units, c = entry["joint"], entry["units"], cal[entry["motor"]]
        q, motor = sim[:, j], medians[:, j]
        lo, hi = _range(units, c)
        entry.update(min=lo, max=hi)
        distinct = len(np.unique(q))
        if distinct >= 2:
            a, b = np.polyfit(q, motor, 1)
            method = "least squares motor = a*sim_q + b"
        elif units == "degrees":
            a = 180 / math.pi
            b = float(np.mean(motor - a * q))
            method = "one sim_q value: offset only, sign +1 and 180/pi direct drive ASSUMED"
            assumed[name] = ["sign", "units_per_rad"]
        else:
            fits[name] = {"method": "not fitted: a 0-100 gripper scale cannot be assumed; record >= 2 distinct "
                                    "gripper sim_q poses", "poses": len(q), "distinct_sim_q": distinct}
            continue
        residual = float(np.max(np.abs(motor - (a * q + b))))
        fits[name] = {"method": method, "poses": len(q), "distinct_sim_q": distinct, "max_abs_residual": residual,
                      "tolerance": RESIDUAL_TOLERANCE,
                      **({"note": "exact by construction with 2 distinct values; a 3rd pose is the real check"}
                         if distinct == 2 and len(q) == 2 else {})}
        if residual > RESIDUAL_TOLERANCE:
            raise ValueError(f"{name}: fit residual {residual:.4f} {units} above tolerance {RESIDUAL_TOLERANCE}")
        mapped = sorted(float(a * x + b) for x in spec.position_limits[j])
        limits_in_units[name] = mapped
        if mapped[0] < lo or mapped[1] > hi:
            raise ValueError(f"{name}: sim limits map to {mapped} {units}, outside the calibrated range [{lo}, {hi}]")
        entry.update(sign=1 if a > 0 else -1, units_per_rad=float(abs(a)), zero_offset=float(b))
    template["status"] = (f"GENERATED by xert map-from-calibration from robot {record['robot_id']}; read provenance, "
                          "then copy into configs/hardware/so101_joint_map.yaml by hand")
    template["provenance"] = {
        "tool": "xert map-from-calibration (slice F1); files only, no bus opened",
        "calibration": {**record, "values": cal}, "tick_to_unit": TICK_TO_UNIT,
        "min_max": "calibrated range_min/range_max through tick_to_unit",
        "still_rule": f"refused if any joint's max - min over the episode exceeds {STILL_STEPS} encoder steps "
                      f"(degrees: 360/{MAX_RES}; gripper: 100/(range_max - range_min)); value = median",
        "poses": [r[2] for r in readings], "fit": fits, "sim_limits_in_motor_units": limits_in_units,
        "assumed": assumed,
        "not_measured": "max_speed and max_accel stay TBD: Daniel's safety choice (P6 QA question 5)"}
    atomic_json(output, template)
    return {"output": str(output), "robot_id": record["robot_id"], "fit": fits, "assumed": assumed}


def add_command(commands):
    parser = commands.add_parser("map-from-calibration",
                                 help="SO-101 joint map kinematic fields from a calibration file + static poses")
    parser.add_argument("--calibration", type=Path, required=True, help="LeRobot <robot.id>.json (lerobot-calibrate)")
    parser.add_argument("--poses", type=Path, required=True, help="poses.json: [{label, dataset, episode, sim_q}]")
    parser.add_argument("--output", type=Path, required=True, help="A fresh map path; never the real config")


def execute(args):
    print(json.dumps(map_from_calibration(args.calibration, args.poses, args.output)))
