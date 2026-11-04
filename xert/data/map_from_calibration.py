"""`xert map-from-calibration`: the SO-101 joint map's kinematic fields from evidence FILES (slice F1).

Inputs are a LeRobot 0.6.1 follower calibration file (lerobot-calibrate) and labelled static poses: lerobot-record
episodes where the arm was held still, each with the sim joint vector `sim_q` (radians, spec order) that Daniel
asserts the arm was in. This module reads files only: it never imports lerobot.robots, lerobot.motors or serial,
and never opens a bus.

Per joint, in the map's units (`motor = sign * units_per_rad * sim_rad + zero_offset`):
  min/max         the calibrated tick range through LeRobot's normalization (TICK_TO_UNIT).
  sign, units_per_rad, zero_offset
                  least squares `motor = a*sim_q + b`, sign = sign(a), scale = |a|, offset b, when the joint has
                  >= 2 distinct sim_q values, its reading moved beyond still noise and its sim_q spread is at least
                  MIN_SPREAD; one distinct value leaves them TBD (never assumed).
  max_speed/max_accel stay TBD: they are Daniel's safety choice, not something these files measure.
"""
import json
from pathlib import Path
import sys

import numpy as np

from xert.backends.so101_hw_backend import DOCUMENTED, MEASURED, TBD
from xert.data.lerobot_io import JOINT_MAP, ROOT, _calibration, _columns, _fresh, _hashes, _names, source_kind
from xle_workbench.storage import atomic_json

# sts3215: MODEL_RESOLUTION 4096, LeRobot divides by resolution - 1 (feetech/tables.py:186-190, motors_bus.py:876).
MAX_RES = 4095
# so_follower.py:50-60 with use_degrees=True: body joints DEGREES, gripper RANGE_0_100.
NORM_MODES = ["degrees"] * 5 + ["range_0_100"]
TICK_TO_UNIT = {
    "degrees": "deg = (ticks - mid) * 360 / 4095, mid = (range_min + range_max) / 2 (LeRobot 0.6.1 motors_bus.py:874-877; "
               "not clamped, drive_mode unused); the zero is the calibrated range midpoint, not the sim zero",
    "range_0_100": "norm = (clamp(ticks, range_min, range_max) - range_min) / (range_max - range_min) * 100, "
                   "100 - norm when drive_mode (LeRobot 0.6.1 motors_bus.py:867,871-873)"}
# Slice D rule: a still arm reads +-1 encoder step, so a pose whose max - min exceeds 2 steps on any joint moved.
STILL_STEPS = 2
# ponytail: tooling choice, not measured. 1 motor unit (1 deg, 1% of gripper travel) allows for hand-posing error
# against the asserted sim_q; tighten once real poses show their scatter.
RESIDUAL_TOLERANCE = 1.0
# Minimum sim_q spread per joint: a still reading is uncertain by STILL_STEPS steps, so the slope is uncertain by
# about STILL_STEPS*step/spread, which at the farthest sim limit must stay within RESIDUAL_TOLERANCE motor units.
MIN_SPREAD = "STILL_STEPS * step * max|sim limit| / RESIDUAL_TOLERANCE"


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
    if not isinstance(entry, dict) or set(entry) != {"label", "dataset", "episode", "sim_q"} or \
            type(entry["episode"]) is not int:
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
    fits, limits_in_units, excess = {}, {}, {}
    for j, entry in enumerate(template["joints"]):
        name, units, c = entry["joint"], entry["units"], cal[entry["motor"]]
        q, motor = sim[:, j], medians[:, j]
        lo, hi = _range(units, c)
        entry.update(min=lo, max=hi)
        sim_lo, sim_hi = (float(x) for x in spec.position_limits[j])
        min_spread = STILL_STEPS * steps[j] * max(abs(sim_lo), abs(sim_hi)) / RESIDUAL_TOLERANCE
        distinct = len(np.unique(q))
        fits[name] = {"poses": len(q), "distinct_sim_q": distinct, "min_sim_spread_rad": min_spread}
        if distinct < 2:
            fits[name]["method"] = "not fitted: one sim_q value; record >= 2 distinct sim_q (sign, scale, offset TBD)"
            continue
        if np.ptp(motor) <= 2 * STILL_STEPS * steps[j]:
            raise ValueError(f"{name}: reading moved {np.ptp(motor):.4f} {units} across poses, within still noise "
                             f"({2 * STILL_STEPS * steps[j]:.4f}); the joint did not move")
        if np.ptp(q) < min_spread:
            raise ValueError(f"{name}: sim_q spread {np.ptp(q):.4f} rad below the minimum {min_spread:.4f} rad")
        a, b = np.polyfit(q, motor, 1)
        residual = float(np.max(np.abs(motor - (a * q + b))))
        fits[name].update(method="least squares motor = a*sim_q + b", max_abs_residual=residual,
                          tolerance=RESIDUAL_TOLERANCE,
                          **({"note": "exact by construction with 2 distinct values; a 3rd pose is the real check"}
                             if distinct == 2 and len(q) == 2 else {}))
        if residual > RESIDUAL_TOLERANCE:
            raise ValueError(f"{name}: fit residual {residual:.4f} {units} above tolerance {RESIDUAL_TOLERANCE}")
        mapped = sorted(float(a * x + b) for x in (sim_lo, sim_hi))
        limits_in_units[name] = mapped
        # Owner rule: a full-turn joint (LeRobot calibrates wrist_roll as 0..4095, so_follower.py:142-143) and the
        # gripper (clamped to 0..100) only need the sim range to FIT in the calibrated span; the part outside is
        # cut off by the backend's min/max clamp and recorded. Other joints must map inside the calibrated range.
        if units == "range_0_100" or (c["range_min"], c["range_max"]) == (0, MAX_RES):
            if abs(a) * (sim_hi - sim_lo) > hi - lo:
                raise ValueError(f"{name}: sim range spans {abs(a) * (sim_hi - sim_lo):.2f} {units}, more than the "
                                 f"calibrated span {hi - lo:.2f}")
            reach = sorted(((lo - b) / a, (hi - b) / a))  # Sim radians inside the calibrated range.
            cut = {"low_rad": max(0.0, float(reach[0] - sim_lo)), "high_rad": max(0.0, float(sim_hi - reach[1]))}
            if cut["low_rad"] or cut["high_rad"]:
                excess[name] = cut
                print(f"warning: {name}: sim limits exceed the calibrated range by {cut}; the backend clamp cuts "
                      "that off", file=sys.stderr)
        elif mapped[0] < lo or mapped[1] > hi:
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
        "min_spread_rule": MIN_SPREAD, "poses": [r[2] for r in readings], "fit": fits,
        "sim_limits_in_motor_units": limits_in_units, "sim_limits_excess": excess,
        "sim_limits_excess_rule": "sim radians beyond the calibrated range at each end, cut off by the backend "
                                  "min/max clamp (full-turn joints and the gripper only)",
        "not_measured": "max_speed and max_accel stay TBD: Daniel's safety choice (P6 QA question 5)"}
    atomic_json(output, template)
    return {"output": str(output), "robot_id": record["robot_id"], "fit": fits, "sim_limits_excess": excess}


def add_command(commands):
    parser = commands.add_parser("map-from-calibration",
                                 help="SO-101 joint map kinematic fields from a calibration file + static poses")
    parser.add_argument("--calibration", type=Path, required=True, help="LeRobot <robot.id>.json (lerobot-calibrate)")
    parser.add_argument("--poses", type=Path, required=True, help="poses.json: [{label, dataset, episode, sim_q}]")
    parser.add_argument("--output", type=Path, required=True, help="A fresh map path; never the real config")


def execute(args):
    print(json.dumps(map_from_calibration(args.calibration, args.poses, args.output)))
