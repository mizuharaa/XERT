"""Measure the drawer scene against both bench arms: handle reach at 0/50/100 % travel, and handle vs jaw size.

Run from the checkout: .venv/Scripts/python.exe scripts/drawer_reach.py
Reach: the slice B/H method. Position-only generalized IK (xert.retarget.ik.solve, no new solver math) from the
bench reset pose to the handle centre, then FK. Approach ORIENTATION is not measured: a pose that reaches the handle
centre may hold the jaws at an angle that cannot close on it; the expert (TODO(daniel)) has to choose one.
`shell_contacts` counts arm-cabinet contacts at the IK pose (IK ignores collisions); informational only.
Jaws: a probe cylinder of the handle's radius and length, its axis along the gripper hinge axis, on a 2 mm grid
(+/-4 cm) in the closing plane through the grasp site. A position counts as graspable when, at the gripper's OPEN
actuator limit, the probe touches neither jaw, and at the CLOSED limit the moving jaw overlaps it (so closing reaches
the handle before the limit) while the fixed jaw does not. Static geometry only: no contact forces are simulated.
Exits non-zero unless every FK error is below 1 mm and the handle fits the open jaws and is gripped closed.
"""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xert.retarget.ik import solve
from xert.spec import load_spec
from xert.tasks.drawer import DRAWER_TRAVEL, HANDLE, drawer_scene, drawer_spec
from xle_workbench.sim import ROBOTS

TOLERANCE = 1e-3  # The P4.1 bench-profile IK tolerance, as scripts/stack_reach.py.


def reach_report(robot):
    spec = load_spec(ROOT / f"xert/spec/{robot}.yaml")
    model, _, _ = drawer_scene(robot)
    data, fk = mujoco.MjData(model), mujoco.MjData(model)
    qids = [model.joint(name).qposadr[0] for name in spec.joint_names]
    slide = model.joint("drawer_slide").qposadr[0]
    shell = {model.body("cabinet").id}
    rows = []
    for fraction in (0, .5, 1):
        data.qpos[:] = 0
        data.qpos[qids], data.qpos[slide] = ROBOTS[robot]["reset_pose"], fraction * DRAWER_TRAVEL
        mujoco.mj_forward(model, data)
        target = data.geom("handle").xpos.copy()
        q = solve(spec, model, data, {"grasp": target}, iterations=200)
        fk.qpos[:] = data.qpos
        fk.qpos[qids] = q
        mujoco.mj_forward(model, fk)
        bodies = model.geom_bodyid[np.stack([fk.contact.geom1, fk.contact.geom2], axis=1)]
        rows.append({"travel_fraction": fraction, "target": target.tolist(), "q": q.tolist(),
                     "fk_error_m": float(np.linalg.norm(fk.site("grasp").xpos - target)),
                     "shell_contacts": int(sum(bool(shell & {a, b}) and 0 not in (a, b) for a, b in bodies))})
    return {"robot": robot, "ik": "xert.retarget.ik.solve, position-only, 200 iterations", "start": "bench reset pose",
            "orientation": "not measured", "tolerance_m": TOLERANCE, "samples": rows,
            "worst_fk_error_m": max(row["fk_error_m"] for row in rows)}


def jaw_sizing(robot):
    joints = ROBOTS[robot]["joints"]
    spec, _, _ = drawer_spec(robot)
    model = spec.compile()
    data = mujoco.MjData(model)
    qids = [model.joint(name).qposadr[0] for name in joints]
    data.qpos[qids] = ROBOTS[robot]["reset_pose"]
    mujoco.mj_forward(model, data)
    hinge = model.joint(joints[-1])
    moving = hinge.bodyid[0]
    fixed = model.body_parentid[moving]
    axis = data.xmat[moving].reshape(3, 3) @ model.jnt_axis[hinge.id]
    site = data.site("grasp").xpos.copy()
    quat = np.zeros(4)
    mujoco.mju_quatZ2Vec(quat, axis)
    spec.worldbody.add_geom(name="jaw_probe", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=site, quat=quat,
                            size=[HANDLE["radius"], HANDLE["half_length"], 0], contype=0, conaffinity=0)
    model = spec.compile()
    data = mujoco.MjData(model)
    probe = model.geom("jaw_probe").id
    jaws = {body: [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                   and (model.geom_contype[g] or model.geom_conaffinity[g])] for body in (fixed, moving)}
    closed, opened = model.actuator_ctrlrange[model.actuator(joints[-1]).id]

    def clearance(gripper, centre):
        """(to fixed jaw, to moving jaw) surface distance of the probe, metres; negative = overlap."""
        data.qpos[qids] = ROBOTS[robot]["reset_pose"]
        data.qpos[hinge.qposadr[0]] = gripper
        model.geom_pos[probe] = centre
        mujoco.mj_forward(model, data)
        return tuple(min(mujoco.mj_geomDistance(model, data, probe, g, 1., None) for g in jaws[body])
                     for body in (fixed, moving))

    # Candidate handle axes: a 2 mm grid in the jaws' closing plane (normal = hinge axis) through the grasp site.
    u = np.cross(axis, [0, 0, 1.]) if abs(axis[2]) < .9 else np.cross(axis, [1., 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    grid = [site + a * u + b * v for a in np.arange(-.04, .0401, .002) for b in np.arange(-.04, .0401, .002)]
    rows = []
    for centre in grid:
        open_fixed, open_moving = clearance(opened, centre)
        if open_fixed > 0 and open_moving > 0:  # Fits between the open jaws, touching neither.
            closed_fixed, closed_moving = clearance(closed, centre)
            if closed_moving < 0 <= closed_fixed:  # The closing jaw reaches it; the fixed jaw does not overlap it.
                rows.append((float(np.linalg.norm(centre - site)), centre, open_fixed, open_moving, closed_moving))
    result = {"robot": robot, "handle_radius_m": HANDLE["radius"], "hinge_axis_world_at_reset": axis.tolist(),
              "gripper_open_rad": float(opened), "gripper_closed_rad": float(closed), "grid_step_m": .002,
              "graspable_positions": len(rows)}
    if rows:
        distance, centre, open_fixed, open_moving, closed_moving = min(rows, key=lambda row: row[0])
        result["nearest_to_grasp_site"] = {"offset_m": distance, "open_clearance_fixed_m": open_fixed,
                                           "open_clearance_moving_m": open_moving,
                                           "closed_clearance_moving_m": closed_moving}
    return result


if __name__ == "__main__":
    reports = [{"reach": reach_report(robot), "jaws": jaw_sizing(robot)} for robot in ROBOTS]
    print(json.dumps(reports, indent=2))
    raise SystemExit(0 if all(r["reach"]["worst_fk_error_m"] < TOLERANCE and r["jaws"]["graspable_positions"] > 0
                              for r in reports) else 1)
