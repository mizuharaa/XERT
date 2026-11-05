"""Slice K drawer task: scene composition, reach and jaw sizing, and the success predicate on hand-built MjData.

No policy, no expert and no ledger rows: every state below is placed by hand, then scored.
"""
import hashlib
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from xert.tasks.drawer import DRAWER_TRAVEL, PLACEMENT, SLIDE, drawer_scene

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = ("xlerobot_left", "so101")


@pytest.fixture(scope="module", params=ROBOTS)
def scene(request):
    model, fingerprint, composition = drawer_scene(request.param)
    return request.param, model, fingerprint, composition


def test_scene_adds_a_fixed_cabinet_and_leaves_vendored_files_pinned(scene):
    from xert.data.cameras import bench_fingerprint
    from xle_workbench.sim import ASSETS, SO101_ASSETS, ManipulationEnv
    robot, model, fingerprint, composition = scene
    files = sorted(p for root in (ASSETS, SO101_ASSETS) for p in root.rglob("*") if p.is_file())
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    drawer_scene(robot)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    bench = ManipulationEnv({"robot": robot})
    assert composition["bench_model_sha256"] == bench_fingerprint(robot) == bench.metadata()["model_sha256"]
    assert len(fingerprint) == 64 and fingerprint != composition["bench_model_sha256"]
    assert model.opt.timestep == bench.model.opt.timestep == .002
    # Block and tray are gone; the drawer is the only object joint (the slide), so nq = arm + 1.
    for name in ("block", "tray"):
        with pytest.raises(KeyError):
            model.body(name)
    assert model.nq == bench.model.nq - 7 + 1 and model.nv == bench.model.nv - 6 + 1
    cabinet, drawer = model.body("cabinet"), model.body("drawer")
    assert cabinet.parentid[0] == 0 and model.body_weldid[cabinet.id] == 0  # Welded to the world: fixed.
    np.testing.assert_allclose(cabinet.pos, PLACEMENT[robot]["pos"])
    slide = model.joint("drawer_slide")
    assert slide.type[0] == mujoco.mjtJoint.mjJNT_SLIDE and slide.bodyid[0] == drawer.id
    np.testing.assert_array_equal(slide.range, [0, DRAWER_TRAVEL])
    assert model.jnt_limited[slide.id]
    assert (model.dof_damping[slide.dofadr[0]], model.dof_frictionloss[slide.dofadr[0]],
            model.dof_armature[slide.dofadr[0]]) == (SLIDE["damping"], SLIDE["frictionloss"], SLIDE["armature"])
    handle = model.geom("handle")
    assert handle.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER and handle.bodyid[0] == drawer.id
    assert model.geom_friction[handle.id, 0] == 1.0
    # A different cabinet pose is a different scene.
    assert drawer_scene("so101")[1] != drawer_scene("xlerobot_left")[1]


def test_drawer_slides_horizontally_and_the_arm_starts_clear_of_the_cabinet(scene):
    robot, model, _, _ = scene
    from xle_workbench.sim import ROBOTS as BENCH
    data = mujoco.MjData(model)
    qids = [model.joint(name).qposadr[0] for name in BENCH[robot]["joints"]]
    data.qpos[qids] = BENCH[robot]["reset_pose"]
    mujoco.mj_forward(model, data)
    assert data.ncon == sum(model.geom(g).name == "mount" for g in data.contact.geom1)  # Only the bench's own.
    closed = data.geom("handle").xpos.copy()
    data.qpos[model.joint("drawer_slide").qposadr[0]] = DRAWER_TRAVEL
    mujoco.mj_forward(model, data)
    moved = data.geom("handle").xpos - closed
    assert np.isclose(np.linalg.norm(moved), DRAWER_TRAVEL) and abs(moved[2]) < 1e-12
    yaw = PLACEMENT[robot]["yaw"]
    np.testing.assert_allclose(moved / DRAWER_TRAVEL, [np.cos(yaw), np.sin(yaw), 0], atol=1e-12)


def test_both_arms_reach_the_handle_and_the_handle_fits_their_jaws():
    sys.path.insert(0, str(ROOT / "scripts"))
    from drawer_reach import TOLERANCE, jaw_sizing, reach_report
    for robot in ROBOTS:
        report = reach_report(robot)
        assert [row["travel_fraction"] for row in report["samples"]] == [0, .5, 1]
        assert report["worst_fk_error_m"] < TOLERANCE, robot
        jaws = jaw_sizing(robot)
        assert jaws["graspable_positions"] > 0, robot
        nearest = jaws["nearest_to_grasp_site"]
        assert nearest["open_clearance_moving_m"] > 0 and nearest["closed_clearance_moving_m"] < 0


# ---------------------------------------------------------------- Success predicate on hand-built states.

def place(model, opening, arm=None, speed=0.):
    """A fresh MjData: drawer.pos = opening (slide speed `speed`), arm at `arm` (joint -> radians, default qpos0)."""
    data = mujoco.MjData(model)
    if arm is not None:
        data.qpos[[model.joint(name).qposadr[0] for name in arm]] = list(arm.values())
    slide = model.joint("drawer_slide")
    data.qpos[slide.qposadr[0]], data.qvel[slide.dofadr[0]] = opening, speed
    mujoco.mj_forward(model, data)
    return data


def reset_arm(robot):
    from xle_workbench.sim import ROBOTS as BENCH
    return dict(zip(BENCH[robot]["joints"], BENCH[robot]["reset_pose"]))


def ik_touch(robot, model, opening, offset, closed=False):
    """Arm pose putting the grasp site at the handle centre + offset (world metres), drawer at `opening`, by the
    existing IK (xert.retarget.ik.solve, as scripts/drawer_reach.py); `closed` then shuts the gripper to its lower
    actuator limit, so the jaws close on whatever sits there."""
    from xert.retarget.ik import solve
    from xert.spec import load_spec
    spec = load_spec(ROOT / f"xert/spec/{robot}.yaml")
    data = place(model, opening, reset_arm(robot))
    target = data.geom("handle").xpos + offset
    q = solve(spec, model, data, {"grasp": target}, iterations=200)
    if closed:
        q[-1] = spec.position_limits[-1][0]
    return dict(zip(spec.joint_names, q))


def ik_hold(robot, model, opening):
    """Gripper closed with a jaw pressed 2 mm into the handle: grasp site at the handle centre, then moved by the
    gap to the arm geom nearest the handle (position-only IK cannot choose the jaw orientation, so this is a hand-built
    touching state, not a grasp an expert would plan)."""
    first = ik_touch(robot, model, opening, [0, 0, 0], closed=True)
    data = place(model, opening, first)
    handle, scene = model.geom("handle").id, {0, model.body("drawer").id, model.body("cabinet").id}
    arm = [g for g in range(model.ngeom) if model.geom_bodyid[g] not in scene
           and (model.geom_contype[g] or model.geom_conaffinity[g])]
    fromto, gaps = np.zeros(6), {g: mujoco.mj_geomDistance(model, data, handle, g, 1., None) for g in arm}
    nearest = min(gaps, key=gaps.get)
    mujoco.mj_geomDistance(model, data, handle, nearest, 1., fromto)
    toward_jaw = (fromto[3:] - fromto[:3]) / np.linalg.norm(fromto[3:] - fromto[:3])
    return ik_touch(robot, model, opening, -toward_jaw * (gaps[nearest] + .002), closed=True)


def touched(model, data):
    """Names of drawer geoms in contact with the arm."""
    drawer, cabinet = model.body("drawer").id, model.body("cabinet").id
    names = set()
    for g1, g2 in zip(data.contact.geom1, data.contact.geom2):
        b1, b2 = model.geom_bodyid[g1], model.geom_bodyid[g2]
        for mine, other, body in ((g1, b2, b1), (g2, b1, b2)):
            if body == drawer and other not in (0, cabinet):
                names.add(model.geom(mine).name)
    return names


def test_each_check_on_hand_built_states(scene):
    robot, model, _, _ = scene
    from xert.tasks.drawer import OPEN_M, SLIDE_SPEED, DrawerTask
    task, arm = DrawerTask(model), reset_arm(robot)
    assert task.checks(place(model, .1, arm)) == {"open": True, "released": True, "still": True}
    assert task.checks(place(model, OPEN_M, arm))["open"] and not task.checks(place(model, OPEN_M - 1e-4, arm))["open"]
    assert not task.checks(place(model, .1, arm, speed=SLIDE_SPEED + 1e-3))["still"]
    assert task.checks(place(model, .1, arm, speed=-(SLIDE_SPEED - 1e-3)))["still"]
    held = place(model, .1, ik_hold(robot, model, .1))
    assert "handle" in touched(model, held) and not task.checks(held)["released"]


def test_a_held_open_drawer_never_succeeds_until_released(scene):
    """Gripper on the handle with the drawer open for 20 ticks: no success. Release: success on clean tick 12."""
    from xert.tasks.drawer import SETTLE_TICKS, DrawerTask
    robot, model, _, _ = scene
    held = place(model, .1, ik_hold(robot, model, .1))
    task = DrawerTask(model)
    for _ in range(20):
        info = task.step(held)
        assert not info["success"] and info["object_in_robot_contact"] and info["stable_steps"] == 0
    assert info["drawer_open_fraction"] == pytest.approx(.1 / DRAWER_TRAVEL)
    released = place(model, .1, reset_arm(robot))
    infos = [task.step(released) for _ in range(SETTLE_TICKS)]
    assert [i["success"] for i in infos] == [False] * (SETTLE_TICKS - 1) + [True] and task.finished
    assert infos[-1]["failure_reason"] is None and not infos[-1]["object_in_robot_contact"]
    with pytest.raises(RuntimeError, match="episode finished"):
        task.step(released)


def test_an_arm_pushing_on_the_drawer_body_is_no_success_until_released(scene):
    """Robot contact on a drawer panel (not the handle), drawer open: no success; clear of it, success."""
    from xert.tasks.drawer import SETTLE_TICKS, DrawerTask
    robot, model, _, _ = scene
    yaw = PLACEMENT[robot]["yaw"]
    # Grasp site on the front panel's outer face, 2.5 cm below the handle axis (the handle is 3.5 cm proud of it).
    face = -.035 * np.array([np.cos(yaw), np.sin(yaw), 0]) + [0, 0, -.025]
    pushing = place(model, .1, ik_touch(robot, model, .1, face))
    contacts = touched(model, pushing)
    assert contacts and "handle" not in contacts, contacts
    task = DrawerTask(model)
    assert not any(task.step(pushing)["success"] for _ in range(20)) and task.stable_steps == 0
    clear = place(model, .1, reset_arm(robot))
    assert [task.step(clear)["success"] for _ in range(SETTLE_TICKS)] == [False] * (SETTLE_TICKS - 1) + [True]


def test_a_dip_below_open_resets_the_count_and_the_horizon_is_the_only_failure(scene):
    from xert.tasks.drawer import OPEN_M, SETTLE_TICKS, DrawerTask
    robot, model, _, _ = scene
    arm = reset_arm(robot)
    opened, dipped = place(model, .1, arm), place(model, OPEN_M - .001, arm)
    task = DrawerTask(model)
    infos = [task.step(state) for state in [opened] * (SETTLE_TICKS - 1) + [dipped] + [opened] * SETTLE_TICKS]
    assert [i["stable_steps"] for i in infos] == [*range(1, SETTLE_TICKS), 0, *range(1, SETTLE_TICKS + 1)]
    assert [i["success"] for i in infos].index(True) == len(infos) - 1
    task = DrawerTask(model, horizon=3)
    infos = [task.step(place(model, .01, arm)) for _ in range(3)]
    assert [i["failure_reason"] for i in infos] == [None, None, "timeout"] and task.finished
    assert not any(i["success"] or i["lifted"] for i in infos)


def test_initial_opening_is_seeded_inside_its_range():
    from xert.tasks.drawer import INITIAL_OPEN, OPEN_M, initial_opening
    values = [initial_opening(seed) for seed in range(200)]
    assert initial_opening(3) == initial_opening(3) and initial_opening(0) != initial_opening(1)
    assert all(INITIAL_OPEN[0] <= v <= INITIAL_OPEN[1] < OPEN_M for v in values)  # No episode starts open.


def test_drawer_environment_names_are_one_list():
    from xert.tasks import environment_names
    assert environment_names("drawer") == ["drawer.pos", "handle.x", "handle.y", "handle.z",
                                           "grasp.x", "grasp.y", "grasp.z", "time_fraction"]
