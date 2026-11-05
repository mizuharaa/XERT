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
