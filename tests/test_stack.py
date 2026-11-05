"""Slice H stack task: scene composition, seeded spawns, and the success predicate on hand-built MjData states.

No policy, no expert and no ledger rows: every state below is placed by hand, then scored.
"""
import hashlib
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from xert.tasks import environment_names
from xert.tasks.stack import GAP_BELOW_M, GAP_M, SETTLE_TICKS, SPAWN, StackTask, spawn, stack_scene

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = ("xlerobot_left", "so101")
HEIGHT = .032  # Bench block height (2 x .016 m half-size).


@pytest.fixture(scope="module", params=ROBOTS)
def scene(request):
    model, fingerprint, composition = stack_scene(request.param)
    return request.param, model, fingerprint, composition


def place(model, base, block, block_velocity=(0,) * 6, at_grasp=False, arm=None):
    """A fresh MjData with both blocks placed (xyz centres, upright) and the arm at `arm` (default qpos0)."""
    data = mujoco.MjData(model)
    if arm is not None:
        data.qpos[[model.joint(name).qposadr[0] for name in arm]] = list(arm.values())
    if at_grasp:  # Stack the pair around the grasp site, so the jaws touch `block`: a held stack.
        mujoco.mj_forward(model, data)
        block = data.site("grasp").xpos.copy()
        base = block - [0, 0, HEIGHT - 1e-5]
    for name, xyz in (("base_block", base), ("block", block)):
        address = model.joint(f"{name}_joint").qposadr[0]
        data.qpos[address:address + 7] = [*xyz, 1, 0, 0, 0]
    dof = model.joint("block_joint").dofadr[0]
    data.qvel[dof:dof + 6] = block_velocity
    mujoco.mj_forward(model, data)
    return data


def ik_grasp(robot, model, target, base, block):
    """Arm pose (joint name -> radians) putting the grasp site at `target`, by the existing generalized IK
    (xert.retarget.ik.solve, as scripts/stack_reach.py); the jaws then touch whatever block sits there."""
    from xert.retarget.ik import solve
    from xert.spec import load_spec
    spec = load_spec(ROOT / f"xert/spec/{robot}.yaml")
    q = solve(spec, model, place(model, base, block), {"grasp": np.asarray(target)}, iterations=200)
    return dict(zip(spec.joint_names, q))


BASE = np.array([.24, .03, .096])
STACKED = BASE + [0, 0, HEIGHT - 1e-5]  # 10 um penetration so the resting contact exists.


def test_scene_adds_base_block_and_leaves_vendored_files_pinned(scene):
    from xert.data.cameras import bench_fingerprint
    from xle_workbench.sim import ASSETS, SO101_ASSETS, ManipulationEnv
    robot, model, fingerprint, composition = scene
    files = sorted(p for root in (ASSETS, SO101_ASSETS) for p in root.rglob("*") if p.is_file())
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    stack_scene(robot)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    bench = ManipulationEnv({"robot": robot})
    assert (model.nq, model.nbody, model.ngeom) == (bench.model.nq + 7, bench.model.nbody + 1, bench.model.ngeom + 1)
    for field in ("geom_size", "geom_friction", "geom_condim"):
        np.testing.assert_array_equal(getattr(model, field)[model.geom("base_block_geom").id],
                                      getattr(bench.model, field)[bench.model.geom("block_geom").id])
    assert model.body_mass[model.body("base_block").id] == bench.model.body_mass[bench.block]
    assert model.geom_bodyid[model.geom("table").id] == 0  # Table contact is world contact, never robot contact.
    assert composition["bench_model_sha256"] == bench_fingerprint(robot) == bench.metadata()["model_sha256"]
    assert len(fingerprint) == 64 and fingerprint != composition["bench_model_sha256"]
    assert model.opt.timestep == bench.model.opt.timestep


def test_environment_names_are_the_native_bench_names_plus_base_block():
    from xle_workbench.sim import ManipulationEnv
    assert environment_names("pick_place") == ManipulationEnv().metadata()["observation_names"][12:]
    stack = environment_names("stack")
    assert stack[:7] == environment_names("pick_place")[:7] and stack[7:14] == [
        f"base_block.{axis}" for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    assert stack[14:] == environment_names("pick_place")[7:]


def test_spawns_are_seeded_inside_their_rectangles_and_never_overlap():
    assert [a.tolist() for a in spawn(3)] == [a.tolist() for a in spawn(3)]
    for seed in range(200):
        block, base = spawn(seed)
        for xy, name in ((block, "block"), (base, "base_block")):
            assert np.all(SPAWN[name][0] <= xy) and np.all(xy <= SPAWN[name][1])
        assert abs(block[1] - base[1]) >= .016 + .014  # Footprints 16 mm square; 3 cm apart at the closest.


def test_both_arms_reach_both_spawn_rectangles_at_stacking_heights():
    sys.path.insert(0, str(ROOT / "scripts"))
    from stack_reach import TOLERANCE, reach_report
    for robot in ROBOTS:
        report = reach_report(robot)
        assert len(report["samples"]) == 24 and report["worst_fk_error_m"] < TOLERANCE, robot


def test_each_check_on_hand_built_states(scene):
    _, model, _, _ = scene
    task = StackTask(model)
    assert task.checks(place(model, BASE, STACKED)) == dict.fromkeys(("contact", "released", "base_on_table", "gap", "centred", "still"), True)
    apart = task.checks(place(model, BASE, [.24, -.03, .096]))
    assert not apart["contact"] and apart["released"] and not apart["gap"]
    assert not task.checks(place(model, BASE, STACKED + [.009, 0, 0]))["centred"]  # 9 mm > 8 mm half width.
    assert task.checks(place(model, BASE, STACKED + [.0075, 0, 0]))["centred"]
    sunk = task.checks(place(model, BASE, STACKED - [0, 0, .004]))  # 4 mm into the base: interpenetration.
    assert sunk["contact"] and not sunk["gap"]
    assert task.checks(place(model, BASE, STACKED - [0, 0, GAP_BELOW_M - .0002]))["gap"]
    hover = task.checks(place(model, BASE, STACKED + [0, 0, GAP_M + .001]))
    assert not hover["contact"] and not hover["gap"]
    assert not task.checks(place(model, BASE, STACKED, [.011, 0, 0, 0, 0, 0]))["still"]
    assert not task.checks(place(model, BASE, STACKED, [0, 0, 0, 0, .11, 0]))["still"]
    held = task.checks(place(model, None, None, at_grasp=True))
    assert held["contact"] and not held["released"]
    floating = task.checks(place(model, BASE + [0, 0, .05], STACKED + [0, 0, .05]))  # Base in mid-air, untouched.
    assert floating["released"] and not floating["base_on_table"]
    assert not task.checks(place(model, BASE - [0, 0, .004], STACKED - [0, 0, .004]))["base_on_table"]


def scores(model, states, horizon=600):
    task = StackTask(model, horizon)
    return [task.step(state) for state in states]


def test_success_needs_twelve_consecutive_stacked_ticks(scene):
    _, model, _, _ = scene
    stacked, apart = place(model, BASE, STACKED), place(model, BASE, [.24, -.03, .096])
    infos = scores(model, [stacked] * (SETTLE_TICKS - 1) + [apart] + [stacked] * SETTLE_TICKS)
    assert [i["stable_steps"] for i in infos] == [*range(1, SETTLE_TICKS), 0, *range(1, SETTLE_TICKS + 1)]
    assert [i["success"] for i in infos].index(True) == len(infos) - 1 and infos[-1]["failure_reason"] is None


def test_a_held_perfect_stack_never_succeeds_until_released(scene):
    """Review H-01: the gripper holding a perfectly placed stack for 20 ticks is no success; release, then 12 clean
    ticks, is."""
    from xle_workbench.sim import ROBOTS
    robot, model, _, _ = scene
    # A perfect stack on the table, the grasp site at the block centre: the jaws touch `block`.
    held = place(model, BASE, STACKED, arm=ik_grasp(robot, model, STACKED, BASE, STACKED))
    task = StackTask(model)
    for _ in range(20):
        info = task.step(held)
        assert not info["success"] and info["object_in_robot_contact"] and info["stable_steps"] == 0
    # Release: the same block poses, the arm back at its bench reset pose, clear of both blocks.
    released = place(model, held.xpos[model.body("base_block").id], held.xpos[model.body("block").id],
                     arm=dict(zip(ROBOTS[robot]["joints"], ROBOTS[robot]["reset_pose"])))
    assert StackTask(model).checks(released) == dict.fromkeys(("contact", "released", "base_on_table", "gap", "centred", "still"), True)
    infos = [task.step(released) for _ in range(SETTLE_TICKS)]
    assert [i["success"] for i in infos] == [False] * (SETTLE_TICKS - 1) + [True] and task.finished
    with pytest.raises(RuntimeError, match="episode finished"):
        task.step(released)


def test_a_base_held_in_the_air_with_the_block_on_top_never_succeeds(scene):
    """QA F3: the gripper lifting base_block with block resting on it is no stack; both blocks on the table is."""
    robot, model, _, _ = scene
    base, block = BASE + [0, 0, .06], STACKED + [0, 0, .06]
    lifted = place(model, base, block, arm=ik_grasp(robot, model, base, base, block))
    checks = StackTask(model).checks(lifted)
    assert checks["contact"] and not checks["released"] and not checks["base_on_table"]
    task = StackTask(model)
    assert not any(task.step(lifted)["success"] for _ in range(20)) and task.stable_steps == 0
    on_table = place(model, BASE, STACKED)
    assert [task.step(on_table)["success"] for _ in range(SETTLE_TICKS)] == [False] * (SETTLE_TICKS - 1) + [True]


def test_failures_off_table_and_horizon(scene):
    _, model, _, _ = scene
    [info] = scores(model, [place(model, BASE, [.24, -.03, .03])])
    assert info["failure_reason"] == "fell_off_table" and not info["success"]
    [info] = scores(model, [place(model, [.24, .03, .03], [.24, -.03, .096])])
    assert info["failure_reason"] == "fell_off_table"  # Either block off the table ends the episode.
    infos = scores(model, [place(model, BASE, [.24, -.03, .096])] * 3, horizon=3)
    assert [i["failure_reason"] for i in infos] == [None, None, "timeout_no_lift"]
    infos = scores(model, [place(model, BASE, [.24, -.03, .15]), place(model, BASE, [.24, -.03, .096])], horizon=2)
    assert infos[-1]["failure_reason"] == "timeout_after_lift" and infos[-1]["lifted"]
