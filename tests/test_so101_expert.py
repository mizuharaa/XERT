"""Done-criteria for the SO-101 scripted expert (TODO(daniel)), each shown passing on the XLeRobot expert.

Criteria come from the XLeRobot expert's own tests, not new ones: seed-0 lift through finger contact,
release and native success (test_sim.py, test_core.cpp); every CTest export seed 0..8 (CMakeLists.txt);
the evaluate CLI's held-out seeds from 10000; pre-grasp within 5 mm (PLAN.md B; the XLeRobot expert
measures 4.4 mm worst on seeds 0..8 at the end of its hover). The SO-101 cases are strict xfails that
expect NotImplementedError, so CI turns red once the stubs exist: as a failure if a criterion is
missed, as XPASS if all pass. Then delete the marker.
"""
import numpy as np
import pytest

from xle_workbench.sim import ManipulationEnv
from xle_workbench.so101_expert import so101_expert_targets

TODO = pytest.mark.xfail(strict=True, raises=NotImplementedError,
                         reason="TODO(daniel): xle_workbench/so101_expert.py is a stub")
ARMS = ["xlerobot_left", pytest.param("so101", marks=TODO)]


def pre_grasp(env):
    """(hover target, hover steps). XLeRobot: cpp/src/ik.cpp:130. SO-101: the first constant target."""
    if env.config["robot"] == "xlerobot_left":
        return env.initial_block + [0, 0, .06], 65
    tray = env.model.body("tray").pos
    first = np.asarray(so101_expert_targets(0, env.initial_block, tray)[0])
    steps = 1
    while steps < env.config["horizon"] and np.array_equal(so101_expert_targets(steps, env.initial_block, tray)[0], first):
        steps += 1
    return first, steps


def run(env, seed):
    env.reset(seed)
    lifted_in_contact, info = False, {}
    while not env.finished:
        _, _, _, info = env.step(env.expert_action())
        lifted_in_contact |= info["object_in_robot_contact"] and info["object_position"][2] > .145
    return lifted_in_contact, info


@pytest.mark.parametrize("robot", ARMS)
def test_expert_reaches_pre_grasp_within_5mm_on_seeds_0_8(robot):
    env = ManipulationEnv({"robot": robot})
    for seed in range(9):
        env.reset(seed)
        target, steps = pre_grasp(env)
        for _ in range(steps):
            env.step(env.expert_action())
        assert np.linalg.norm(env.data.site_xpos[env.site] - target) < .005, seed


@pytest.mark.parametrize("robot", ARMS)
def test_expert_lifts_through_contact_and_succeeds_on_ctest_seeds_0_8(robot):
    env = ManipulationEnv({"robot": robot})
    for seed in range(9):
        lifted_in_contact, info = run(env, seed)
        assert lifted_in_contact and info["success"] and not info["object_in_robot_contact"], (seed, info)


@pytest.mark.parametrize("robot", ARMS)
def test_expert_succeeds_on_held_out_seeds_10000_10009(robot):
    env = ManipulationEnv({"robot": robot})
    outcomes = {seed: run(env, seed)[1] for seed in range(10000, 10010)}
    assert all(info["success"] for info in outcomes.values()), outcomes
