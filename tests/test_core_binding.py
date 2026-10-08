"""The native core borrows Python allocations and preserves the public state."""
from collections import deque
import gc
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from xle_workbench._reach_core import Env
from xle_workbench.sim import ManipulationEnv


def test_borrowed_owners_validation_and_state_mirror():
    subprocess.run([sys.executable, "-c", "import xle_workbench._reach_core"], check=True)
    env = ManipulationEnv({"delay_steps": 3, "horizon": 4})
    core = env._core
    observation = env.reset(7)
    assert observation.dtype == np.float32 and observation.shape == (26,)
    assert isinstance(env.pending, deque) and len(env.pending) == 3
    np.testing.assert_array_equal(env.initial_block, env.data.xpos[env.block])
    before = env.data.qpos.copy()
    for action in [np.zeros((1, 6)), np.zeros(5), np.full(6, np.nan), np.full(6, np.inf), np.full(6, 2)]:
        with pytest.raises(ValueError):
            env.step(action)
        np.testing.assert_array_equal(env.data.qpos, before)
    env.step(env.expert_action())
    np.testing.assert_array_equal(observation, env.reset(7))
    for _ in range(4):
        obs, _, _, info = env.step(np.zeros(6))
        assert env.steps == info["steps"] == core.steps
        assert env.lifted == info["lifted"] == core.lifted
        assert env.stable_steps == core.stable_steps and env.finished == core.finished
        assert len(env.pending) == 3
        np.testing.assert_array_equal(obs, env._obs())
    assert env.finished and info["failure_reason"] == "timeout_no_lift"
    other = mujoco.MjModel.from_xml_string("<mujoco/>")
    with pytest.raises(ValueError, match="owners"):
        Env(env.model, mujoco.MjData(other), env.ik_data, 600)
    expected = env._obs()
    del env
    gc.collect()
    np.testing.assert_array_equal(core.observation(), expected)
    core.reset_state([.24, -.03], 0)
    core.step(np.zeros(6))  # All three native owners remain alive via keep_alive.
