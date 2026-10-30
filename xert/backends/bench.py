"""Pick-and-place bench driven through the native core, so success is its contact rule."""
import numpy as np

from .base import JointState


class BenchBackend:
    """One bench episode. Absolute targets are normalized exactly as ManipulationEnv expects.

    The core steps 20 physics substeps per call and decides success, failure and timeout;
    `info` holds its latest result and `finished` means the core refuses further steps.
    """
    def __init__(self, spec, seed):
        from xle_workbench.sim import JOINTS, ManipulationEnv
        self.env = ManipulationEnv()
        if (spec.joint_names != tuple(JOINTS) or spec.control_hz != self.env.metadata()["control_hz"] or
                not np.array_equal(spec.position_limits, self.env.limits)):
            raise ValueError("Spec does not describe the bench arm joints, limits and control rate")
        self.spec, self.model, self.data = spec, self.env.model, self.env.data
        self.object_names = self.env.metadata()["observation_names"][12:]  # The P1.4 ACT environment_state split.
        self.env.reset(seed)
        self.info, self.closed = None, False

    @property
    def finished(self):
        return self.env.finished

    def read_state(self):
        if self.closed:
            raise RuntimeError("Backend is closed")
        return JointState(self.data.qpos[self.env.qids].copy(), self.data.qvel[self.env.dids].copy(),
                          float(self.data.time))

    def object_state(self):
        """Native observation after the 12 joint entries: block pose, grasp, tray and time fraction."""
        return self.env._obs()[12:].tolist()  # ponytail: private accessor; the native env has no public one.

    def send_targets(self, q, t_cmd):
        if self.closed:
            raise RuntimeError("Backend is closed")
        # ponytail: the core validates range/finiteness and rejects steps after the episode ends.
        self.info = self.env.step(self.env.normalize(np.asarray(q, dtype=float)))[3]

    def close(self):
        self.closed = True
