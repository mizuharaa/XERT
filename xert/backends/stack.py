"""TaskBackend for the stack task: the composed stack scene on MuJoCoBackend, scored by StackTask in Python."""
from copy import deepcopy

import mujoco
import numpy as np

from xert.tasks import environment_names
from xert.tasks.stack import HORIZON, REST_Z, SPAWN, SUCCESS_RULE, StackTask, spawn, stack_scene
from .mujoco_backend import MuJoCoBackend


class StackBackend(MuJoCoBackend):
    """One seeded stack episode on a bench arm (xlerobot_left or so101), same arm pose and control rate as the bench.

    Each send_targets holds the targets for 1/control_hz (20 physics substeps on both arms), then StackTask scores
    the state. Observations are float32, as the native bench computes them, so live values equal recorded ones.
    """
    task_id = "stack"

    def __init__(self, spec, seed, horizon=HORIZON):
        from xle_workbench.sim import ROBOTS
        if spec.name not in ROBOTS:
            raise ValueError(f"Stacking needs a bench arm of {sorted(ROBOTS)}, not {spec.name!r}")
        model, self.model_sha256, self.composition = stack_scene(spec.name)
        super().__init__(spec, model=model)
        self.limits = self.model.actuator_ctrlrange[self.actuators].copy()
        if spec.joint_names != tuple(ROBOTS[spec.name]["joints"]) or not np.array_equal(spec.position_limits, self.limits):
            raise ValueError("Spec does not describe the bench arm joints and limits")
        self.config = {"robot": spec.name, "task_id": "stack", "seed": seed, "horizon": horizon, "friction": 1.0,
                       "spawn_xy": SPAWN}
        self.object_names = environment_names("stack")
        self.observation_names = ([f"joint_position.{n}" for n in spec.joint_names] +
                                  [f"joint_velocity.{n}" for n in spec.joint_names] + self.object_names)
        self.task = StackTask(self.model, horizon)
        reset = ROBOTS[spec.name]["reset_pose"]  # The bench reset pose, so a bench calibration's grasp matches.
        self.data.qpos[self.qids] = self.data.ctrl[self.actuators] = reset
        for name, xy in zip(("block", "base_block"), spawn(seed)):
            address = self.model.joint(f"{name}_joint").qposadr[0]
            self.data.qpos[address:address + 7] = [*xy, REST_Z, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.info = None

    @property
    def finished(self):
        return self.task.finished

    def metadata(self):
        return {"task_id": "stack", "observation_names": list(self.observation_names),
                "action_names": list(self.spec.joint_names), "control_hz": self.spec.control_hz,
                "horizon": self.task.horizon, "model_sha256": self.model_sha256,
                "source": {"composition": deepcopy(self.composition)}, "config": deepcopy(self.config),
                "success_rule": deepcopy(SUCCESS_RULE), "action_convention": "absolute joint position targets, radians",
                "observation_convention": "privileged simulator state, float32; world XYZ metres, wxyz quaternion, "
                                          "joint radians, rad/s",
                "simulation_only": True, "mujoco_version": mujoco.__version__}

    def object_state(self):
        values = [*self.data.qpos[self.model.joint("block_joint").qposadr[0]:][:7],
                  *self.data.qpos[self.model.joint("base_block_joint").qposadr[0]:][:7],
                  *self.data.site("grasp").xpos, *self.data.body("tray").xpos, self.task.steps / self.task.horizon]
        return np.asarray(values, dtype=np.float32).tolist()

    def observation(self):
        state = self.read_state()
        values = np.concatenate([state.q, state.qd]).astype(np.float32).tolist() + self.object_state()
        return dict(zip(self.observation_names, values))

    def send_targets(self, q, t_cmd=None):
        """t_cmd=None (policy rollouts): stamped with the sim time at the command, finite and increasing."""
        if self.finished:
            raise RuntimeError("episode finished; reset before stepping")
        super().send_targets(q, self.data.time if t_cmd is None else t_cmd)
        self.info = self.task.step(self.data)
