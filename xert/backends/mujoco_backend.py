"""Spec-driven MuJoCo backend for direct, unit-gear position actuators."""
import mujoco
import numpy as np

from .base import JointState


class MuJoCoBackend:
    def __init__(self, spec):
        self.spec = spec
        self.model = spec.load_model()
        self.data = mujoco.MjData(self.model)
        self.qids = np.array([self.model.joint(n).qposadr[0] for n in spec.joint_names])
        self.dids = np.array([self.model.joint(n).dofadr[0] for n in spec.joint_names])
        self.actuators = []
        for name in spec.joint_names:
            joint = self.model.joint(name).id
            matches = [a for a in range(self.model.nu)
                       if self.model.actuator_trntype[a] == int(mujoco.mjtTrn.mjTRN_JOINT) and
                       self.model.actuator_trnid[a, 0] == joint]
            if len(matches) != 1:
                raise ValueError(f"Expected one direct position actuator for {name}")
            a = matches[0]
            if not (self.model.actuator_gear[a, 0] == 1 and
                    np.all(self.model.actuator_gear[a, 1:] == 0) and
                    self.model.actuator_dyntype[a] == int(mujoco.mjtDyn.mjDYN_NONE) and
                    self.model.actuator_gaintype[a] == int(mujoco.mjtGain.mjGAIN_FIXED) and
                    self.model.actuator_biastype[a] == int(mujoco.mjtBias.mjBIAS_AFFINE) and
                    self.model.actuator_gainprm[a, 0] > 0 and
                    self.model.actuator_biasprm[a, 0] == 0 and
                    self.model.actuator_biasprm[a, 1] == -self.model.actuator_gainprm[a, 0]):
                raise ValueError(f"Unsupported actuator: {name}; expected a unit-gear position servo")
            self.actuators.append(a)
        ratio = 1 / (spec.control_hz * self.model.opt.timestep)
        self.substeps = round(ratio)
        if self.substeps < 1 or not np.isclose(ratio, self.substeps, rtol=0, atol=1e-9):
            raise ValueError("Control period must contain an integer number of physics steps")
        self.data.qpos[self.qids] = spec.rest
        self.data.ctrl[self.actuators] = spec.rest
        mujoco.mj_forward(self.model, self.data)
        self.last_command = None
        self.closed = False

    def read_state(self):
        if self.closed:
            raise RuntimeError("Backend is closed")
        return JointState(self.data.qpos[self.qids].copy(), self.data.qvel[self.dids].copy(), float(self.data.time))

    def send_targets(self, q, t_cmd):
        if self.closed:
            raise RuntimeError("Backend is closed")
        q = np.asarray(q, dtype=float)
        if (q.shape != self.spec.rest.shape or not np.isfinite(q).all() or
                np.any(q < self.spec.position_limits[:, 0] - 1e-10) or
                np.any(q > self.spec.position_limits[:, 1] + 1e-10)):
            raise ValueError("Targets must be finite ordered positions inside spec limits")
        if not np.isfinite(t_cmd) or (self.last_command is not None and t_cmd <= self.last_command):
            raise ValueError("Command timestamps must be finite and strictly increasing")
        self.data.ctrl[self.actuators] = q
        previous_time = self.data.time
        try:
            mujoco.mj_step(self.model, self.data, nstep=self.substeps)
            if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or
                    not np.isclose(self.data.time - previous_time, 1 / self.spec.control_hz, rtol=0, atol=1e-9)):
                raise RuntimeError("Simulation produced nonfinite state or reset its clock")
            # mj_step integrates qpos after its last position-dependent calculations.
            mujoco.mj_forward(self.model, self.data)
        except Exception:
            self.closed = True
            raise
        self.last_command = float(t_cmd)

    def close(self):
        self.closed = True
