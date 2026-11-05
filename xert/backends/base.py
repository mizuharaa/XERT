from dataclasses import dataclass
from typing import Protocol

import numpy as np
from xert.spec import EmbodimentSpec


@dataclass(frozen=True)
class JointState:
    q: np.ndarray
    qd: np.ndarray
    t: float


class Backend(Protocol):
    spec: EmbodimentSpec

    def read_state(self) -> JointState: ...
    def send_targets(self, q: np.ndarray, t_cmd: float) -> None: ...
    def close(self) -> None: ...


class TaskBackend(Backend, Protocol):
    """One seeded bench-task episode: what teleop, the P5.3 harness and image-ACT evaluate use (review H-02).

    Implemented by BenchBackend (pick_place, native C++ rule) and StackBackend (stack, Python StackTask).
    `send_targets(q, None)` is allowed: the backend then stamps the command with its own sim time. After each
    send, `info` holds {success, failure_reason, lifted, steps, object_position, object_in_robot_contact} and
    `finished` means further steps are refused.
    """
    task_id: str
    model: object  # mujoco.MjModel / MjData, read by the calibrated mapper's IK and the viewer.
    data: object
    limits: np.ndarray  # (6, 2) actuator ctrlrange = spec position limits, radians.
    config: dict  # Includes "horizon" (control ticks).
    info: dict | None
    finished: bool
    object_names: list[str]  # xert.tasks.environment_names(task_id).

    def metadata(self) -> dict: ...  # Recorded as source.bench; carries model_sha256 and observation_names.
    def object_state(self) -> list[float]: ...  # Values of object_names, this tick.
    def observation(self) -> dict[str, float]: ...  # Every observation_names entry by name, this tick.


def task_backend(spec, seed, task_id) -> TaskBackend:
    """The bench backend for a task_id (xert.tasks.TASKS): one protocol, two implementations."""
    if task_id == "pick_place":
        from .bench import BenchBackend
        return BenchBackend(spec, seed)
    if task_id == "stack":
        from .stack import StackBackend
        return StackBackend(spec, seed)
    raise ValueError(f"Unknown task_id {task_id!r}")
