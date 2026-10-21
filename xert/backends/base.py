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
