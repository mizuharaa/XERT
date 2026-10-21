"""Host-monotonic timestamps and explicitly named hand-estimator outputs."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class HandFrame:
    image: np.ndarray
    world: np.ndarray
    handedness_score: float

    def __post_init__(self):
        for name in ("image", "world"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != (21, 3) or not np.isfinite(value).all():
                raise ValueError("A hand requires finite image/world arrays of shape (21, 3)")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.isfinite(self.handedness_score) or not 0 <= self.handedness_score <= 1:
            raise ValueError("Handedness score must be in [0, 1]")


@dataclass(frozen=True)
class MotionFrame:
    sequence: int
    t_capture: float
    t_landmarks: float
    hands: dict[str, HandFrame]
    source: str
    image_size: tuple[int, int]

    def __post_init__(self):
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("Frame sequence must be a nonnegative integer")
        if not np.isfinite([self.t_capture, self.t_landmarks]).all() or self.t_landmarks < self.t_capture:
            raise ValueError("Frame timestamps must be finite host-monotonic seconds in order")
        if self.source not in ("webcam", "scripted", "recorded_human"):
            raise ValueError("Motion source must identify its provenance")
        if set(self.hands) - {"left", "right"} or any(not isinstance(h, HandFrame) for h in self.hands.values()):
            raise ValueError("Hands must be unambiguous left/right HandFrames")
        if len(self.image_size) != 2 or any(type(v) is not int or v <= 0 for v in self.image_size):
            raise ValueError("Image size must contain positive integer width and height")


class MotionSource(Protocol):
    def read(self) -> MotionFrame | None: ...
    def close(self) -> None: ...
