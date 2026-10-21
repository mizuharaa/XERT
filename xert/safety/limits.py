"""Position, speed and acceleration limits on fixed-rate joint commands."""
from dataclasses import dataclass
import numpy as np

from .watchdog import Watchdog


@dataclass
class SafeCommand:
    q: np.ndarray
    clamped: bool
    phase: str
    events: list[dict]


class SafetyController:
    def __init__(self, spec, initial=None, *, stale_s=.2, hold_s=.5, return_s=2.0):
        self.spec = spec
        self.dt = 1 / spec.control_hz
        self.q = np.asarray(spec.rest if initial is None else initial, dtype=float).copy()
        if (self.q.shape != spec.rest.shape or not np.isfinite(self.q).all() or
                np.any(self.q < spec.position_limits[:, 0]) or np.any(self.q > spec.position_limits[:, 1])):
            raise ValueError("Initial pose must be finite and inside limits")
        self.velocity = np.zeros_like(self.q)
        self.watchdog = Watchdog(spec.rest, stale_s, hold_s, return_s)
        self.last_time = None
        self.phase = None
        self.stopped = False
        self.events = []

    def handle_key(self, key):
        """UI key hook: Escape latches emergency hold on the next control tick."""
        if key in ("Escape", "escape", "esc", "\x1b"):
            self.stopped = True
        return self.stopped

    def _braking_speed(self, distance):
        # Exact discrete stopping distance; reserve space to decelerate on later ticks.
        low = np.zeros_like(distance)
        high = self.spec.velocity_limits.copy()
        increment = self.spec.acceleration_limits * self.dt
        for _ in range(48):
            speed = (low + high) / 2
            count = np.maximum(np.ceil(speed / increment) - 1, 0)
            travel = self.dt * ((count + 1) * speed - increment * count * (count + 1) / 2)
            valid = travel <= np.maximum(distance, 0)
            low = np.where(valid, speed, low)
            high = np.where(valid, high, speed)
        return low

    def step(self, target, *, t_now, t_capture=None, estop=False):
        """Call exactly once per control period; e-stop is latched until reconstructed."""
        if type(estop) is not bool:
            raise ValueError("estop must be a boolean")
        if not np.isfinite(t_now) or (self.last_time is not None and
                not np.isclose(t_now - self.last_time, self.dt, rtol=1e-6, atol=1e-9)):
            raise ValueError("SafetyController requires a finite fixed-rate command clock")
        desired = None if target is None else np.asarray(target, dtype=float)
        if desired is not None and (desired.shape != self.q.shape or not np.isfinite(desired).all()):
            raise ValueError("Target must be a finite ordered joint vector")
        self.stopped |= bool(estop)
        events = []
        if self.stopped:
            # Emergency hold intentionally overrides the normal acceleration ramp.
            self.velocity[:] = 0
            phase = "estop"
            clamped = True
            events.append({"t": float(t_now), "kind": "clamp", "reason": "estop",
                           "joints": list(range(len(self.q))),
                           "requested": None if desired is None else desired.tolist(), "applied": self.q.tolist()})
        else:
            desired, phase = self.watchdog.select(desired, self.q, t_now, t_capture)
            limits = self.spec.position_limits
            bounded = np.clip(desired, *limits.T)
            requested_velocity = (bounded - self.q) / self.dt
            increment = self.spec.acceleration_limits * self.dt
            low = np.maximum(self.velocity - increment, -self._braking_speed(self.q - limits[:, 0]))
            high = np.minimum(self.velocity + increment, self._braking_speed(limits[:, 1] - self.q))
            if np.any(low > high + 1e-9):
                raise ValueError("Command history cannot satisfy the configured braking limits")
            velocity = np.clip(requested_velocity, low, np.maximum(low, high))
            result = self.q + velocity * self.dt
            changed = np.flatnonzero(~np.isclose(result, desired, rtol=0, atol=1e-12))
            clamped = bool(len(changed))
            if clamped:
                events.append({"t": float(t_now), "kind": "clamp", "joints": changed.tolist(),
                               "requested": desired.tolist(), "applied": result.tolist()})
            self.q, self.velocity = result, velocity
        if phase != self.phase:
            events.append({"t": float(t_now), "kind": "watchdog", "phase": phase})
        self.phase, self.last_time = phase, float(t_now)
        self.events.extend(events)
        return SafeCommand(self.q.copy(), clamped, phase, events)
