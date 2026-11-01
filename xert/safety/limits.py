"""Position, speed and acceleration limits on fixed-rate joint commands."""
from dataclasses import dataclass
import numpy as np

from .watchdog import Watchdog


def braking_speed(distance, velocity_limit, acceleration_limit, dt):
    """Fastest speed that can still stop within distance at this acceleration (exact discrete ticks)."""
    # Exact discrete stopping distance; reserve space to decelerate on later ticks.
    low = np.zeros_like(distance)
    high = np.array(velocity_limit, dtype=float)
    increment = acceleration_limit * dt
    for _ in range(48):
        speed = (low + high) / 2
        count = np.maximum(np.ceil(speed / increment) - 1, 0)
        travel = dt * ((count + 1) * speed - increment * count * (count + 1) / 2)
        valid = travel <= np.maximum(distance, 0)
        low = np.where(valid, speed, low)
        high = np.where(valid, high, speed)
    return low


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
        self.last_watchdog_time = None
        self.phase = None
        self.stopped = False
        self.events = []

    def handle_key(self, key):
        """UI key hook: Escape latches emergency hold on the next control tick."""
        if key in ("Escape", "escape", "esc", "\x1b"):
            self.stopped = True
        return self.stopped

    def _braking_speed(self, distance):
        return braking_speed(distance, self.spec.velocity_limits, self.spec.acceleration_limits, self.dt)

    def step(self, target, *, t_now, t_capture=None, estop=False, watchdog_now=None):
        """Nominal t_now sets clamp dt; optional wall clock ages capture timestamps."""
        if type(estop) is not bool:
            raise ValueError("estop must be a boolean")
        self.stopped |= estop
        if not np.isfinite(t_now) or (self.last_time is not None and
                not np.isclose(t_now - self.last_time, self.dt, rtol=1e-6, atol=1e-9)):
            raise ValueError("SafetyController requires a finite fixed-rate command clock")
        watch_time = t_now if watchdog_now is None else watchdog_now
        if not np.isfinite(watch_time) or (self.last_watchdog_time is not None and watch_time < self.last_watchdog_time):
            raise ValueError("Watchdog clock must be finite and monotonic")
        desired = None if target is None else np.asarray(target, dtype=float)
        if desired is not None and (desired.shape != self.q.shape or not np.isfinite(desired).all()):
            raise ValueError("Target must be a finite ordered joint vector")
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
            desired, phase = self.watchdog.select(desired, self.q, watch_time, t_capture)
            limits = self.spec.position_limits
            bounded = np.clip(desired, *limits.T)
            error = bounded - self.q
            requested_velocity = np.sign(error) * np.minimum(np.abs(error) / self.dt, self._braking_speed(np.abs(error)))
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
        self.last_watchdog_time = float(watch_time)
        self.events.extend(events)
        return SafeCommand(self.q.copy(), clamped, phase, events)
