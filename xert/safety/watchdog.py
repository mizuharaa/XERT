"""Stale input means hold, then a smooth requested return to rest."""
import numpy as np


class Watchdog:
    def __init__(self, rest, stale_s, hold_s, return_s):
        if any(not np.isfinite(v) or v < 0 for v in (stale_s, hold_s, return_s)) or return_s == 0:
            raise ValueError("Watchdog intervals must be finite and nonnegative; return_s must be positive")
        self.rest = rest.copy()
        self.stale_s, self.hold_s, self.return_s = stale_s, hold_s, return_s
        self.last_capture = None
        self.last_live = None
        self.started = None
        self.hold = None

    def select(self, target, current, now, capture):
        if self.started is None:
            self.started = now
        if capture is not None:
            if not np.isfinite(capture) or capture > now + 1e-9:
                raise ValueError("Capture timestamp must be finite and no later than command time")
            if target is None:
                raise ValueError("A capture timestamp requires a target")
            if self.last_capture is not None and capture < self.last_capture:
                raise ValueError("Capture timestamps must not move backwards")
            self.last_capture = capture
        elif target is not None:
            raise ValueError("A target requires its capture timestamp")
        fresh = target is not None and now - capture <= self.stale_s
        if fresh:
            self.last_live = capture
        last = self.started if self.last_live is None else self.last_live
        age = now - last
        if fresh:
            self.hold = None
            return target, "live"
        if self.hold is None:
            self.hold = current.copy()
        returning = age - self.stale_s - self.hold_s
        if returning <= 0:
            return self.hold, "hold"
        fraction = min(returning / self.return_s, 1.0)
        smooth = fraction * fraction * (3 - 2 * fraction)
        return (1 - smooth) * self.hold + smooth * self.rest, "rest" if fraction == 1 else "return"
