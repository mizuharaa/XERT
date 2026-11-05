"""ROS-free bridge logic: the topic contract, command packing and the sim follower tick.

Nothing here imports rclpy, so the base venv tests it without ROS. The two nodes are thin rclpy shells
around these functions.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from xert.safety import SafetyController

# (name, type, reliability, durability, keep-last depth): must equal ros/xert_ros/README.md's topic table.
COMMANDS = ("/xert/joint_commands", "sensor_msgs/msg/JointState", "reliable", "volatile", 1)
STATES = ("/joint_states", "sensor_msgs/msg/JointState", "best_effort", "volatile", 5)
STAMP_RULE = ("stamp = generation time of the safety-checked command; "
              "capture freshness is enforced upstream by the safety layer")

STALE_S = .2  # Transport freshness: now - header.stamp above this means the command is late or lost.
# A stamp may lead the follower's clock by this much: half a 25 Hz period covers the bridge's one-time
# clock-offset measurement (two consecutive clock reads) on one host. Across hosts, sync clocks first.
FUTURE_S = .02
# SafetyController's stale + hold + return defaults (xert/safety/limits.py:33), plus 1 s to settle at rest:
# a clip run ends only after the safety layer has walked the arm back to rest.
CLIP_TAIL_S = .2 + .5 + 2.0 + 1.0


def open_pipeline(source, spec, profile, *, clip=None, calibration=None, duration=None, camera=0, model=None):
    """(motion source, MuJoCo backend, CalibratedMapper, duration) for run_loop: the P3.5 loop, unchanged.

    clip: a recorded or scripted landmark clip; it calibrates itself from its first frame showing every
    mapped hand (as P3.6 and hw-dry-run) unless a calibration file is given. webcam: needs a calibration
    from `xert calibrate` for that camera and the pinned model; it opens the camera last.
    """
    from xert.backends import MuJoCoBackend
    from xert.retarget.calibrate import CalibratedMapper, create_calibration, load_profile, required_hands
    from xert.spec import load_spec
    spec, profile = Path(spec), Path(profile)
    inputs = {"spec_bytes": spec.read_bytes(), "profile_bytes": profile.read_bytes()}
    loaded = load_spec(spec, contents=inputs["spec_bytes"])
    backend = MuJoCoBackend(loaded)
    try:
        if source == "clip":
            from xert.sources.motion_file import MotionFileSource, load_clip
            if not clip:
                raise ValueError("--source clip needs --clip <landmark clip>")
            _, frames = load_clip(clip)
            if calibration is None:
                hands = required_hands(load_profile(profile, loaded, contents=inputs["profile_bytes"]))
                neutral = next((f for f in frames if hands <= set(f.hands)), None)
                if neutral is None:
                    raise ValueError("Clip never shows every mapped hand; cannot calibrate")
                calibration = create_calibration(spec, profile, neutral, backend, **inputs)
            else:
                calibration = json.loads(Path(calibration).read_bytes())
            mapper = CalibratedMapper(spec, profile, calibration, backend, **inputs)
            if duration is None:
                duration = frames[-1].t_landmarks - frames[0].t_capture + CLIP_TAIL_S
            return MotionFileSource(frames), backend, mapper, duration
        if source == "webcam":
            from xert.runtime.cli import ROOT, _calibration, _pinned_model
            from xert.sources.webcam_mediapipe import WebcamSource
            model = ROOT / "build/models/hand_landmarker.task" if model is None else Path(model)
            if calibration is None:
                raise ValueError("--source webcam needs --calibration from `xert calibrate`")
            pinned = _pinned_model(model)
            _, calibration = _calibration(SimpleNamespace(calibration=Path(calibration), camera=camera), pinned)
            mapper = CalibratedMapper(spec, profile, calibration, backend, **inputs)
            return (WebcamSource(model, pinned["sha256"], camera=camera), backend, mapper,
                    300. if duration is None else duration)
        raise ValueError(f"Unknown source {source!r}; expected clip or webcam")
    except BaseException:
        backend.close()
        raise


def publishable(record):
    """A run_loop record whose step overran its period is never published (run_loop aborts after it)."""
    return not record["deadline_overrun"]


def pack(record, joint_names, offset_ns):
    """(names, positions, stamp_ns) of one applied command, positions in spec order.

    stamp = record["t_command"], the perf_counter instant run_loop generated the command (before
    send_targets), moved to the ROS clock by offset_ns = ROS now - perf_counter, measured once at node start.
    A backend stall between generation and publish therefore ages the command instead of hiding.
    """
    return list(joint_names), [float(q) for q in record["command"]], offset_ns + round(record["t_command"] * 1e9)


class Follower:
    """Visualization stand-in for a robot, NOT a controller: it adds no control law.

    Each tick it picks an effective target (the accepted command while transport is fresh, else the target
    it applied last tick), passes it through its own SafetyController used only as the spec's position clip
    and velocity/acceleration limiter, and steps a continuous MuJoCoBackend (no task, episode or horizon).
    The limiter's watchdog is fed t_capture = watchdog_now, so it never holds or returns to rest by itself:
    return to rest only ever arrives as commands from the bridge's safety layer. `applied` is what the
    follower sends to the sim; `backend.read_state()` is what the sim measured.
    """

    def __init__(self, spec, backend):
        self.spec, self.backend = spec, backend
        self.names = list(spec.joint_names)
        self.limiter = SafetyController(spec, backend.read_state().q)
        self.applied = self.limiter.q.copy()
        self.target = self.stamp = None  # Accepted target (spec order) and its generation stamp, seconds.
        self.tick = 0

    def accept(self, names, positions, stamp, now):
        """Validate one command (stamp and now in seconds, ROS clock). Return None if accepted, else why not.

        A rejected message changes neither the accepted target nor its freshness.
        """
        names = list(names)
        if len(names) != len(positions):
            return "name/position length mismatch"
        if len(set(names)) != len(names) or set(names) != set(self.names):
            return "joint names must cover the spec joints exactly once"
        by_name = dict(zip(names, positions))
        target = np.array([by_name[name] for name in self.names], dtype=float)
        if not np.isfinite(target).all():
            return "non-finite position"
        if not math.isfinite(stamp) or stamp > now + FUTURE_S:
            return "stamp is non-finite or in the future"
        if self.stamp is not None and stamp < self.stamp:
            return "stamp moved backwards"
        self.target, self.stamp = target, stamp
        return None

    def fresh(self, now):
        return self.stamp is not None and now - self.stamp <= STALE_S

    def step(self, now):
        """One control tick at the spec rate; return the measured sim JointState."""
        target = self.target if self.fresh(now) else self.applied  # Transport hold: keep the last applied.
        t = self.tick * self.limiter.dt
        self.applied = self.limiter.step(target, t_now=t, t_capture=t, watchdog_now=t).q
        self.backend.send_targets(self.applied, t)
        self.tick += 1
        return self.backend.read_state()
