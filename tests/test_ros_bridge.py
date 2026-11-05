"""Slice I bridge logic without ROS: packing, receiver validation, follower hold/limits, the README contract.

Scripted landmarks and simulated clocks only; nothing here is a measurement. The rclpy test at the end
skips without ROS and runs in ros.yml.
"""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_motion_file import scripted_clip
from test_runtime import Clock, Source, frame
from xert.backends import MuJoCoBackend
from xert.runtime.loop import run_loop
from xert.safety import SafetyController
from xert.sources.motion_file import MotionFileSource, load_clip
from xert.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros/xert_ros_bridge"))
from xert_ros_bridge.core import (CLIP_TAIL_S, COMMANDS, FUTURE_S, STALE_S, STAMP_RULE, STATES,  # noqa: E402
                                  Follower, open_pipeline, pack, publishable)

SPEC = ROOT / "xert/spec/so101.yaml"
PROFILE = ROOT / "configs/teleop/so101.json"
DT = .04
EPS = 1e-9
MOVES = [(.004 * k, .05 * k) for k in range(1, 31)]  # Same clip as the launch test: the gripper closes.


@pytest.fixture
def spec():
    return load_spec(SPEC)


@pytest.fixture
def follower(spec):
    backend = MuJoCoBackend(spec)
    yield Follower(spec, backend)
    backend.close()


def assert_within_limits(applied, spec):
    """Applied targets, one row per follower tick: position, velocity and acceleration limits."""
    applied = np.asarray(applied)
    velocity = np.diff(applied, axis=0) / DT
    assert np.all(applied >= spec.position_limits[:, 0] - EPS) and np.all(applied <= spec.position_limits[:, 1] + EPS)
    assert np.all(np.abs(velocity) <= spec.velocity_limits + EPS)
    assert np.all(np.abs(np.diff(velocity, axis=0)) <= spec.acceleration_limits * DT + 1e-6)


def phase_runs(phases):
    return [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p]


def assert_hold_return_rest(commands, rest, hold_ticks=8):
    """Command stream after camera loss, from positions alone (the launch test sees no phases):
    identical commands away from rest (hold), then monotonically closer to rest (return), ending at rest."""
    commands = np.asarray(commands)
    distance = np.abs(commands - rest).max(axis=1)
    same = np.r_[False, np.all(commands[1:] == commands[:-1], axis=1)]
    hold = next(i for i in range(hold_ticks, len(commands))
                if same[i - hold_ticks + 1:i + 1].all() and distance[i] > .1)
    assert np.all(np.diff(distance[hold:]) <= 1e-9)
    np.testing.assert_allclose(commands[-1], rest, atol=1e-9)


def test_hold_commands_without_frames_carry_generation_stamps(spec, tmp_path):
    mapper = SimpleNamespace(map=lambda f: spec.rest + .05)
    # First read returns None: the scripted clip's landmarks finish 5 ms after capture.
    _, frames = load_clip(scripted_clip(tmp_path, MOVES[:2]))
    clock = Clock()
    records = run_loop(MotionFileSource(frames, clock=clock), MuJoCoBackend(spec), mapper, .2,
                       clock=clock, sleep=clock.sleep)
    first = records[0]
    assert first["frame_sequence"] is None and first["phase"] == "hold" and publishable(first)
    names, positions, stamp = pack(first, spec.joint_names, 7 * 10**9)
    assert names == list(spec.joint_names) and positions == first["command"] == spec.rest.tolist()
    assert stamp == 7 * 10**9 + round(first["t_command"] * 1e9)
    # An empty-camera interval longer than stale_s: hold commands, each stamped when it was generated.
    clock = Clock()
    records = run_loop(Source([frame(100.), *[None] * 8, frame(100.35, 1)]), MuJoCoBackend(spec), mapper, .4,
                       clock=clock, sleep=clock.sleep)
    held = [r for r in records if r["phase"] == "hold"]
    assert held and all(r["frame_sequence"] == 0 and publishable(r) for r in held)
    stamps = [pack(r, spec.joint_names, 0)[2] for r in records]
    assert stamps == sorted(set(stamps)) == [round(r["t_command"] * 1e9) for r in records]


def test_receiver_validation_table(follower, spec):
    names, target = list(spec.joint_names), np.linspace(-.1, .1, 6)
    assert follower.accept(names[::-1], target[::-1].tolist(), 10., 10.) is None  # reordered: stored by name
    np.testing.assert_array_equal(follower.target, target)
    bad = {"nan": (names, [*target[:5], np.nan], 10.),
           "missing joint": (names[:5], target[:5], 10.),
           "duplicate joint": ([*names[:5], names[0]], target, 10.),
           "unknown joint": ([*names[:5], "elbow"], target, 10.),
           "length mismatch": (names, target[:5], 10.),
           "backward stamp": (names, target + .1, 9.99),
           "future stamp": (names, target + .1, 10. + FUTURE_S + 1e-3),
           "non-finite stamp": (names, target + .1, float("nan"))}
    for case, (n, p, stamp) in bad.items():
        assert follower.accept(n, list(p), stamp, 10.) is not None, case
        np.testing.assert_array_equal(follower.target, target)  # rejection changes neither target ...
        assert follower.stamp == 10., case  # ... nor its freshness
    assert follower.accept(names, target + .1, 10. + FUTURE_S, 10.) is None  # inside the stated tolerance
    assert follower.fresh(10. + FUTURE_S + STALE_S) and not follower.fresh(10. + FUTURE_S + STALE_S + 1e-6)


def test_coalesced_delivery_keeps_spec_limits(follower, spec):
    bridge = SafetyController(spec)  # The bridge's limiter chasing a far goal: a limit-saturated stream.
    goal = spec.position_limits[:, 1] * .9
    applied = [follower.applied]
    for k in range(80):
        t = k * DT
        command = bridge.step(goal, t_now=t, t_capture=t, watchdog_now=t).q
        if k % 2 == 0:  # every other command lost or coalesced by keep-last-1
            assert follower.accept(spec.joint_names, command, t, t) is None
        follower.step(t)
        applied.append(follower.applied)
    assert_within_limits(applied, spec)
    assert np.all(applied[-1] - applied[0] > .5)  # it did follow


def test_transport_loss_brakes_then_holds_without_returning(follower, spec):
    applied, measured = [follower.applied], []
    accel = spec.acceleration_limits[0]

    def tick(k, deliver):
        t = k * DT
        q = spec.rest.copy()
        q[0] = spec.velocity_limits[0] * t  # shoulder_pan ramp at its 1 rad/s velocity limit
        if deliver:
            assert follower.accept(spec.joint_names, q, t, t) is None
        measured.append(follower.step(t).q)
        applied.append(follower.applied)

    for k in range(40):
        tick(k, True)
    v0 = (applied[-1][0] - applied[-2][0]) / DT
    assert v0 == pytest.approx(spec.velocity_limits[0])
    gap, last_stamp = len(applied), 39 * DT
    for k in range(40, 110):  # 2.8 s of silence: past stale + hold + return (2.7 s) of a watchdog
        tick(k, False)
    pan = np.array(applied)[:, 0]
    # First tick whose command is stale: from then on the target is the last applied one. The joint is
    # still moving (it was braking toward the last command inside the 0.2 s fresh window), so it must
    # brake by exactly one acceleration step, not stop dead: (v - a*dt)*dt.
    stale = gap + next(i for i in range(70) if (40 + i) * DT - last_stamp > STALE_S)
    v_before = (pan[stale - 1] - pan[stale - 2]) / DT
    assert v_before > .1
    assert pan[stale] - pan[stale - 1] == pytest.approx((v_before - accel * DT) * DT)
    braking_ticks = int(np.ceil(v0 / (accel * DT))) + 1
    assert np.all(pan[stale + braking_ticks:] == pan[-1])  # then the applied target is constant ...
    assert pan[-1] - pan[gap - 1] <= v0 * (STALE_S + DT) + v0 ** 2 / (2 * accel) + EPS
    assert pan[-1] > .9  # ... and it never heads back to rest on its own
    assert abs(measured[-1][0] - pan[-1]) < .01  # the sim (measured) settled where the applied target holds
    for k in range(110, 135):  # commands return: tracking resumes, still inside the limits
        tick(k, True)
    assert applied[-1][0] > pan[-1] + .1
    assert_within_limits(applied, spec)


def test_delayed_backend_stamp_is_generation_time_and_overrun_is_not_published(follower, spec):
    clock, backend = Clock(), MuJoCoBackend(spec)
    send = backend.send_targets

    def stalled(q, t):
        clock.sleep(.3)
        send(q, t)

    backend.send_targets = stalled
    seen = []
    with pytest.raises(RuntimeError, match="exceeded control period"):
        run_loop(Source([frame()]), backend, SimpleNamespace(map=lambda f: spec.rest + .05), 1.,
                 clock=clock, sleep=clock.sleep, on_step=seen.append)
    record, = seen
    assert record["deadline_overrun"] and not publishable(record)
    names, positions, stamp_ns = pack(record, spec.joint_names, 0)
    assert stamp_ns == round(record["t_command"] * 1e9) and record["t_applied"] - record["t_command"] >= .3 - EPS
    before = follower.applied
    assert follower.accept(names, positions, stamp_ns / 1e9, clock()) is None  # valid, but 0.3 s old
    assert not follower.fresh(clock())
    follower.step(clock())
    np.testing.assert_array_equal(follower.applied, before)  # transport hold: the late target is not chased


@pytest.mark.parametrize("estop", [False, True])
def test_clip_chain_follower_executes_hold_return_rest(spec, tmp_path, estop):
    """Bridge loop -> pack -> follower in lockstep; the loop clock is the ROS clock (offset 0, no latency)."""
    clip = scripted_clip(tmp_path, MOVES)
    source, backend, mapper, duration = open_pipeline("clip", SPEC, PROFILE, clip=clip)
    _, frames = load_clip(clip)
    assert duration == pytest.approx(frames[-1].t_landmarks - frames[0].t_capture + CLIP_TAIL_S)
    clock = Clock()
    source.clock = clock  # before the first read, which anchors the clip to the loop clock
    follower = Follower(spec, MuJoCoBackend(spec))
    phases, commands = [], []

    def on_step(record):
        assert publishable(record)
        names, positions, stamp = pack(record, spec.joint_names, 0)
        assert follower.accept(names, positions, stamp / 1e9, clock()) is None
        follower.step(clock())
        phases.append(record["phase"])
        commands.append(positions)

    run_loop(source, backend, mapper, duration, clock=clock, sleep=clock.sleep, on_step=on_step,
             stop_requested=lambda: estop and "return" in phases)
    commands = np.array(commands)
    hold = commands[[p == "hold" for p in phases]][1:]  # [0] is the first-read-None tick at rest
    assert np.abs(hold - spec.rest).max() > .1  # camera loss left the arm away from rest, and it held there
    if estop:
        assert phase_runs(phases) == ["hold", "live", "hold", "return", "estop"]
    else:
        assert phase_runs(phases) == ["hold", "live", "hold", "return", "rest"]
        assert_hold_return_rest(commands, spec.rest)
        # Measured sim pose at rest: 0.6 mrad worst joint when written; 0.01 rad leaves headroom.
        np.testing.assert_allclose(follower.backend.read_state().q, spec.rest, atol=.01)
    # The bridge has exited: the follower keeps running (no horizon) and holds indefinitely.
    t0 = clock()
    held = []
    for k in range(1, 650):
        follower.step(t0 + k * DT)
        held.append(follower.applied)
    assert follower.tick > 600
    assert np.all(np.array(held[20:]) == held[-1])
    np.testing.assert_allclose(held[-1], commands[-1], atol=1e-3)
    if estop:
        assert np.abs(held[-1] - spec.rest).max() > .01  # held mid-return, not walked to rest
    follower.backend.close()


def test_topic_contract_matches_xert_ros_readme():
    text = (ROOT / "ros/xert_ros/README.md").read_text(encoding="utf-8")
    rows = {}
    for line in text.splitlines():
        cells = line.strip().strip("|").split("|")
        if len(cells) == 4 and re.fullmatch(r"`/[\w/]+`", cells[0].strip()):
            rows[cells[0].strip(" `")] = cells
    for name, kind, reliability, durability, depth in (COMMANDS, STATES):
        _, row_kind, _, row_qos = rows[name]
        assert re.findall(r"`([^`]+)`", row_kind)[0] == kind
        row_qos = row_qos.lower()
        assert ("best effort" in row_qos, "reliable" in row_qos) == (reliability == "best_effort",
                                                                     reliability == "reliable")
        assert durability in row_qos and f"keep last {depth}" in row_qos
    assert STAMP_RULE in " ".join(text.split())


def test_ros_messages_and_qos_follow_the_contract(follower, spec):
    pytest.importorskip("rclpy")
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
    from xert_ros_bridge import bridge_node, follower_node  # noqa: F401  (both nodes import)
    from xert_ros_bridge.ros_io import joint_state, qos, seconds
    commands, states = qos(COMMANDS), qos(STATES)
    assert (commands.reliability, commands.durability, commands.depth) == (
        ReliabilityPolicy.RELIABLE, DurabilityPolicy.VOLATILE, 1)
    assert (states.reliability, states.durability, states.depth) == (
        qos_profile_sensor_data.reliability, qos_profile_sensor_data.durability, qos_profile_sensor_data.depth)
    offset = 1_700_000_000 * 10**9
    record = {"command": [.1] * 6, "t_command": 12.345678901, "deadline_overrun": False}
    msg = joint_state(*pack(record, spec.joint_names, offset))
    assert list(msg.name) == list(spec.joint_names) and list(msg.position) == [.1] * 6
    assert msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec == offset + 12_345_678_901
    assert follower.accept(msg.name, msg.position, seconds(msg.header.stamp), seconds(msg.header.stamp)) is None
    assert bridge_node.parse(["--spec=s", "--profile=p", "--clip="]).source == "clip"
