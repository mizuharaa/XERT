"""bridge.launch.py end to end on Jazzy: a scripted clip through xert_bridge and xert_sim_follower.

Launches the real launch file (use_rviz:=false). Scripted landmarks and a sim follower: a contract
check, not a measurement. Runs under `colcon test` in ros.yml.
"""
import importlib.util
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_source import LaunchDescriptionSource
import launch_testing
import launch_testing.actions
import launch_testing.asserts
from launch_ros.actions import Node
import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, ReliabilityPolicy
from sensor_msgs.msg import JointState
import xert
from xert.sources.motion_file import load_clip
from xert.spec import load_spec
from xert_ros_bridge.core import CLIP_TAIL_S, COMMANDS, STATES
from xert_ros_bridge.ros_io import joint_state, qos

ROOT = Path(xert.__file__).resolve().parents[1]  # ros.yml installs XERT editable from this checkout.
sys.path.insert(0, str(ROOT / 'tests'))
from test_motion_file import scripted_clip  # noqa: E402
from test_ros_bridge import assert_hold_return_rest, MOVES, replay_follower  # noqa: E402

SPEC = load_spec(ROOT / 'xert/spec/so101.yaml')
NAMES = list(SPEC.joint_names)
CLIP = scripted_clip(Path(tempfile.mkdtemp()), MOVES)
EXPECTED_TOPICS = {COMMANDS[0], STATES[0], '/robot_description', '/tf', '/tf_static', '/rosout',
                   '/parameter_events'}


def bridge_launch():
    path = Path(get_package_share_directory('xert_ros_bridge')) / 'launch' / 'bridge.launch.py'
    spec = importlib.util.spec_from_file_location('bridge_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


@pytest.mark.launch_test
def generate_test_description():
    described = bridge_launch()
    nodes = {e.node_executable: e for e in described.entities if isinstance(e, Node)}
    # wait_subscribers 2: the follower and this test, so neither misses the first command.
    include = IncludeLaunchDescription(LaunchDescriptionSource(described), launch_arguments={
        'clip': str(CLIP), 'use_rviz': 'false', 'wait_subscribers': '2'}.items())
    return (LaunchDescription([include, launch_testing.actions.ReadyToTest()]),
            {'bridge': nodes['xert_bridge'], 'follower': nodes['xert_sim_follower']})


def wait_for(condition, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(.02)
    return condition()


class TestBridgeLaunch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('bridge_launch_test')
        cls.executor = SingleThreadedExecutor()
        cls.executor.add_node(cls.node)
        cls.commands, cls.states = [], []
        cls.command_times, cls.command_ages = [], []

        def on_command(msg):
            cls.command_times.append(time.monotonic_ns())
            # Age at receipt on this node's ROS clock: the follower judges freshness the same way.
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            cls.command_ages.append(cls.node.get_clock().now().nanoseconds - stamp)
            cls.commands.append(msg)

        # Record /joint_states first: the bridge is released (wait_subscribers:=2) only once the command
        # subscription below exists, so the state window covers the whole run (run 41995929551 matched the
        # best-effort state subscription late and missed the gripper peak).
        states = cls.node.create_subscription(JointState, STATES[0], cls.states.append, qos_profile_sensor_data)
        threading.Thread(target=cls.executor.spin, daemon=True).start()
        if not wait_for(lambda: states.get_publisher_count() >= 1 and cls.states, 30):
            raise RuntimeError('no /joint_states from the follower within 30 s')
        cls.node.create_subscription(JointState, COMMANDS[0], on_command, qos(COMMANDS))

    @classmethod
    def tearDownClass(cls):
        cls.executor.shutdown()
        cls.node.destroy_node()
        rclpy.shutdown()

    def endpoints(self, topic, publishers):
        info = (self.node.get_publishers_info_by_topic(topic) if publishers else
                self.node.get_subscriptions_info_by_topic(topic))
        return {(i.node_name, i.topic_type, i.qos_profile.reliability, i.qos_profile.durability)
                for i in info}

    def assertEndpoints(self, topic, publishers, expected, key=lambda e: e, timeout=10.):
        """Graph discovery lags in each process: poll until the endpoint set is complete, then compare sets.

        No spin_once here: the class executor already spins this node on its own thread, and the graph
        cache rmw fills is read without spinning.
        """
        wait_for(lambda: {key(e) for e in self.endpoints(topic, publishers)} >= expected, timeout)
        self.assertEqual({key(e) for e in self.endpoints(topic, publishers)}, expected,
                         f'{topic} {"publishers" if publishers else "subscriptions"} after {timeout} s')

    def latest(self):
        return np.array(self.states[-1].position)

    def test_end_to_end(self, proc_info, bridge):
        node, rest = self.node, SPEC.rest
        reliable, best_effort = ReliabilityPolicy.RELIABLE, ReliabilityPolicy.BEST_EFFORT
        volatile = DurabilityPolicy.VOLATILE
        # Start after discovery: the bridge publishes only once its publisher matched both command
        # subscribers (wait_subscribers:=2), and the follower has published at least one state.
        self.assertTrue(wait_for(lambda: self.commands, 30), 'no command on /xert/joint_commands')
        self.assertTrue(wait_for(lambda: self.states, 10), 'no state on /joint_states')

        # Graph: exactly the expected topics, endpoints and QoS (slice E's introspection approach).
        # The bridge's publisher first: it lives only for the clip plus its return-to-rest tail.
        self.assertEndpoints(COMMANDS[0], True, {('xert_bridge', COMMANDS[1], reliable, volatile)})
        self.assertEndpoints(COMMANDS[0], False, {('bridge_launch_test', COMMANDS[1], reliable, volatile),
                                                  ('xert_sim_follower', COMMANDS[1], reliable, volatile)})
        self.assertEndpoints(STATES[0], True, {('xert_sim_follower', STATES[1], best_effort, volatile)})
        # robot_state_publisher's own QoS choice is its business: check that it listens, by name.
        self.assertEndpoints(STATES[0], False, {'robot_state_publisher', 'bridge_launch_test'},
                             key=lambda e: e[0])
        self.assertTrue(wait_for(lambda: {n for n, _ in node.get_topic_names_and_types()} == EXPECTED_TOPICS,
                                 10), node.get_topic_names_and_types())

        # The clip runs to its end, then the safety layer's hold -> return -> rest; no deadline abort.
        proc_info.assertWaitForShutdown(process=bridge, timeout=30)
        _, frames = load_clip(CLIP)
        expected = int(np.ceil((frames[-1].t_landmarks - frames[0].t_capture + CLIP_TAIL_S) * SPEC.control_hz))
        commands = list(self.commands)
        self.assertGreaterEqual(len(commands), expected - 2)
        self.assertTrue(all(list(m.name) == NAMES and len(m.position) == len(NAMES) for m in commands))
        stamps = np.array([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for m in commands])
        self.assertTrue(np.all(np.diff(stamps) > 0), 'stamps not monotonic')
        # Rate from generation stamps, which discovery cannot shift; the receive-side rate is taken over
        # the stream after its first command, so it also excludes the discovery wait.
        self.assertAlmostEqual(float(np.median(np.diff(stamps))), 1 / SPEC.control_hz, delta=.005)
        received = np.array(self.command_times[1:len(commands)]) * 1e-9
        self.assertAlmostEqual(float(np.median(np.diff(received))), 1 / SPEC.control_hz, delta=.01)
        positions = np.array([m.position for m in commands])
        assert_hold_return_rest(positions, rest)

        # The follower executed it: limits held and its measured pose reached rest.
        states = np.array([m.position for m in self.states])
        self.assertTrue(all(list(m.name) == NAMES for m in self.states))
        self.assertTrue(np.all(states >= SPEC.position_limits[:, 0] - 1e-6) and
                        np.all(states <= SPEC.position_limits[:, 1] + 1e-6))
        # Diagnostics first, so a failure below says which side lost motion: the follower's tick rate
        # (header stamps of /joint_states) or the command ages it judged freshness on.
        commanded = np.abs(positions - rest).max(axis=0)
        measured = np.abs(states - rest).max(axis=0)
        ages = np.array(self.command_ages[:len(commands)]) * 1e-9
        state_stamps = np.array([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for m in self.states])
        report = (f'max |cmd - rest| {commanded.round(4)}; max |state - rest| {measured.round(4)}; '
                  f'command age at receipt p50 {np.median(ages):.4f} s max {ages.max():.4f} s; '
                  f'/joint_states spacing p50 {np.median(np.diff(state_stamps)):.4f} s '
                  f'max {np.diff(state_stamps).max():.4f} s over {len(self.states)} states; '
                  # Late start vs best-effort loss: contiguous stamps (max spacing one period) that begin
                  # after the first command mean recording started late; gaps mean samples were dropped.
                  f'states span {state_stamps[0] - stamps[0]:+.3f} .. {state_stamps[-1] - stamps[-1]:+.3f} s '
                  f'around the first .. last command = '
                  f'{round((state_stamps[-1] - state_stamps[0]) * SPEC.control_hz) + 1} follower ticks, '
                  f'{len(state_stamps)} received')
        print(report, flush=True)
        # The window covers the run, and silent best-effort loss cannot hide in it.
        self.assertLess(state_stamps[0], stamps[0], report)
        ticks = round((state_stamps[-1] - state_stamps[0]) * SPEC.control_hz) + 1
        self.assertGreaterEqual(len(state_stamps), .8 * ticks, report)
        # The follower is a 25 Hz node: its state stamps must be one control period apart.
        self.assertAlmostEqual(float(np.median(np.diff(state_stamps))), 1 / SPEC.control_hz, delta=.005, msg=report)
        # Per stamp, not peak vs peak: replay core.Follower offline over every received command at the
        # follower's own tick stamps; each measured state must match the replayed one. Same sim, so actuator
        # lag cancels; the bound is one tick of phase at the velocity limit (a command that arrived just
        # after a tick in one run and just before it in the replay).
        commands_s = sorted(((list(m.name), list(m.position), m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
                             for m in commands), key=lambda c: c[2])
        order = np.argsort(state_stamps)
        expected_states = replay_follower(SPEC, commands_s, list(state_stamps[order]))
        np.testing.assert_array_less(np.abs(states[order] - expected_states), SPEC.velocity_limits / SPEC.control_hz,
                                     err_msg=report)
        self.assertTrue(wait_for(lambda: np.abs(self.latest() - rest).max() < .01, 5), self.latest())

        # Malformed commands are rejected; a reordered valid stream proves the subscription is live.
        publisher = node.create_publisher(JointState, COMMANDS[0], qos(COMMANDS))
        self.assertTrue(wait_for(lambda: publisher.get_subscription_count() >= 2, 10))
        far = rest.copy()
        far[0] = .5

        def send(names, values, ahead_s=0., count=10):
            for _ in range(count):
                stamp = node.get_clock().now().nanoseconds + round(ahead_s * 1e9)
                publisher.publish(joint_state(names, values, stamp))
                time.sleep(1 / SPEC.control_hz)

        # (Backward stamps are only in the ROS-free table: a fresh-but-backward stamp needs a ~10 ms window.)
        send(NAMES, [*far[:5], float('nan')])
        send(NAMES[:5], far[:5])
        send(NAMES, far, ahead_s=1.)
        time.sleep(.3)
        self.assertLess(abs(self.latest()[0] - rest[0]), .02, 'a malformed command moved the arm')
        send(NAMES[::-1], far[::-1], count=30)
        self.assertTrue(wait_for(lambda: self.latest()[0] > .3, 2), self.latest())

        # No horizon: the follower keeps publishing past 600 ticks and holds without commands.
        self.assertTrue(wait_for(lambda: len(self.states) > 600, 40), len(self.states))
        recent = np.array([m.position for m in self.states[-25:]])
        self.assertLess(np.abs(recent - recent[-1]).max(), 1e-3)
        self.assertGreater(recent[-1][0], .3)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info, bridge, follower):
        launch_testing.asserts.assertExitCodes(proc_info, process=bridge)
        launch_testing.asserts.assertExitCodes(proc_info, process=follower)
