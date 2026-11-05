"""xert_sim_follower: /xert/joint_commands -> continuous MuJoCo sim -> /joint_states (SensorDataQoS).

A visualization stand-in, not a controller: see core.Follower. Callbacks run on one executor thread,
so the subscription and the 25 Hz timer never interleave.
"""
import argparse
from collections import Counter
import signal
import sys
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState

from xert.backends import MuJoCoBackend
from xert.spec import load_spec

from .core import COMMANDS, STATES, Follower
from .ros_io import joint_state, qos, seconds


def main(argv=None):
    argv = sys.argv if argv is None else argv
    parser = argparse.ArgumentParser(prog='xert_sim_follower', description=__doc__.splitlines()[0])
    parser.add_argument('--spec', required=True)
    args = parser.parse_args(remove_ros_args(argv)[1:])
    spec = load_spec(args.spec)
    backend = MuJoCoBackend(spec)
    follower = Follower(spec, backend)
    # Own SIGINT/SIGTERM handling, as in xert_bridge: the context stays valid until the summary is logged.
    rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
    halt = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: halt.set())
    node = rclpy.create_node('xert_sim_follower')
    publisher = node.create_publisher(JointState, STATES[0].lstrip('/'), qos(STATES))

    # Diagnostics (CI run 41994577985 under-tracked): tick rate vs wall time, how many ticks had a fresh
    # command, and how old commands were on arrival. Logged every 125 ticks and at shutdown.
    stats = {'ticks': 0, 'fresh_ticks': 0, 'accepted': 0, 'rejected': Counter(), 'ages': []}
    started = time.monotonic()

    def summary():
        elapsed = time.monotonic() - started
        ages = sorted(stats['ages'])
        age = f"p50 {ages[len(ages) // 2]:.4f} s max {ages[-1]:.4f} s" if ages else 'n/a'
        node.get_logger().info(
            f"{stats['ticks']} ticks in {elapsed:.1f} s wall ({stats['ticks'] / max(elapsed, 1e-9):.1f} Hz); "
            f"fresh command on {stats['fresh_ticks']} ticks; accepted {stats['accepted']} "
            f"(age at receipt {age}); rejected {dict(stats['rejected'])}")

    def on_command(msg):
        now = node.get_clock().now().nanoseconds * 1e-9
        stamp = seconds(msg.header.stamp)
        reason = follower.accept(msg.name, msg.position, stamp, now)
        if reason is not None:
            stats['rejected'][reason] += 1
            node.get_logger().warning(f'rejected command: {reason}')
        else:
            stats['accepted'] += 1
            stats['ages'].append(now - stamp)

    def tick():
        now = node.get_clock().now()
        stats['fresh_ticks'] += follower.fresh(now.nanoseconds * 1e-9)
        state = follower.step(now.nanoseconds * 1e-9)
        publisher.publish(joint_state(follower.names, state.q, now.nanoseconds, state.qd))
        stats['ticks'] += 1
        if stats['ticks'] % 125 == 0:
            summary()

    node.create_subscription(JointState, COMMANDS[0].lstrip('/'), on_command, qos(COMMANDS))
    node.create_timer(1 / spec.control_hz, tick)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while not halt.is_set() and rclpy.ok():  # on this thread, so it has stopped before the teardown
            executor.spin_once(timeout_sec=.1)
    except ExternalShutdownException:
        pass
    finally:
        summary()  # before the context goes: rosout still has a valid publisher
        # Same teardown order as xert_bridge: executor, node, context, then the sim.
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
        backend.close()


if __name__ == '__main__':
    main()
