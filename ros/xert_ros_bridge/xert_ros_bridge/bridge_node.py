"""xert_bridge: XERT's run_loop (source -> CalibratedMapper -> SafetyController) published on ROS 2.

Every applied command goes out from run_loop's on_step as a position-only JointState on
/xert/joint_commands, header.stamp = the command's generation time (core.pack). The node only
publishes; its executor spins on a second thread so run_loop's wall-clock deadlines never wait on ROS.
SIGINT/SIGTERM become run_loop's e-stop: one final e-stop command is published, then the node exits.
"""
import argparse
import signal
import sys
import threading
import time

import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState

from xert.runtime.loop import run_loop

from .core import COMMANDS, open_pipeline, pack, publishable
from .ros_io import joint_state, qos


def parse(argv):
    parser = argparse.ArgumentParser(prog='xert_bridge', description=__doc__.splitlines()[0])
    parser.add_argument('--source', choices=['clip', 'webcam'], default='clip')
    parser.add_argument('--clip', default='', help='Landmark clip (record-motion or scripted); --source clip')
    parser.add_argument('--spec', required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--calibration', default='', help='Required for webcam; a clip calibrates itself')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', default='', help='MediaPipe hand model (webcam); default build/models')
    parser.add_argument('--duration', type=float, default=0,
                        help='Seconds; 0 = clip length plus the safety return-to-rest tail, or 300 s for webcam')
    parser.add_argument('--wait-subscribers', type=int, default=1,
                        help='Start run_loop only once this many subscribers matched the command topic')
    return parser.parse_args(argv)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    args = parse(remove_ros_args(argv)[1:])
    rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    node = rclpy.create_node('xert_bridge')
    publisher = node.create_publisher(JointState, COMMANDS[0].lstrip('/'), qos(COMMANDS))
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    halt = threading.Event()

    def spin():
        # spin_once with a timeout, not executor.spin(): Executor.shutdown() destroys the guard condition
        # without waiting for a spinning thread (rclpy jazzy executors.py), so the thread stops first.
        while not halt.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=.1)

    spinner = threading.Thread(target=spin, name='xert_bridge_spin')
    spinner.start()
    try:
        source, backend, mapper, duration = open_pipeline(
            args.source, args.spec, args.profile, clip=args.clip or None, calibration=args.calibration or None,
            duration=args.duration or None, camera=args.camera, model=args.model or None)
        names = list(backend.spec.joint_names)
        # One offset for the whole run: ROS clock minus run_loop's perf_counter clock. On Linux NTP slews
        # CLOCK_MONOTONIC (perf_counter) and CLOCK_REALTIME (ROS time) at the same rate, so it does not
        # drift; only a realtime step changes it, and both directions fail safe: forward ages commands
        # (stale, follower holds), backward by > FUTURE_S makes them future (rejected with a warning).
        offset_ns = node.get_clock().now().nanoseconds - time.perf_counter_ns()
        node.get_logger().info(f'clock offset (ROS time - perf_counter) = {offset_ns} ns')
        while publisher.get_subscription_count() < args.wait_subscribers and not stop.is_set():
            time.sleep(.05)
        if stop.is_set():  # Stopped before any command: run_loop never ran, so close what it would have.
            source.close()
            backend.close()
            return
        sent = []

        def on_step(record):
            if publishable(record):
                publisher.publish(joint_state(*pack(record, names, offset_ns)))
                sent.append(record['phase'])

        run_loop(source, backend, mapper, duration, on_step=on_step, stop_requested=stop.is_set)
        # The last command (rest or e-stop) is the one nothing replaces: let reliable delivery finish.
        publisher.wait_for_all_acked(Duration(seconds=1))
        node.get_logger().info(f'published {len(sent)} commands; last phase {sent[-1] if sent else None}')
    finally:
        # Order matters (CI run 41994577985 aborted with SIGABRT at exit): stop and join the spin thread,
        # then shut the executor down, destroy the node, and only then shut down the context.
        halt.set()
        spinner.join()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
