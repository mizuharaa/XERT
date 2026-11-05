"""xert_sim_follower: /xert/joint_commands -> continuous MuJoCo sim -> /joint_states (SensorDataQoS).

A visualization stand-in, not a controller: see core.Follower. Callbacks run on one executor thread,
so the subscription and the 25 Hz timer never interleave.
"""
import argparse
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
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
    rclpy.init(args=argv)
    node = rclpy.create_node('xert_sim_follower')
    publisher = node.create_publisher(JointState, STATES[0].lstrip('/'), qos(STATES))

    def on_command(msg):
        now = node.get_clock().now().nanoseconds * 1e-9
        reason = follower.accept(msg.name, msg.position, seconds(msg.header.stamp), now)
        if reason is not None:
            node.get_logger().warning(f'rejected command: {reason}')

    def tick():
        now = node.get_clock().now()
        state = follower.step(now.nanoseconds * 1e-9)
        publisher.publish(joint_state(follower.names, state.q, now.nanoseconds, state.qd))

    node.create_subscription(JointState, COMMANDS[0].lstrip('/'), on_command, qos(COMMANDS))
    node.create_timer(1 / spec.control_hz, tick)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        backend.close()


if __name__ == '__main__':
    main()
