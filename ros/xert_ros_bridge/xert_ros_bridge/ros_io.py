"""rclpy conversions shared by both nodes: QoS from the contract tuples, JointState packing, stamps."""
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState


def qos(topic):
    """QoSProfile for a core.COMMANDS / core.STATES tuple (keep last <depth>)."""
    _, _, reliability, durability, depth = topic
    return QoSProfile(depth=depth,
                      reliability={'reliable': ReliabilityPolicy.RELIABLE,
                                   'best_effort': ReliabilityPolicy.BEST_EFFORT}[reliability],
                      durability={'volatile': DurabilityPolicy.VOLATILE}[durability])


def joint_state(names, positions, stamp_ns, velocities=None):
    msg = JointState()
    msg.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    msg.name = list(names)
    msg.position = [float(q) for q in positions]
    if velocities is not None:
        msg.velocity = [float(v) for v in velocities]
    return msg


def seconds(stamp):
    """builtin_interfaces/Time -> float seconds on the ROS clock."""
    return stamp.sec + stamp.nanosec * 1e-9
