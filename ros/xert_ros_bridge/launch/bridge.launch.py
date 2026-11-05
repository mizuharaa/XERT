"""XERT bridge -> /xert/joint_commands -> sim follower -> /joint_states -> robot_state_publisher (+ RViz)."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from xert.runtime.cli import ROOT


def generate_launch_description():
    share = Path(get_package_share_directory('xert_ros'))
    defaults = {
        'source': ('clip', 'clip (CI path) or webcam'),
        'clip': ('', 'Landmark clip for source:=clip'),
        'spec': (str(ROOT / 'xert/spec/so101.yaml'), 'XERT embodiment spec'),
        'profile': (str(ROOT / 'configs/teleop/so101.json'), 'Retargeting profile'),
        'calibration': ('', 'Neutral calibration; required for webcam, a clip calibrates itself'),
        'wait_subscribers': ('1', 'Bridge starts once this many command subscribers matched'),
        'use_rviz': ('true', 'Start RViz 2.'),
    }
    config = {name: LaunchConfiguration(name) for name in defaults}
    bridge_args = ('source', 'clip', 'spec', 'profile', 'calibration', 'wait_subscribers')
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value, description=text)
          for name, (value, text) in defaults.items()],
        # `--name=value` keeps an empty value (clip, calibration) one argument.
        Node(package='xert_ros_bridge', executable='xert_bridge', output='screen',
             arguments=[[f'--{name.replace("_", "-")}=', config[name]] for name in bridge_args]),
        # Swap for Daniel's node once its control loop exists (see docs/handoff/I-ros-bridge.md).
        Node(package='xert_ros_bridge', executable='xert_sim_follower', output='screen',
             arguments=[['--spec=', config['spec']]]),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': (share / 'urdf' / 'so101.urdf').read_text()}]),
        Node(package='rviz2', executable='rviz2', arguments=['-d', str(share / 'rviz' / 'xert.rviz')],
             condition=IfCondition(config['use_rviz'])),
    ])
