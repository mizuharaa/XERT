"""Start the SO-101 description, the XERT node and (optionally) RViz 2."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory('xert_ros'))
    robot_description = (share / 'urdf' / 'so101.urdf').read_text()
    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true', description='Start RViz 2.'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(package='xert_ros', executable='xert_node', output='screen'),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', str(share / 'rviz' / 'xert.rviz')],
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
    ])
