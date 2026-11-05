from glob import glob

from setuptools import setup

package_name = 'xert_ros_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mizuharaa',
    maintainer_email='84390611+mizuharaa@users.noreply.github.com',
    description='XERT retargeting -> /xert/joint_commands, and a MuJoCo sim follower -> /joint_states.',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'xert_bridge = xert_ros_bridge.bridge_node:main',
        'xert_sim_follower = xert_ros_bridge.follower_node:main',
    ]},
)
