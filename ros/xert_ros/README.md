# xert_ros

A ROS 2 Jazzy (`ament_cmake`) package for the XERT SO-101 arm. It contains:

- `urdf/so101.urdf`: the official TheRobotStudio SO-101 URDF, pinned and
  licensed (see `urdf/SOURCE.json` and `urdf/LICENSE`), with mesh paths
  rewritten to `package://xert_ros/meshes/` and joint limits set to
  `xert/spec/so101.yaml`.
- `xert_node`: a C++17 `rclcpp` node (`XertNode`). It declares its
  parameters, the topics below and a wall timer. **The control loop is not
  written yet**; see "TODO(daniel)" below.
- `launch/xert.launch.py`: `robot_state_publisher` with the URDF, `xert_node`,
  and RViz 2 with `rviz/xert.rviz` (RobotModel + TF, fixed frame `base_link`).

The meshes are not copied into this package. `CMakeLists.txt` installs the
SO-101 STL files already vendored for MuJoCo
(`assets/menagerie/robotstudio_so101/assets`, Apache-2.0) into
`share/xert_ros/meshes`, with their LICENSE. The 13 meshes the URDF uses are
byte-identical to the SO-ARM100 ones at the pinned commit (same git blob
hashes).

## Parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `joint_names` | string[] | `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` | Joint order for every message (LeRobot `so101_follower` order, same as `xert/spec/so101.yaml`) |
| `loop_rate_hz` | double | `25.0` | Control loop and publish rate (`control_hz` in the spec). Must be > 0. |

## Topics

| Name | Type | Direction | QoS |
|---|---|---|---|
| `/joint_states` | `sensor_msgs/msg/JointState` | published | `SensorDataQoS`: best effort, volatile, keep last 5 |
| `/xert/joint_commands` | `sensor_msgs/msg/JointState` (position only) | subscribed | reliable, volatile, keep last 1 |
| `/robot_description`, `/tf`, `/tf_static` | (from `robot_state_publisher`) | published | `robot_state_publisher` defaults |

Both names are relative in the code (`joint_states`, `xert/joint_commands`),
so a launch namespace moves them together with `robot_state_publisher`.

### Command stamp

On `/xert/joint_commands`, header.stamp = generation time of the
safety-checked command; capture freshness is enforced upstream by the safety
layer. The sender's `SafetyController` already turns stale camera input into
hold, then return, then rest, and publishes those commands too. A receiver
only checks transport freshness: a command whose stamp is more than 0.2 s
old arrived late or was lost, so it is not acted on as new. The publisher in
`ros/xert_ros_bridge` (`xert_bridge`) follows this rule, and
`tests/test_ros_bridge.py` checks this table and sentence against its code.

### Why JointState for commands

A position-only `JointState` carries the joint names next to the values, so
the receiver never has to trust an implicit order. It is one setpoint per
message, which is what XERT's safety layer takes each tick
(`SafetyController.step` gets one target vector). `trajectory_msgs/JointTrajectory`
would describe a timed trajectory with velocities and timing that nothing in
XERT plans or executes, and the node would still have to reduce it to a
setpoint. It also keeps the only message dependency `sensor_msgs`.

### QoS choices

- **State: best effort, volatile, small depth (`SensorDataQoS`).** A joint
  state is useful only while it is fresh. If one is lost, the next one
  arrives one period later (40 ms at 25 Hz), so retransmitting the old one
  is wasted bandwidth and delays newer data. *Volatile*: a late subscriber
  should get the next live sample, not a stored old pose. *Small depth*: a
  backlog of old states is useless. `robot_state_publisher` (Jazzy)
  subscribes to `joint_states` with `SensorDataQoS` too, so the two match.
  Note: a subscriber that requires *reliable* will not match this
  publisher. Use best effort (or the default "system default"/auto QoS in
  `ros2 topic echo`).
- **Commands: reliable, volatile, depth 1.** A command is the last
  setpoint the sender has safety-checked, and the node keeps acting on it
  until a newer one arrives. If it is lost, the arm keeps the previous target
  instead. That matters most for the final command before the operator stops
  (the pose the arm should settle at), because no later message replaces it.
  A *queue* of old commands is dangerous: after a network hiccup the arm
  would replay stale targets in a burst. Depth 1 keeps only the newest, so
  reliability adds at most a retry of one fresh message, never a backlog.
  *Volatile*: a node that restarts must never receive a command that was
  sent before it was running.
  - Why not best effort, when a lost state is replaced 40 ms later? For a
    steady 25 Hz stream, best effort would also work, and it would be a fair
    choice. Reliable costs latency only on a loss: the repaired sample
    arrives later than the original would have, by at least one round trip
    plus the DDS repair delay (not measured here). That late sample is still
    safe, because the node ages each command by its `header.stamp` and holds
    once it is older than 0.2 s (`xert/safety/watchdog.py`). The stale check
    bounds how old an executed command can be, not the QoS. So reliable
    trades a rare extra delay for not losing sparse or final setpoints.
  - **Matching rule:** a reliable subscription does *not* match a
    best-effort publisher. Every command publisher (for example a
    retargeter node) must use reliable QoS. Otherwise no connection forms
    and every command is dropped, and the only sign is an incompatible-QoS
    warning in the log.

## Build and run (Ubuntu 24.04 + ROS 2 Jazzy)

From the XERT repo root (the package installs meshes from `../../assets`,
so it must stay inside this checkout):

```bash
source /opt/ros/jazzy/setup.bash
cd ros
rosdep install --from-paths . --ignore-src -y
colcon build --packages-select xert_ros
source install/setup.bash
ros2 launch xert_ros xert.launch.py            # add use_rviz:=false for headless
colcon test --packages-select xert_ros && colcon test-result --verbose
```

`colcon` is run from `ros/` because the repo root has its own
`CMakeLists.txt` (the MuJoCo core), and colcon does not look for packages
inside a directory it has already identified as a package.

Until `control_step` is written, `xert_node` publishes nothing and logs
"not implemented" every 5 s. `robot_state_publisher` therefore has no
`/joint_states` to read, and RViz shows the movable links without
transforms.

## TODO(daniel)

A `TODO(daniel):` marks code that is Daniel's to write: the control loop is
CORE logic, not scaffolding. Each stub has its signature and a comment block
with inputs, outputs and invariants in `include/xert_ros/xert_node.hpp`. Each
stub either logs "not implemented" (`control_step`, `on_command`, so the node
stays up for RViz) or throws `std::logic_error` (`read_state`), so nothing
pretends to work. No control math, command forwarding or hardware I/O is in
this package.

The tests that define "done" are in `test/test_xert_node.cpp` as `DISABLED_`
gtests. gtest has no xfail; a `DISABLED_` test is compiled but not run, and
gtest prints "YOU HAVE 2 DISABLED TESTS", which is the closest equivalent.
Enable one by removing the `DISABLED_` prefix once its stub is written.

## Lint

`colcon test` runs `ament_cmake_cppcheck`, `ament_cmake_cpplint`,
`ament_cmake_flake8` (launch file), `ament_cmake_lint_cmake` and
`ament_cmake_xmllint` (package.xml), plus the gtest. `ament_cppcheck` skips
cppcheck 2.x as too slow unless `AMENT_CPPCHECK_ALLOW_SLOW_VERSIONS=1` is
set; CI sets it, so set it locally too for the same result. Not enabled:
`ament_cmake_uncrustify` (an exact-format check; uncrustify with the ROS
config cannot be run on the Windows dev PC, so the code could not be
formatted to it before CI), `ament_cmake_copyright` (the repo has no project
licence yet, so there is no licence header to check for), and
`ament_cmake_pep257` (`ament_flake8` already runs flake8-docstrings on the
one Python file).

## Verification

The dev PC is Windows without ROS, so it cannot build this package.
`.github/workflows/ros.yml` (container `ros:jazzy`) is the verification: it
installs build and test deps with rosdep, then runs `colcon build` and
`colcon test` + `colcon test-result --verbose`. `tests/test_xert_ros_urdf.py`
at the repo root checks, without ROS, that the URDF joint names, kinematic
order, types and limits equal `xert/spec/so101.yaml` and that every mesh
path resolves to a vendored file.
