# Slice E: ROS 2 Jazzy `xert_ros` scaffold

Branch `feat/e-xert-ros`, from `integration/sim-mvp` @ 5cd6853. Not pushed.

## What changed and why

- `ros/xert_ros/` is a new `ament_cmake` package (C++17). It is
  scaffolding only. It has no control math, no command forwarding and no
  hardware I/O.
  - `XertNode` (`include/xert_ros/xert_node.hpp`, `src/xert_node.cpp`,
    `src/main.cpp`) declares the parameters `joint_names` (LeRobot
    so101_follower order) and `loop_rate_hz` (25.0, the spec's
    `control_hz`). It creates a `/joint_states` publisher
    (`SensorDataQoS`), a `/xert/joint_commands` subscriber (position-only
    `JointState`, reliable, depth 1, volatile) and a wall timer. The control
    loop, the command callback and state reading are `TODO(daniel)` stubs
    with written contracts.
  - `urdf/so101.urdf` is the official TheRobotStudio SO-ARM100
    `Simulation/SO101/so101_new_calib.urdf`, pinned at commit
    `a758567c3978dfeefe282ede0500085a48fe8f78`, licence Apache-2.0 (from
    `gh api repos/TheRobotStudio/SO-ARM100`; `urdf/LICENSE` is the upstream
    file). The edits are mesh paths (`package://xert_ros/meshes/`) and the
    joint position/velocity limits, set to `xert/spec/so101.yaml` (largest
    position change 3e-6 rad). `urdf/SOURCE.json` records the hashes and the
    edits.
  - Meshes are not copied. CMake installs the vendored
    `assets/menagerie/robotstudio_so101/assets/*.stl` (Apache-2.0, LICENSE
    installed beside them) as `share/xert_ros/meshes`. The 13 STLs the URDF
    uses have the same git blob hashes as upstream SO-ARM100 at the pinned
    commit.
  - `launch/xert.launch.py` starts robot_state_publisher, xert_node, and
    rviz2 (`use_rviz`, default true). `rviz/xert.rviz` shows Grid,
    RobotModel and TF, with fixed frame `base_link`.
  - `README.md` covers the topics/QoS table with reasons, the command
    message choice, build/run, the TODO(daniel) convention and the linters.
- `tests/test_xert_ros_urdf.py` (base env, no ROS) checks that the URDF
  movable joints, in kinematic order from `base_link`, have the names, types
  (revolute) and position/velocity limits of `so101.yaml`. It also checks
  that every mesh is `package://xert_ros/meshes/...` and resolves to a
  vendored file, and that package.xml, the LICENSE and SOURCE.json parse.
- `.github/workflows/ros.yml` runs in the `ros:jazzy` container. It runs
  rosdep (build/test deps only, which skips rviz2's Qt/Ogre), then
  `colcon build` and `colcon test` + `colcon test-result --verbose` from
  `ros/`. The repo root has its own CMakeLists.txt, and colcon does not look
  inside a package, so colcon must run from `ros/`.

## How to verify

- Windows (done): `python -m pytest -q` gave 154 passed, 25 skipped,
  1 xfailed, 8 subtests (the 151 baseline plus 3 new). There is no `build/`
  in this worktree, so ctest was not run. `package.xml` and the URDF parse
  with `xml.etree`. The RViz YAML was not parsed because no Python here has
  PyYAML.
- Ubuntu 24.04 + Jazzy (CI or by hand): see `ros/xert_ros/README.md`. The
  expected result is the gtest (5 tests run, 2 disabled) and the linters
  passing. **Nothing ROS-side has been compiled or run yet**; the first CI
  run is the first build.

## Interview questions

1. *Why is `/joint_states` best effort but `/xert/joint_commands` reliable
   with depth 1?* A lost state is replaced by a fresher one a period later,
   so retrying it is wasted work; a lost command matters, but a queue of old
   ones would replay stale targets after a hiccup, so the topic keeps exactly
   the newest one and delivers it reliably.
2. *Why a position-only `JointState` for commands and not
   `JointTrajectory`?* XERT's safety layer takes one named target per tick,
   and a `JointState` carries the names with the values, while a trajectory
   adds timing that nothing in XERT plans or executes.
3. *How do you know the URDF matches the robot the rest of XERT uses?* A
   ROS-free pytest walks the URDF from `base_link` and asserts the joint
   names, order, types and limits equal `xert/spec/so101.yaml`, and the
   meshes are the same files as the MuJoCo model, checked by git blob hash
   against the pinned upstream commit.

## Every TODO(daniel)

| Where | What | Done means |
|---|---|---|
| `ros/xert_ros/include/xert_ros/xert_node.hpp:34` (body `src/xert_node.cpp:54`) | `control_step()` | One `JointState` per tick on `/joint_states`: stamp = now, names in declared order, sizes match, radians. Commands go through the safety limits (`xert/safety`). A stale command (> 0.2 s) holds, then after 0.5 s returns smoothly to rest over 2 s. Never blocks past one period. `DISABLED_ControlLoopPublishesOneStatePerPeriod` passes. |
| `ros/xert_ros/include/xert_ros/xert_node.hpp:50` (body `src/xert_node.cpp:62`) | `on_command()` | Validates names, sizes and finiteness, reorders into `joint_names` order, and stores the target with its receive time. Never forwards or publishes. A rejected message does not refresh the stale clock. `DISABLED_StaleCommandHoldsPose` passes. |
| `ros/xert_ros/include/xert_ros/xert_node.hpp:63` (body `src/xert_node.cpp:70`) | `read_state()` | Returns measured state only (sim backend, later `Present_Position`), named in declared order. It never echoes the last command and throws when there is no source. `ReadStateIsNotImplementedYet` is then replaced by a real test. |
| `ros/xert_ros/test/test_xert_node.cpp:96` | `DISABLED_ControlLoopPublishesOneStatePerPeriod` | Write the body as its comment says: at 50 Hz, 50 +/- 2 messages in 1 s, names and sizes as above, stamps about 20 ms apart. Then drop the `DISABLED_` prefix. |
| `ros/xert_ros/test/test_xert_node.cpp:106` | `DISABLED_StaleCommandHoldsPose` | Write the body as its comment says: hold after 0.2 s of silence; an unknown-name or NaN command is dropped without refreshing the clock. Then drop the prefix. |

Owner decision (not code): `package.xml:8` licence is "TODO: no project
licence chosen yet". The repo has no LICENSE. `ament_copyright` stays off
until there is one.

## Linters

Kept: cppcheck, cpplint, flake8 (launch file), lint_cmake, xmllint
(package.xml). Dropped: uncrustify (an exact-format check that could not be
run here before CI), copyright (no project licence) and pep257
(`ament_flake8` already runs flake8-docstrings). `ament_cppcheck` on
Jazzy skips cppcheck 2.x as too slow (checked in its `main.py`) unless
`AMENT_CPPCHECK_ALLOW_SLOW_VERSIONS` is set, so `ros.yml` sets it. If
cppcheck turns out noisy or slow in CI, removing that variable turns it back
into a skip.

## Unverified until CI runs

- That the C++ compiles against Jazzy rclcpp. It was reviewed statically
  only; there is no ROS on this PC.
- That cpplint, cppcheck, flake8, lint_cmake and xmllint pass. They were
  written to their rules (100-col C++, 99-col Python, single quotes, header
  guard `XERT_ROS__XERT_NODE_HPP_`, copyright line, `TODO(name):` format)
  but never run.
- The graph-based gtest (`AdvertisesStatePublisherAndCommandSubscriber`),
  which polls up to 2 s for the node's own endpoints.
- That rosdep resolves every key with `-t buildtool -t build -t build_export
  -t test`, and the CI time.
- The launch file and RViz config. CI never starts them (exec deps are not
  installed). Run `ros2 launch xert_ros xert.launch.py` on a Jazzy machine.
