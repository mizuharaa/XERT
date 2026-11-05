# Slice I: ROS 2 bridge (`xert_ros_bridge`)

Branch `feat/i-ros-bridge`, from `integration/sim-mvp` @ 9115f43. Not
pushed; integration pushes it and iterates on CI. **No ROS on the dev PC:
the two rclpy nodes, the launch file, the launch test and the new `ros.yml`
steps have never been executed.** Only the ROS-free core and its tests ran
here.

## What changed and why

- `ros/xert_ros_bridge/` is a new `ament_python` package. It makes the
  slice E scaffold do something end to end without hardware and without
  Daniel's control loop. It adds no control math.
  - `xert_bridge` (`xert_ros_bridge/bridge_node.py`) runs XERT's existing
    `run_loop` (`xert/runtime/loop.py`) with a motion source
    (`--source clip <file>`, or `--source webcam` with a calibration), the
    `CalibratedMapper` and the `SafetyController`. It publishes every
    applied command from `on_step` as a position-only `JointState` (names
    in spec order) on `/xert/joint_commands`, reliable / volatile / keep
    last 1. Hold, return, rest and e-stop phases are published too, so the
    topic always carries what the safety layer decided. The node only
    publishes; its executor spins on a daemon thread, so ROS never delays
    `run_loop`'s wall-clock deadlines. It waits for `--wait-subscribers`
    matched subscribers (default 1, the follower) before the loop starts.
    SIGINT/SIGTERM become `run_loop`'s e-stop: one final e-stop command
    goes out, then the node waits up to 1 s for its reliable ack and exits.
    A clip calibrates itself from its first frame showing every mapped
    hand (as P3.6 and `hw-dry-run`); its run lasts the clip plus 3.7 s
    (stale 0.2 + hold 0.5 + return 2.0 + 1.0 to settle), so the safety
    layer walks the arm back to rest before the node exits.
  - `xert_sim_follower` (`xert_ros_bridge/follower_node.py`) subscribes
    `/xert/joint_commands`, drives a continuous `MuJoCoBackend` on the
    spec (no task, no episode, no horizon) at the spec rate (25 Hz), and
    publishes the measured sim state on `/joint_states` with
    `SensorDataQoS` (best effort / volatile / keep last 5), which
    `robot_state_publisher` reads.
  - `xert_ros_bridge/core.py` holds all logic with no rclpy import:
    the topic contract tuples, `pack` (the stamp rule), `open_pipeline`
    (source, backend, mapper, duration for `run_loop`) and `Follower`
    (validation and the tick). `ros_io.py` is the rclpy glue (QoS from
    the tuples, `JointState` packing, stamp conversion).
  - `launch/bridge.launch.py` = `xert_bridge` (args `source` (default
    `clip`), `clip`, `spec`, `profile`, `calibration`, `wait_subscribers`)
    + `xert_sim_follower` + `robot_state_publisher` with the slice E URDF
    + RViz (`use_rviz`, default true). `xert_node` is not launched.
- `ros/xert_ros/include/xert_ros/xert_node.hpp` and
  `ros/xert_ros/README.md`: comments and docs only. The stamp changed from
  "capture time" to "generation time of the safety-checked command;
  capture freshness is enforced upstream by the safety layer" (README:
  new "Command stamp" subsection).
- `requirements-bridge-linux-lock.txt`: the minimal Jazzy environment.
- `.github/workflows/ros.yml`: XERT on the Jazzy system Python, then build
  and test both packages, then the bridge unit tests with rclpy present.

## The freshness contract (two clocks)

- **Motion freshness, upstream.** The bridge's `SafetyController` and
  `Watchdog` already turn stale camera input into hold, then return, then
  rest (`xert/safety/watchdog.py`, `limits.py:33`). The bridge publishes
  whatever phase it is in.
- **Transport freshness, at the follower.** `header.stamp` =
  `record["t_command"]`, the `perf_counter` instant `run_loop` produced the
  command, before `send_targets` (`loop.py:48-51`). It is moved to the
  ROS clock by one offset, `ROS now - perf_counter`, measured once at node
  start and logged. It is not the publish time, so a backend stall between
  generation and `on_step` makes the command look old, not fresh. A record
  with `deadline_overrun` is never published (`run_loop` aborts after it).
- **Follower rules** (`core.Follower`):
  - Accept a command only if its names cover the spec joints exactly once,
    `name`/`position` lengths match, all positions are finite, and its
    stamp is finite, not earlier than the last accepted stamp, and not
    more than **0.02 s** ahead of the follower's clock (`FUTURE_S`: half a
    25 Hz period, for the one-time offset measurement on one host; on two
    hosts, sync clocks first). Positions are reordered by name. A
    rejected message changes neither the accepted target nor its stamp.
  - Each tick, the effective target is the accepted target while
    `now - stamp <= 0.2 s`, else the target the follower applied last tick
    (transport hold).
  - That target goes through the follower's own `SafetyController`, used
    only as the spec's position clip and velocity/acceleration limiter.
    It is fed `t_capture = watchdog_now`, so its watchdog never holds or
    returns to rest. A joint that is moving when transport is lost brakes
    at the acceleration limit over a few ticks, then the applied target
    stays constant indefinitely. Return to rest only ever arrives as
    commands from the bridge.
- **The follower is not a controller.** It adds no control law. It is a
  visualization stand-in that keeps the spec limits on whatever the
  bridge's safety layer already decided. Applied targets (what it sends to
  MuJoCo) and measured positions (what MuJoCo reports, on `/joint_states`)
  are kept apart in the tests.

## Lock resolution

`printf 'numpy==1.26.4\nmujoco==3.14.0\npybind11==3.1.0\n' | uv pip compile -
--python-version 3.12 --python-platform x86_64-unknown-linux-gnu
--only-binary :all: --no-annotate --no-header` resolved 11 packages
(uv 0.12.13), so every pin, mujoco 3.14.0 included, has a cp312 manylinux
binary wheel. pybind11 is 3.1.0, as `scripts/build_core.ps1` installs.
numpy is pinned to 1.26.4, Ubuntu 24.04's `python3-numpy`, which Jazzy's
apt packages are built against; pip installs with `--ignore-installed` so
it never tries to remove apt-owned copies. XERT on this path needs only
numpy and mujoco: the spec file is JSON, and `xle_workbench/__init__.py`
imports nothing.

## How to verify

Windows (done, this branch):

```
.\.venv\Scripts\python.exe -m pytest -q
```

On Ubuntu 24.04 + ROS 2 Jazzy, from the repo root (what `ros.yml` runs):

```bash
source /opt/ros/jazzy/setup.bash
export PIP_BREAK_SYSTEM_PACKAGES=1
rosdep install --from-paths ros --ignore-src -y --rosdistro jazzy \
  -t buildtool -t build -t build_export -t test -t exec --skip-keys rviz2
sudo apt-get install -y python3-pip python3-dev
python3 -m pip install --no-deps --ignore-installed -r requirements-bridge-linux-lock.txt
python3 -c "import numpy, mujoco, pybind11"
cmake -S . -B build/ros -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=/usr/bin/python3
cmake --build build/ros --target _reach_core --parallel 2
python3 -m pip install --no-deps -e .
python3 -c "from xert.retarget.calibrate import CalibratedMapper"
cd ros && colcon build --packages-select xert_ros xert_ros_bridge
colcon test --packages-select xert_ros xert_ros_bridge && colcon test-result --verbose
cd .. && source ros/install/setup.bash && python3 -m pytest -q tests/test_ros_bridge.py
# Watch it: any landmark clip (record-motion, or one scripted like tests/test_motion_file.py)
ros2 launch xert_ros_bridge bridge.launch.py clip:=/path/to/clip.json
```

## Tests

- `tests/test_ros_bridge.py` (base venv; scripted landmarks, simulated
  clocks):
  - first read returns None, and an empty-camera interval longer than
    0.2 s: both publish hold commands, names in spec order, positions =
    the command, stamp = offset + generation time;
  - the validation table: reordered names accepted and reordered; NaN,
    missing, duplicate and unknown joint, length mismatch, backward stamp,
    stamp past the 0.02 s tolerance and a NaN stamp rejected without
    changing the target or its stamp; the 0.2 s freshness edge;
  - coalesced delivery (every other command): applied per-tick velocity
    and acceleration stay inside the spec limits;
  - transport loss for 2.8 s while panning at the 1 rad/s velocity limit:
    inside the 0.2 s fresh window the joint brakes toward the last command,
    so it is still moving at 0.4 rad/s on the first stale tick, where it
    advances exactly one acceleration step less, `(v - a*dt)*dt` = 0.0112
    rad (a follower that stopped dead would fail this); braking
    displacement is bounded; the applied target is then constant with no
    return to rest; tracking resumes, still inside the limits;
  - a 300 ms sleep in `send_targets`: the stamp is the generation time,
    the record is an overrun and is not published, and the follower treats
    the command as stale (no movement);
  - the clip through `open_pipeline`, `run_loop`, `pack` and the follower
    in lockstep: phases hold, live, hold, return, rest; the command stream
    holds away from rest, closes in on rest monotonically and ends at rest;
    the follower's measured pose reaches rest within 0.01 rad; after the
    bridge stops, more than 600 ticks of indefinite hold. The e-stop
    variant (stop during the return) holds mid-return instead;
  - the contract check: the README topic table (names, types, QoS) and the
    stamp sentence equal `core.COMMANDS`, `core.STATES` and
    `core.STAMP_RULE`;
  - with rclpy (skips here, runs in `ros.yml`): QoS profiles, `JointState`
    packing and stamp round trip, both node modules import.
- `ros/xert_ros_bridge/test/test_bridge_launch.py` (`launch_testing`,
  `colcon test`): launches `bridge.launch.py` itself (imports its
  description and includes it with `use_rviz:=false`,
  `wait_subscribers:=2`) with the same scripted clip. It checks the exact
  topic set and the publisher/subscriber endpoints with their reliability
  and durability (slice E's introspection approach); about 25 Hz commands
  (median stamp spacing 40 ms +/- 5 ms), monotonic stamps, names in spec
  order, at most 2 commands missing; the hold, return, rest stream; the
  follower inside the position limits and reaching rest; NaN, missing
  joint and future-stamped commands not moving the arm, then a
  reordered-name valid stream moving it; more than 600 `/joint_states`
  messages with the pose held; exit code 0 for both nodes (the bridge's
  includes "no deadline abort"). Estimated about 30 s; not run yet.

## What only CI can verify (nothing below has run)

- That `colcon build` installs the package, its console scripts and its
  launch file, and that colcon picks its pytest step.
- Both nodes under rclpy: the signal-to-e-stop path, `wait_for_all_acked`,
  the executor thread, QoS matching and rejection logging.
- The launch test as a whole, its timing in the container, and the exact
  topic set (`/rosout`, `/parameter_events` and the
  `robot_state_publisher` topics are expected; anything else fails it).
- The Jazzy Python environment: pip in the `ros:jazzy` image with
  `--ignore-installed`, the `_reach_core` build against `/usr/bin/python3`,
  the editable install, and that numpy 1.26.4 + mujoco 3.14.0 run XERT.
- `--source webcam` is not run anywhere, CI included.
- RViz is never started (rviz2 is skipped by rosdep).

## Swap in Daniel's node

Once `control_step`, `on_command` and `read_state` exist, replace lines
33-34 of `ros/xert_ros_bridge/launch/bridge.launch.py` with:

```python
        Node(package='xert_ros', executable='xert_node', output='screen'),
```

Same topics and QoS, so `xert_bridge` and `robot_state_publisher` need no
change. Note the follower holds on transport loss and never returns to
rest by itself, while the `control_step` stub comment still asks for a
return to rest 0.7 s after the last fresh command. Which one Daniel's node
does is Daniel's decision (see the open question below).

## Interview questions

1. *Why stamp commands with generation time, not capture or publish
   time?* Capture freshness is already enforced by the safety layer, which
   publishes hold and return-to-rest commands itself, and publish time
   would make a command delayed by a backend stall look fresh, so the
   stamp records when the safety-checked command was made and the receiver
   only judges the transport.
2. *Why does the follower run its own SafetyController?* Commands can
   arrive coalesced, late or not at all, and its own limiter keeps every
   step it sends to the sim inside the spec's velocity and acceleration
   limits; it gets `t_capture = watchdog_now` so it limits without ever
   deciding to hold or return to rest on its own.
3. *Why does the bridge publish from `on_step` and spin on another
   thread?* `run_loop` aborts on a missed deadline, so ROS work must never
   run on its thread; the node only publishes, and its executor spins on a
   daemon thread for discovery and parameter services.

## Open questions

- Daniel's `control_step` contract still says "return to rest 0.7 s after
  the last fresh command" on transport loss. The bridge's safety layer
  already returns to rest on camera loss, and this follower only holds.
  Should Daniel's node hold on transport loss too?
- The 0.02 s future-stamp tolerance assumes one host. Two hosts need clock
  sync (chrony) and probably a larger tolerance.
- The clock offset is measured once per run. On Linux, NTP slewing moves
  `CLOCK_MONOTONIC` (perf_counter) and `CLOCK_REALTIME` (ROS time) at the
  same rate, so the offset does not drift; only a realtime step changes
  it. Both directions fail safe: a forward step makes commands look old
  (the follower holds), a backward step of more than 0.02 s makes them
  look future (rejected, with warnings). No mitigation on the clip path.
- No ament linters run on the Python package (no flake8/pep257 tests): the
  ROS flake8 config could not be run here before CI. Add
  `test/test_flake8.py` once a CI run shows the style it reports.

## For integration (not edited here)

- `README.md` ROS section: one line for `ros2 launch xert_ros_bridge
  bridge.launch.py clip:=<clip>` and the sim follower.
- `docs/handoff/TASKS.md`: slice I row (bridge + follower + launch test;
  CI pending at the pushed SHA).
- `docs/handoff/E-xert-ros.md` "Still unverified": `ros.yml` now installs
  exec deps (rviz2 skipped) and starts `robot_state_publisher` in the
  bridge launch test.
- `MVP_PROGRESS`: none (no measured numbers; ledger untouched).
