// Copyright 2026 XERT contributors

#ifndef XERT_ROS__XERT_NODE_HPP_
#define XERT_ROS__XERT_NODE_HPP_

#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace xert_ros
{

// ROS 2 front end for the SO-101 arm. The constructor wires parameters,
// topics and the timer. The three methods marked TODO(daniel) are the control
// loop and are deliberately not implemented yet.
//
// Parameters:
//   joint_names  (string[]) joint order for every message; default is the
//                LeRobot so101_follower order, same as xert/spec/so101.yaml.
//   loop_rate_hz (double)   control loop rate; default 25.0 (spec control_hz).
// Topics: see README.md (joint_states out, xert/joint_commands in).
class XertNode : public rclcpp::Node
{
public:
  explicit XertNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  // joint_states: best-effort, volatile, keep last 5 (rclcpp::SensorDataQoS).
  static rclcpp::QoS state_qos();
  // xert/joint_commands: reliable, volatile, keep last 1.
  static rclcpp::QoS command_qos();

  // TODO(daniel): one control tick, called by the wall timer at loop_rate_hz.
  // Inputs:  the latest command stored by on_command (may be absent or stale),
  //          the arm state from read_state().
  // Outputs: exactly one sensor_msgs/JointState on joint_states per tick.
  // Invariants:
  //   - header.stamp = now(); name = joint_names in declared order;
  //     position.size() == name.size(), in radians.
  //   - a command never reaches the arm without the safety layer: position,
  //     speed and acceleration limits from xert/spec/so101.yaml, as in
  //     xert/safety/limits.py SafetyController.step.
  //   - stale command (older than stale_s = 0.2 s, xert/safety defaults)
  //     -> hold the current pose; still stale after hold_s = 0.5 s -> smooth
  //     return to rest over return_s = 2.0 s (xert/safety/watchdog.py).
  //   - never blocks longer than one period.
  void control_step();

  // TODO(daniel): subscription callback for xert/joint_commands.
  // Input:   a position-only JointState (radians, any joint order, names set).
  // Output:  stores the target plus its receive time for the next
  //          control_step; nothing else.
  // Invariants:
  //   - never forwards the command to the arm or publishes; control_step and
  //     the safety layer own that.
  //   - rejects (logs and drops) a message with an unknown or missing joint
  //     name, name/position size mismatch, or a non-finite position; a
  //     rejected message must not refresh the stale-command clock.
  //   - reorders positions into joint_names order before storing.
  void on_command(sensor_msgs::msg::JointState::ConstSharedPtr msg);

  // TODO(daniel): read the arm's current joint state.
  // Output:  JointState with name = joint_names (declared order) and
  //          position in radians; velocity optional.
  // Invariants:
  //   - a measured value only: from the sim backend, or later the
  //     hardware Present_Position read. Never echo the last command as if
  //     it were measured (see docs/learn/P6.md, CommandedState measured=False).
  //   - throws if no state source is available rather than inventing one.
  sensor_msgs::msg::JointState read_state();

private:
  std::vector<std::string> joint_names_;
  double loop_rate_hz_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr state_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr command_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace xert_ros

#endif  // XERT_ROS__XERT_NODE_HPP_
