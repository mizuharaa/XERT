// Copyright 2026 XERT contributors

#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "xert_ros/xert_node.hpp"

namespace xert_ros
{

namespace
{
// LeRobot so101_follower motor order, same as xert/spec/so101.yaml joint_names.
const std::vector<std::string> kSo101Joints{
  "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"};
}  // namespace

XertNode::XertNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("xert_node", options),
  joint_names_(declare_parameter("joint_names", kSo101Joints)),
  loop_rate_hz_(declare_parameter("loop_rate_hz", 25.0))
{
  if (joint_names_.empty()) {
    throw std::invalid_argument("joint_names must not be empty");
  }
  if (!(loop_rate_hz_ > 0.0)) {  // also rejects NaN
    throw std::invalid_argument("loop_rate_hz must be positive");
  }

  state_pub_ = create_publisher<sensor_msgs::msg::JointState>("joint_states", state_qos());
  command_sub_ = create_subscription<sensor_msgs::msg::JointState>(
    "xert/joint_commands", command_qos(),
    [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) {on_command(msg);});

  const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(1.0 / loop_rate_hz_));
  timer_ = create_wall_timer(period, [this]() {control_step();});
}

rclcpp::QoS XertNode::state_qos()
{
  return rclcpp::SensorDataQoS();
}

rclcpp::QoS XertNode::command_qos()
{
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
}

void XertNode::control_step()
{
  // TODO(daniel): implement the control tick; contract in xert_node.hpp.
  RCLCPP_WARN_THROTTLE(
    get_logger(), *get_clock(), 5000,
    "control_step not implemented (TODO(daniel)): publishing nothing");
}

void XertNode::on_command(sensor_msgs::msg::JointState::ConstSharedPtr /*msg*/)
{
  // TODO(daniel): validate and store the command; contract in xert_node.hpp.
  RCLCPP_WARN_THROTTLE(
    get_logger(), *get_clock(), 5000,
    "on_command not implemented (TODO(daniel)): command dropped");
}

sensor_msgs::msg::JointState XertNode::read_state()
{
  // TODO(daniel): read measured state; contract in xert_node.hpp.
  throw std::logic_error("read_state not implemented (TODO(daniel))");
}

}  // namespace xert_ros
