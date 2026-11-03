// Copyright 2026 XERT contributors

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "xert_ros/xert_node.hpp"

class XertNodeTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}
};

namespace
{
// The graph can lag endpoint creation by a moment, so poll for up to 2 s.
std::vector<rclcpp::TopicEndpointInfo> endpoints(
  rclcpp::Node & node, const std::string & topic, bool publishers)
{
  for (int i = 0; i < 100; ++i) {
    auto info = publishers ?
      node.get_publishers_info_by_topic(topic) :
      node.get_subscriptions_info_by_topic(topic);
    if (!info.empty()) {
      return info;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return {};
}
}  // namespace

TEST_F(XertNodeTest, DeclaresLeRobotJointOrderAndLoopRate)
{
  auto node = std::make_shared<xert_ros::XertNode>();
  const std::vector<std::string> expected{
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"};
  EXPECT_EQ(node->get_parameter("joint_names").as_string_array(), expected);
  EXPECT_DOUBLE_EQ(node->get_parameter("loop_rate_hz").as_double(), 25.0);
}

TEST_F(XertNodeTest, RejectsNonPositiveLoopRate)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides({rclcpp::Parameter("loop_rate_hz", 0.0)});
  EXPECT_THROW(xert_ros::XertNode node(options), std::invalid_argument);
}

TEST_F(XertNodeTest, QosProfiles)
{
  const auto state = xert_ros::XertNode::state_qos();
  EXPECT_EQ(state.reliability(), rclcpp::ReliabilityPolicy::BestEffort);
  EXPECT_EQ(state.durability(), rclcpp::DurabilityPolicy::Volatile);
  EXPECT_EQ(state.history(), rclcpp::HistoryPolicy::KeepLast);
  EXPECT_EQ(state.depth(), 5u);

  const auto command = xert_ros::XertNode::command_qos();
  EXPECT_EQ(command.reliability(), rclcpp::ReliabilityPolicy::Reliable);
  EXPECT_EQ(command.durability(), rclcpp::DurabilityPolicy::Volatile);
  EXPECT_EQ(command.history(), rclcpp::HistoryPolicy::KeepLast);
  EXPECT_EQ(command.depth(), 1u);
}

TEST_F(XertNodeTest, AdvertisesStatePublisherAndCommandSubscriber)
{
  auto node = std::make_shared<xert_ros::XertNode>();

  const auto pubs = endpoints(*node, "/joint_states", true);
  ASSERT_EQ(pubs.size(), 1u);
  EXPECT_EQ(pubs[0].topic_type(), "sensor_msgs/msg/JointState");
  EXPECT_EQ(pubs[0].qos_profile().reliability(), rclcpp::ReliabilityPolicy::BestEffort);

  const auto subs = endpoints(*node, "/xert/joint_commands", false);
  ASSERT_EQ(subs.size(), 1u);
  EXPECT_EQ(subs[0].topic_type(), "sensor_msgs/msg/JointState");
  EXPECT_EQ(subs[0].qos_profile().reliability(), rclcpp::ReliabilityPolicy::Reliable);
}

TEST_F(XertNodeTest, ReadStateIsNotImplementedYet)
{
  auto node = std::make_shared<xert_ros::XertNode>();
  EXPECT_THROW(node->read_state(), std::logic_error);
}

// gtest has no xfail. A DISABLED_ test is compiled but not run, and gtest
// prints "YOU HAVE n DISABLED TESTS", so the missing work stays visible.

// TODO(daniel): enable once control_step and read_state are implemented.
// Done means: with loop_rate_hz = 50 and a test state source, spinning the
// node for 1 s yields 50 +/- 2 JointState messages on /joint_states, each with
// name == joint_names in declared order, position.size() == name.size(), and
// header.stamp increasing by about 20 ms per message.
TEST_F(XertNodeTest, DISABLED_ControlLoopPublishesOneStatePerPeriod)
{
  FAIL() << "TODO(daniel): write this test with control_step";
}

// TODO(daniel): enable once on_command and control_step are implemented.
// Done means: after one valid command and then silence longer than
// stale_s = 0.2 s, the target sent to the arm stops following the command and
// holds the current pose; a command with an unknown joint name or a NaN is
// dropped and does not refresh the stale clock.
TEST_F(XertNodeTest, DISABLED_StaleCommandHoldsPose)
{
  FAIL() << "TODO(daniel): write this test with on_command";
}
