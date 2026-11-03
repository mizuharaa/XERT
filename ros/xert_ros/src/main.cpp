// Copyright 2026 XERT contributors

#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "xert_ros/xert_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<xert_ros::XertNode>());
  rclcpp::shutdown();
  return 0;
}
