cmake_minimum_required(VERSION 3.8)
project(xert_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

# The node class lives in a static library so the gtest links the same code.
add_library(xert_node_core STATIC src/xert_node.cpp)
target_include_directories(xert_node_core PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(xert_node_core PUBLIC rclcpp::rclcpp ${sensor_msgs_TARGETS})

add_executable(xert_node src/main.cpp)
target_link_libraries(xert_node xert_node_core)

install(TARGETS xert_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_xert_node test/test_xert_node.cpp)
  target_link_libraries(test_xert_node xert_node_core)
endif()

ament_package()
