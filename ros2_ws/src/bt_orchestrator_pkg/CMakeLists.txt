cmake_minimum_required(VERSION 3.8)
project(bt_orchestrator_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Required packages ─────────────────────────────────────────────────────────
find_package(ament_cmake        REQUIRED)
find_package(rclcpp             REQUIRED)
find_package(behaviortree_cpp_v3 REQUIRED)
find_package(tf2           REQUIRED)
find_package(tf2_ros       REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# ROS2 message packages
find_package(sensor_msgs        REQUIRED)
find_package(nav_msgs           REQUIRED)
find_package(geometry_msgs      REQUIRED)
find_package(std_msgs           REQUIRED)

# For resolving the installed XML path at runtime
find_package(ament_index_cpp    REQUIRED)

# ── Expose include directory (for indoor_detector.hpp) ───────────────────────
include_directories(include)

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(bt_brain src/main.cpp)

ament_target_dependencies(bt_brain
  rclcpp
  behaviortree_cpp_v3
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  ament_index_cpp
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Explicit linkage of tf2 imported targets.
# ament_target_dependencies sets include dirs and populates ament's variable
# lists, but in Humble the actual .so linkage for tf2/tf2_ros requires the
# CMake imported targets directly — otherwise the linker throws
# "undefined reference to tf2_ros::TransformBroadcaster / Buffer" etc.
target_link_libraries(bt_brain
  tf2::tf2
  tf2_ros::tf2_ros
)

# ── Install ───────────────────────────────────────────────────────────────────
# Install the header so other packages can reuse IndoorDetector
install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS
  bt_brain
  DESTINATION lib/${PROJECT_NAME}
)

# Install the BT XML so it's found via ament_index at runtime
install(DIRECTORY bt_xml/
  DESTINATION share/${PROJECT_NAME}/bt_xml
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install RViz config (if present)
install(DIRECTORY rviz/
  OPTIONAL
  DESTINATION share/${PROJECT_NAME}/rviz
)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()