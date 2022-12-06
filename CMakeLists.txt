cmake_minimum_required(VERSION 3.0.2)
project(latte)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  visualization_msgs
  lang_corr_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES latte
#  CATKIN_DEPENDS geometry_msgs roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
