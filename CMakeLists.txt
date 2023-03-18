cmake_minimum_required(VERSION 3.0.2)
project(raspberry_package)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  cv_bridge

)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES malina_paket
#  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)




include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  /usr/include/pcl-1.10
  SYSTEM ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(pc2img src/convertpc2img.cpp)
target_link_libraries(pc2img ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

