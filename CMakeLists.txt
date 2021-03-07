cmake_minimum_required(VERSION 3.13.4)
project(Drone-Navigation)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(droverCode ShortestPath.cpp)
target_link_libraries(droverCode ${OpenCV_LIBS})
