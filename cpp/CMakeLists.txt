cmake_minimum_required(VERSION 3.0)
project(omega_zero CXX)

find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "-g -O0 -Wall -Wextra")
# set(CMAKE_CXX_FLAGS "-O2 -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories("config" "mcts" "network")  # TODO: target_include_directories

add_subdirectory(config)
add_subdirectory(mcts)
add_subdirectory(network)

add_executable(main main.cpp)
add_executable(play play.cpp)
add_executable(read_mldata read_mldata.cpp)

target_link_libraries(main config mcts network)
target_link_libraries(play config mcts network)
target_link_libraries(read_mldata config mcts network)
