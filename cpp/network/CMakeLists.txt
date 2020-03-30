cmake_minimum_required(VERSION 3.0)

add_library(network STATIC
    model.cpp
    server.cpp
    "${PROJECT_SOURCE_DIR}/mcts/board.hpp"
)

target_link_libraries(network Threads::Threads ${TORCH_LIBRARIES})
