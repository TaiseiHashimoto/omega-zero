cmake_minimum_required(VERSION 3.0)

add_library(mcts STATIC
    mcts.cpp
    node.cpp
    board.cpp
    mldata.cpp
    misc.cpp
    "${PROJECT_SOURCE_DIR}/network/server.hpp"
)
