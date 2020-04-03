#pragma once

#include <random>

#include "board.hpp"
#include "node.hpp"


void play_game(std::vector<GameNode*>& history, int server_sock, std::default_random_engine& engine);
GameNode *run_mcts(GameNode *current_node, float tau, int server_sock, std::default_random_engine& engine);
