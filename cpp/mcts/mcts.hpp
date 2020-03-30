#pragma once

#include <random>

#include "board.hpp"
#include "node.hpp"


#define N_SIMULATION 100
#define C_PUCT 2.0
#define EXPLORATION_FRAC 0.25
#define DIRICHLET_ALPHA 1.0
// #define EXPLORATION_FRAC 0.0
// #define DIRICHLET_ALPHA 0.0
// #define TAU 1.0
// #define CACHE_SIZE 4000

void play_game(int server_sock, std::vector<GameNode*>& history, std::default_random_engine& engine);
GameNode *run_mcts(GameNode *current_node, int server_sock, std::default_random_engine& engine);
