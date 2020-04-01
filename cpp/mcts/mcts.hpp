#pragma once

#include <random>

#include "board.hpp"
#include "node.hpp"


#define C_PUCT 1.4
#define EXPLORATION_FRAC 0.25
#define DIRICHLET_ALPHA 1.0
// #define EXPLORATION_FRAC 0.0
// #define DIRICHLET_ALPHA 0.0
// #define TAU 1.0
// #define CACHE_SIZE 10000  // ~2.5 MB

void play_game(int n_simulation, int server_sock, std::vector<GameNode*>& history, std::default_random_engine& engine);
GameNode *run_mcts(GameNode *current_node, int n_simulation, int server_sock, std::default_random_engine& engine);
