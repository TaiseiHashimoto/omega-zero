#include <iostream>
#include <vector>
#include <cassert>
#include <random>

#include "mcts.hpp"
#include "board.hpp"
#include "node.hpp"
#include "mldata.hpp"
#include "misc.hpp"


void play_game(int n_simulation, int server_sock, std::vector<GameNode*>& history, std::default_random_engine& engine) {
    Board board;
    GameNode *root = new GameNode(board, Side::BLACK, 0);
    root->expand(server_sock);
    root->backpropagete(root->value(), root);

    GameNode *current_node = root;
    history.push_back(current_node);

    for (int move_count = 0;; move_count++) {
        // printf("move_count = %d\n", move_count+1);
        bool stochastic = move_count < N_SAMPLING_MOVES;
        current_node = run_mcts(current_node, n_simulation, server_sock, engine, stochastic);
        // p(current_node);
        history.push_back(current_node);  // terminal node included
        if (current_node->terminal()) {
            return;
        }
    }
}

GameNode *run_mcts(GameNode *current_node, int n_simulation, int server_sock, std::default_random_engine& engine, bool stochastic) {
    current_node->add_exploration_noise(engine);

    for (int sim_count = 0; sim_count < n_simulation; sim_count++) {
        GameNode* node = current_node;
        // printf("sim_count=%d\n", sim_count);
        while (node->expanded()) {  // terminal => not expanded
            // p("forward");
            // p(node);
            node = node->select_child();
        }
        // before expand => terminal=false even if terminal in fact
        if (!node->terminal()) {
            node->expand(server_sock);
        }
        // p("leaf");
        // p(node);
        node->backpropagete(node->value(), current_node);
        // p();
    }

    GameNode* next_node = current_node->next_node(engine, stochastic);
    // printf("selected ( ");
    // for (unsigned int i = 0; i < current_node->legal_actions().size(); i++) {
    //     auto action = current_node->legal_actions()[i];
        // std::cout << action << ":" << current_node->posteriors()[action] << " ";
    // }
    // p(")");
    // p(next_node);

    for (auto& child : current_node->children()) {  // delete unnecessary data
        if (child != next_node) {
            safe_delete(child);
        }
    }

    return next_node;
}
