#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <unistd.h>

#include "mcts.hpp"
#include "board.hpp"
#include "node.hpp"
#include "network.hpp"
#include "mldata.hpp"
#include "misc.hpp"


void collect_mldata(const char *file_name) {
    int server_sock = connect_to_server();  // NN server

    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::vector<Node*> history;

    for (int i = 0; i < N_GAME; i++) {
        play_game(server_sock, history, engine);

        printf("\n### history ###\n");
        for (unsigned int i = 0; i < history.size(); i++) {
            p("i=", i);
            p(history[i]);
            p("action=", history[i]->action());
        }
        int count_b = history.back()->board().count(CellState::BLACK);
        int count_w = history.back()->board().count(CellState::WHITE);
        float result = history.back()->board().get_result(Side::BLACK);
        p("black:", count_b, " white:", count_w, " result:", result);

        save_game(history, result, file_name);
        safe_delete(history[0]);  // delete root -> whole tree
    }

    close(server_sock);
}

void play_game(int server_sock, std::vector<Node*>& history, std::default_random_engine& engine) {
    Board board;
    Node* root = new Node(board, Side::BLACK, 0);
    root->expand(server_sock);
    root->backpropagete(root->value(), root);

    Node* current_node = root;
    history.push_back(current_node);

    for (int move_count = 0;; move_count++) {
        current_node = run_mcts(current_node, server_sock, engine);
        history.push_back(current_node);
        if (current_node->terminal()) {
            return;
        }
        // printf("\n---\n\n");
    }
}

Node *run_mcts(Node *current_node, int server_sock, std::default_random_engine& engine) {
    current_node->add_exploration_noise(engine);

    for (int sim_count = 0; sim_count < N_SIMULATION; sim_count++) {
        // printf("sim_count=%d\n", sim_count);
        Node* node = current_node;
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

    Node* next_node = current_node->next_node(engine);
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
