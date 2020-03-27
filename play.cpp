#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <cassert>
#include <unistd.h>

#include "node.hpp"
#include "mcts.hpp"
#include "network.hpp"
#include "misc.hpp"


int main(int argc, char const *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Please specify record file name\n");
        exit(-1);
    }
    const char *file_name = argv[1];
    std::ofstream file(file_name);

    pid_t server_pid = create_server_process();
    (void)server_pid;
    int server_sock = connect_to_server();  // NN server

    std::string input;
    std::cout << "valid actions : position (e.g. a1) / back (b)\n";
    std::cout << "[b]lack / [w]hite ? ";
    std::cin >> input;

    Side player_side;
    Side comp_side;
    if (input == "b") {
        player_side = Side::BLACK;
    } else if (input == "w") {
        player_side = Side::WHITE;
    } else {
        std::cerr << "invalid input " << input << "\n";
        exit(-1);
    }
    comp_side = flip_side(player_side);

    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::vector<Node*> history;

    Board board;
    Side side = Side::BLACK;
    Node* root = new Node(board, Side::BLACK, 0);
    root->expand(server_sock);
    root->backpropagete(root->value(), root);

    Node* current_node = root;
    std::cout << current_node->board() << "\n";
    history.push_back(current_node);

    Action action;
    for (int move_count = 0;; move_count++) {
        if (side == comp_side) {
            current_node = run_mcts(current_node, server_sock, engine);
            history.push_back(current_node);
            action = current_node->parent()->action();
            side = flip_side(side);
        } else {
            while (true) {
                std::cout << "action ? ";
                std::cin >> input;
                action = parse_action(input);
                if (!current_node->board().is_legal_action(action, side)) {
                    std::cout << "invalid action \"" << input << "\"\n";
                } else {
                    break;
                }
            }

            if (action == SpetialAction::BACK) {
                current_node = current_node->parent()->parent();
                // re-create children
                for (auto& child : current_node->children()) {
                    safe_delete(child);
                }
                current_node->children().clear();

                std::vector<float> priors(64);  // re-calculate priors
                float value;  // not used
                request(server_sock, current_node->board(), current_node->side(), current_node->legal_flags(), priors, value);
                current_node->add_children(priors);
            } else {
                unsigned int selected = 64;
                auto& legal_actions = current_node->legal_actions();
                for (unsigned int i = 0; i < legal_actions.size(); i++) {
                    if (legal_actions[i] == action) {
                        selected = i;
                        break;
                    }
                }
                assert(selected >= 0 and selected < legal_actions.size());
                current_node = current_node->children()[selected];
                side = flip_side(side);
            }
        }

        std::cout << side << " : " << action << "\n";
        // std::cout << current_node->board() << "\n";
        std::cout << *current_node << "\n";
        file << action << "\n";
        if (current_node->terminal()) {
            break;
        }
    }

    int count_b = current_node->board().count(CellState::BLACK);
    int count_w = current_node->board().count(CellState::WHITE);
    float result = current_node->board().get_result(player_side);
    std::cout << "black:" << count_b << " white:" << count_w << " result:" << result << "\n";

    file.close();

    return 0;
}