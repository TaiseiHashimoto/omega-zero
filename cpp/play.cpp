#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <unistd.h>
#include <getopt.h>

#include "node.hpp"
#include "mcts.hpp"
#include "server.hpp"
#include "misc.hpp"
#include "config.hpp"


int main(int argc, char *argv[]) {
    if ((argc < 2) || (argc > 2 && argv[2][0] != '-')) {
        fprintf(stderr, "Usage: play exp_id [--generation=G] [--n_simulation=N] [--device_id=ID] [--record_fname=NAME]\n");
        exit(-1);
    }
    int exp_id = atoi(argv[1]);
    std::cout << "exp_id = " << exp_id << std::endl;

    char exp_path[100];
    get_exp_path(argv[0], exp_id, exp_path);
    std::cout << "exp_path = " << exp_path << std::endl;

    int generation = -1;  // if -1 select best model
    int n_simulation = 400;
    char record_fname[100] = "./record.txt";
    int device_id = 0;

    int opt, longindex;
    const struct option longopts[] = {
        {"generation", required_argument, NULL, 'g'},
        {"n_simulation", required_argument, NULL, 'n'},
        {"device_id", required_argument, NULL, 'd'},
        {"record_fname", required_argument, NULL, 'r'},
        {0, 0, 0, 0}
    };
    while ((opt = getopt_long(argc, argv, "g:n:d:r:", longopts, &longindex)) != -1) {
        switch (opt) {
            case 'g':
                generation = atoi(optarg);
                break;
            case 'n':
                n_simulation = atoi(optarg);
                break;
            case 'd':
                device_id = atoi(optarg);
                break;
            case 'r':
                strcpy(record_fname, optarg);
                break;
            default:
                fprintf(stderr, "unknown option\n");
                exit(-1);
        }
    }
    std::cout << "generation = " << generation << std::endl;
    std::cout << "n_simulation = " << n_simulation << std::endl;
    std::cout << "device_id = " << device_id << std::endl;
    std::cout << "record_fname = " << record_fname << std::endl;

    init_config(exp_path, generation, device_id);
    // overwrite experiment configuration
    set_config(/*n_thread=*/1, n_simulation, /*e_frac=*/0.0);
    const auto& config = get_config();

    std::ofstream file(record_fname);

    pid_t server_pid = create_server_process();
    (void)server_pid;
    int server_sock = connect_to_server();  // NN server

    std::string input;
    std::cout << "valid actions : position (e.g. a1) / back / pass" << std::endl;
    std::cout << "@ [b]lack / [w]hite ?" << std::endl;
    std::cin >> input;

    Side player_side;
    Side comp_side;
    if (input == "b") {
        player_side = Side::BLACK;
    } else if (input == "w") {
        player_side = Side::WHITE;
    } else {
        std::cerr << "invalid input " << input << std::endl;
        exit(-1);
    }
    comp_side = flip_side(player_side);

    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::vector<GameNode*> history;

    Board board;
    Side side = Side::BLACK;
    GameNode* root = new GameNode(board, Side::BLACK, 0);
    root->expand(server_sock);
    root->backpropagete(root->value(), root);

    GameNode *current_node = root;
    std::cout << "\n" << current_node->board() << std::endl;
    history.push_back(current_node);

    Action action;
    for (int move_count = 0;; move_count++) {
        std::cout << "side : " << side << std::endl;

        if (side == comp_side) {
            float tau = (move_count < config.e_step) ? config.tau : 0.0;
            current_node = run_mcts(current_node, tau, server_sock, engine);
            history.push_back(current_node);
            action = current_node->parent()->action();
            std::cout << "@ action : " << action << "\n";
        } else {
            while (true) {
                std::cout << "@ action ?\n";
                std::cin >> input;
                action = parse_action(input);
                if (!current_node->board().is_legal_action(action, side)) {
                    std::cout << "invalid action \"" << input << "\"\n";
                } else {
                    break;
                }
            }

            if (action == SpetialAction::PASS) {
                current_node = current_node->children()[0];
            } else if (action == SpetialAction::BACK) {
                current_node = current_node->parent()->parent();
                // re-create children
                for (auto& child : current_node->children_()) {
                    safe_delete(child);
                }
                current_node->children_().clear();

                // std::vector<float> priors(64);  // re-calculate priors
                // float value;  // not used
                // request(server_sock, current_node->board(), current_node->side(), current_node->legal_flags(), priors, value);
                // current_node->add_children(priors);
                current_node->expand(server_sock);
                // do not flip side in this case
                side = flip_side(side);
            } else {
                unsigned int selected = 64;
                auto& legal_actions = current_node->legal_actions();
                for (unsigned int i = 0; i < legal_actions.size(); i++) {
                    if (legal_actions[i] == action) {
                        selected = i;
                        break;
                    }
                }
                assert(selected < legal_actions.size());
                current_node = current_node->children()[selected];
            }

            if (!current_node->expanded()) {
                current_node->expand(server_sock);
                current_node->backpropagete(current_node->value(), current_node);
            }
        }

        std::cout << "\n" << *current_node << "\n";
        file << action << "\n";
        file.flush();
        if (current_node->terminal()) {
            break;
        }
        side = flip_side(side);
    }

    int count_b = current_node->board().count(CellState::BLACK);
    int count_w = current_node->board().count(CellState::WHITE);
    float mlresult = current_node->board().get_result(player_side);
    std::cout << "@ result : black=" << count_b << " white=" << count_w << std::endl;
    std::cout << "mlresult=" << mlresult << std::endl;

    file.close();

    return 0;
}
