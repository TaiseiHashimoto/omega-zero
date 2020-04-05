#include <iostream>
#include <thread>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <getopt.h>

#include "mcts.hpp"
#include "mldata.hpp"
#include "server.hpp"
#include "misc.hpp"
#include "config.hpp"


void collect_mldata(int thread_id, int total_game, const char *exp_path) {
    int server_sock = connect_to_server();  // NN server

    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());

    char fname[100], fname_merged[100];
    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < total_game; i++) {
        // create "[game id]_[thread id].dat"
        sprintf(fname, "%s/mldata/%d_%d.dat", exp_path, i, thread_id);
        if (access(fname, F_OK) != -1) {
            fprintf(stderr, "MAIN  WARNING: data file %s already exists\n", basename(fname));
            continue;
        }
        sprintf(fname_merged, "%s/mldata/%d.dat", exp_path, i);
        if (access(fname_merged, F_OK) != -1) {
            fprintf(stderr, "MAIN  WARNING: data file %s already exists\n", basename(fname_merged));
            continue;
        }

        std::vector<GameNode*> history;
        play_game(history, server_sock, engine);

        // printf("\n### history ###\n");
        // for (unsigned int i = 0; i < history.size(); i++) {
        //     p("i=", i);
        //     p(history[i]);
        //     p("action=", history[i]->action());
        // }
        // int count_b = history.back()->board().count(CellState::BLACK);
        // int count_w = history.back()->board().count(CellState::WHITE);
        float result = history.back()->board().get_result(Side::BLACK);
        save_game(history, result, fname);
        safe_delete(history[0]);  // delete root -> whole tree

        if (thread_id % 100 == 0) {
            auto end = std::chrono::system_clock::now();
            int elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
            float remaining = (float)(elapsed) / (i + 1) * (total_game - i - 1) / 60;
            printf("MAIN  [%2d] i=%d (%d sec)  result:%.3f  remaining~%.2f min\n", thread_id, i, elapsed, result, remaining);
        }
    }

    close(server_sock);
}


int main(int argc, char *argv[]) {
    if ((argc < 2) || (argc > 2 && argv[2][0] != '-')) {
        std::cerr << "MAIN  Usage: main exp_id [--device_id=ID]" << std::endl;
        exit(-1);
    }
    int exp_id = atoi(argv[1]);
    std::cout << "MAIN  exp_id = " << exp_id << std::endl;

    char exp_path[100];
    get_exp_path(argv[0], exp_id, exp_path);
    std::cout << "MAIN  exp_path = " << exp_path << std::endl;

    int device_id = 0;

    int opt, longindex;
    const struct option longopts[] = {
        {"device_id", required_argument, NULL, 'd'},
        {0, 0, 0, 0}
    };
    while ((opt = getopt_long(argc, argv, "d:", longopts, &longindex)) != -1) {
        switch (opt) {
            case 'd':
                device_id = atoi(optarg);
                break;
            default:
                fprintf(stderr, "MAIN  unknown option\n");
                exit(-1);
        }
    }
    std::cout << "MAIN  device_id = " << device_id << std::endl;

    init_config(exp_path, device_id);

    const auto& config = get_config();
    int total_game_each = (config.total_game + config.n_thread - 1) / config.n_thread;

    pid_t server_pid = create_server_process();
    (void)server_pid;

    std::vector<std::thread> client_threads(config.n_thread);
    for (int i = 0; i < config.n_thread; i++) {
        client_threads[i] = std::thread(collect_mldata, i, total_game_each, exp_path);
    }
    for (int i = 0; i < config.n_thread; i++) {
        client_threads[i].join();
    }

    return 0;
}
