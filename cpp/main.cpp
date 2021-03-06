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


void collect_mldata(int thread_id, int n_game, const char *fname) {
    int server_sock = connect_to_server();  // NN server

    if (access(fname, F_OK) != -1) {
        fprintf(stderr, "ERROR: data file %s already exists\n", fname);
        close(server_sock);
        return;
    }

    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());

    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < n_game; i++) {
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
            float remaining = (float)(elapsed) / (i + 1) * (n_game - i - 1) / 60;
            printf("[%3d] i=%d (%d sec) result=%.3f remaining~%.2f min\n", thread_id, i, elapsed, result, remaining);
        }
    }

    close(server_sock);
}


int main(int argc, char *argv[]) {
    if ((argc < 3) || (argc > 3 && argv[3][0] != '-')) {
        std::cerr << "Usage: main exp_id generation [--device_id=ID]" << std::endl;
        exit(-1);
    }
    int exp_id = atoi(argv[1]);
    int generation = atoi(argv[2]);
    std::cout << "exp_id = " << exp_id << std::endl;
    std::cout << "generation = " << generation << std::endl;

    char exp_path[100];
    get_exp_path(argv[0], exp_id, exp_path);
    std::cout << "exp_path = " << exp_path << std::endl;

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
                fprintf(stderr, "unknown option\n");
                exit(-1);
        }
    }
    std::cout << "device_id = " << device_id << std::endl;

    init_config(exp_path, /*generation=*/-1, device_id);  // use best model
    const auto& config = get_config();

    int n_game_each = (config.n_game + config.n_thread - 1) / config.n_thread;

    pid_t server_pid = create_server_process();
    (void)server_pid;

    std::vector<std::thread> client_threads(config.n_thread);
    std::vector<char*> fnames(config.n_thread);

    for (int i = 0; i < config.n_thread; i++) {
        fnames[i] = new char[100];
        sprintf(fnames[i], "%s/mldata/%d_%d.dat", exp_path, generation, i);
        // printf("start creating %s\n", fnames[i]);
        client_threads[i] = std::thread(collect_mldata, i, n_game_each, fnames[i]);
    }
    for (int i = 0; i < config.n_thread; i++) {
        client_threads[i].join();
        delete[] fnames[i];
    }

    return 0;
}
