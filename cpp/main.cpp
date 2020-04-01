#include <iostream>
#include <thread>
#include <vector>
#include <cstdlib>
#include <unistd.h>

#include "mcts.hpp"
#include "mldata.hpp"
#include "server.hpp"
#include "misc.hpp"


void collect_mldata(int thread_id, int n_game, int n_simulation, const char *fname) {
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
        play_game(n_simulation, server_sock, history, engine);

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

        if (i % 5 == 0) {
            auto end = std::chrono::system_clock::now();
            int elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
            float remaining = (float)(elapsed) / (i + 1) * (n_game - i - 1) / 60;
            printf("[%2d] i=%d (%d sec)  result:%.3f  remaining~%.2f min\n", thread_id, i, elapsed, result, remaining);
        }

#if USE_CACHE
        if (thread_id == 0) {
            int size, access_count, hit_count;
            std::tie(size, access_count, hit_count) = get_cache_stats();
            printf("CACHE INFO   size=%d access_count=%d hit_count=%d hit_rate=%f\n", size, access_count, hit_count, (float)hit_count/access_count);
        }
#endif
    }

    close(server_sock);
}


int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: main [generation] [n_thread] [n_game] [n_simulation] ([device_idx])\n");
        exit(-1);
    }
    int generation = atoi(argv[1]);
    int n_thread = atoi(argv[2]);
    int n_game = atoi(argv[3]);
    int n_simulation = atoi(argv[4]);
    int n_game_thread = (n_game + n_thread - 1) / n_thread;
    short int device_id = 0;
    if (argc == 6) {
        device_id = atoi(argv[5]);
    }

    char root_path[100];
    get_root_path(argv[0], root_path);
    char model_fname[100];
    sprintf(model_fname, "%s/model/model_jit_%d.pt", root_path, generation);

    pid_t server_pid = create_server_process(n_thread, model_fname, device_id);
    (void)server_pid;
    // printf("server_pid=%d\n", server_pid);

    std::vector<std::thread> client_threads(n_thread);
    std::vector<char*> fnames(n_thread);

    for (int i = 0; i < n_thread; i++) {
        fnames[i] = new char[100];
        sprintf(fnames[i], "%s/mldata/%d_%d.dat", root_path, generation, i);
        printf("start creating %s\n", fnames[i]);
        client_threads[i] = std::thread(collect_mldata, i, n_game_thread, n_simulation, fnames[i]);
    }
    for (int i = 0; i < n_thread; i++) {
        client_threads[i].join();
        delete[] fnames[i];
    }

    return 0;
}
