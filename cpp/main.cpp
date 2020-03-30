#include <iostream>
#include <thread>
#include <vector>
#include <cstdlib>
#include <unistd.h>

#include "mcts.hpp"
#include "mldata.hpp"
#include "server.hpp"
#include "misc.hpp"


void collect_mldata(int thread_id, int n_game, const char *file_name) {
    unlink(file_name);  // delete file
    int server_sock = connect_to_server();  // NN server

    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());

    for (int i = 0; i < n_game; i++) {
        std::vector<GameNode*> history;
        play_game(server_sock, history, engine);

        // printf("\n### history ###\n");
        // for (unsigned int i = 0; i < history.size(); i++) {
        //     p("i=", i);
        //     p(history[i]);
        //     p("action=", history[i]->action());
        // }
        int count_b = history.back()->board().count(CellState::BLACK);
        int count_w = history.back()->board().count(CellState::WHITE);
        float result = history.back()->board().get_result(Side::BLACK);
        save_game(history, result, file_name);
        safe_delete(history[0]);  // delete root -> whole tree
        printf("[%d] i=%d  black:%d white:%d result:%f\n", thread_id, i, count_b, count_w, result);
    }

    close(server_sock);
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: main [n_thread] [n_game]\n");
        exit(-1);
    }
    int n_thread = atoi(argv[1]);
    int n_game = atoi(argv[2]);

    short int device_idx = 0;
    pid_t server_pid = create_server_process(n_thread, device_idx);
    (void)server_pid;

    std::vector<std::thread> client_threads(n_thread);
    std::vector<char*> file_names(n_thread);

    for (int i = 0; i < n_thread; i++) {
        file_names[i] = new char[100];
        sprintf(file_names[i], "game%d.dat", i);
        client_threads[i] = std::thread(collect_mldata, i, n_game, file_names[i]);
    }
    for (int i = 0; i < n_thread; i++) {
        client_threads[i].join();
        delete[] file_names[i];
    }

    return 0;
}
