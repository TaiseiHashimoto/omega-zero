#include <iostream>
#include <thread>
#include <vector>
#include <unistd.h>

#include "mcts.hpp"
#include "mldata.hpp"
#include "server.hpp"
#include "misc.hpp"


void collect_mldata(int thread_id, const char *file_name) {
    unlink(file_name);  // delete file
    int server_sock = connect_to_server();  // NN server

    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::vector<GameNode*> history;

    for (int i = 0; i < N_GAME; i++) {
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
        printf("[%d] ", thread_id);
        p("black:", count_b, " white:", count_w, " result:", result);

        save_game(history, result, file_name);
        safe_delete(history[0]);  // delete root -> whole tree
    }

    close(server_sock);
}


int main(int argc, char* argv[]) {
    short int device_idx = 0;
    pid_t server_pid = create_server_process(N_THREAD, device_idx);
    (void)server_pid;

    std::vector<std::thread> client_threads(N_THREAD);
    // std::string file_name_bas = "game"
    char file_name[100];

    for (int i = 0; i < N_THREAD; i++) {
        sprintf(file_name, "game%d.dat", i);
        client_threads[i] = std::thread(collect_mldata, i, file_name);
    }
    for (int i = 0; i < N_THREAD; i++) {
        client_threads[i].join();
    }

    return 0;
}
