#pragma once

#include <unistd.h>
#include <vector>

#include "board.hpp"


#define UNIXDOMAIN_PATH "/tmp/reversi_server.sock"
#define TIMEOUT_USEC 500
#define MAX_TIMEOUT 4


typedef struct {
    BitBoard black_board;
    BitBoard white_board;
    Side side;
    bool legal_flags[64];
} input_t;

typedef struct {
    float priors[64];
    float value;
} output_t;


pid_t create_server_process(int n_thread, const char *model_fname, short int device_idx);

int connect_to_server();

void request(int server_sock, const Board board, const Side side, const std::vector<bool>& legal_flags, std::vector<float>& priors, float& value);
