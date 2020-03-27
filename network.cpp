#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include <cstdarg>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

#include "network.hpp"
#include "mcts.hpp"
#include "board.hpp"
#include "misc.hpp"


namespace {

struct sockaddr_un server_addr;

}

pid_t create_server_process() {
    init_addr();
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "fork error\n");
        exit(-1);
    } else if (pid == 0) {
        run_server();
        printf("server exit\n");
        exit(0);
    }
    // wait for server to start
    sleep(1);
    return pid;
}

void init_addr() {
    memset(&server_addr, 0, sizeof(struct sockaddr_un));
    server_addr.sun_family = AF_UNIX;
    strcpy(server_addr.sun_path, UNIXDOMAIN_PATH);
}

int connect_to_server() {
    int server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_sock < 0){
        fprintf(stderr, "socket error %s\n", strerror(errno));
        exit(-1);
    }
    if (connect(server_sock, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_un)) < 0){
        fprintf(stderr, "connect error %s\n", strerror(errno));
        exit(-1);
    }
    return server_sock;
}

int connect_to_clients(int* client_socks) {
    int listen_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_sock < 0){
        fprintf(stderr, "socket error %s\n", strerror(errno));
        exit(-1);
    }
    if(bind(listen_sock, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_un)) < 0){
        fprintf(stderr, "bind error %s\n", strerror(errno));
        exit(-1);
    }
    if(listen(listen_sock, 5) < 0){
        fprintf(stderr, "listen error %s\n", strerror(errno));
        exit(-1);
    }
    for (int i = 0; i < N_THREAD; i++) {
        client_socks[i] = accept(listen_sock, NULL, NULL);
        if(client_socks[i] < 0){
            fprintf(stderr, "accept error %s\n", strerror(errno));
            exit(-1);
        }
    }
    return listen_sock;
}

void request(int server_sock, const Board board, const Side side, const std::vector<bool>& legal_flags, std::vector<float>& priors, float& value) {
    int retval;

    input_t send_data;
    send_data.black_board = board.get_black_board();
    send_data.white_board = board.get_white_board();
    send_data.side = side;
    std::copy(legal_flags.begin(), legal_flags.end(), std::begin(send_data.legal_flags));

    retval = write(server_sock, &send_data, sizeof(input_t));
    if (retval < 0){
        fprintf(stderr, "write error %s\n", strerror(errno));
        exit(-1);
    }

    output_t recv_data;
    retval = read(server_sock, &recv_data, sizeof(output_t));
    if (retval < 0){
        fprintf(stderr, "read error %s\n", strerror(errno));
        exit(-1);
    }

    std::copy(std::begin(recv_data.priors), std::end(recv_data.priors), priors.begin());
    value = recv_data.value;
}

void run_server() {
    unlink(UNIXDOMAIN_PATH);  // remove file

    int client_socks[N_THREAD];
    int listen_sock = connect_to_clients(client_socks);
    // fprintf(stdout, "accepted %d clients\n", N_THREAD);

    // initialization for select()
    fd_set fds_org;
    int maxfd = 0;
    FD_ZERO(&fds_org);
    for (int i = 0; i < N_THREAD; i++) {
        FD_SET(client_socks[i], &fds_org);
        maxfd = std::max(client_socks[i], maxfd);
    }

    int retval;
    int n_disc = 0;

    while (true) {  // loop until all clients disconnect
        input_t recv_data[N_THREAD];
        std::vector<int> to_respond;
        int n_recv = 0;
        int timeout_count = 0;

        // receive data
        while (n_recv + n_disc < N_THREAD && timeout_count < MAX_TIMEOUT) {
            fd_set fds = fds_org;
            if (n_recv == 0) {    // no waiting client
                retval = select(maxfd+1, &fds, NULL, NULL, NULL);
            } else {
                struct timeval tv;
                tv.tv_sec = 0;
                tv.tv_usec = TIMEOUT_USEC;
                retval = select(maxfd+1, &fds, NULL, NULL, &tv);
                timeout_count += 1;
            }

            if (retval < 0) {
                fprintf(stderr, "select error %s\n", strerror(errno));
                exit(-1);
            } else if (retval == 0) {
                // fprintf(stdout, "timeout (timeout_count = %d, n_recv = %d)\n", timeout_count, n_recv);
                continue;
            }

            for (int i = 0; i < N_THREAD; i++) {
                if (FD_ISSET(client_socks[i], &fds)) {
                    char buf[100];
                    memset(buf, 0, sizeof(buf));
                    retval = read(client_socks[i], &recv_data[i], sizeof(input_t));

                    if (retval < 0) {
                        fprintf(stderr, "read error %s\n", strerror(errno));
                        exit(-1);
                    } else if (retval == 0) {
                        // fprintf(stdout, "disconnected by client %d\n", i);
                        FD_CLR(client_socks[i], &fds_org);  // stop monitoring this client
                        n_disc += 1;
                        continue;
                    }

                    to_respond.push_back(i);  // exclude disconnection
                    n_recv += 1;  // include disconnection
                    // fprintf(stdout, "receive %d from client %d\n", recv_data[i].num_i, i);
                }
                // else {
                //     // fprintf(stdout, "no data from client %d\n", i);
                // }
            }
            // printf("timeout_count = %d, n_recv = %d\n", timeout_count, n_recv);
        }

        if (n_disc == N_THREAD) {  // all clients disconnected
            break;
        }

        output_t send_data[N_THREAD];
        // process data (treat all clients equally)
        for (int i = 0; i < N_THREAD; i++) {
            int n_legal_action = 0;
            for (int j = 0; j < 64; j++) {
                if (recv_data[i].legal_flags[j]) {
                    n_legal_action += 1;
                }
            }
            for (int j = 0; j < 64; j++) {
                if (recv_data[i].legal_flags[j]) {
                    send_data[i].priors[j] = 1.0 / n_legal_action;
                } else {
                    send_data[i].priors[j] = 0.0;
                }
            }

            int black_c = bit_count(recv_data[i].black_board);
            int white_c = bit_count(recv_data[i].white_board);
            if (recv_data[i].side == Side::BLACK) {
                send_data[i].value = (black_c - white_c) / 64.0;
            } else {
                send_data[i].value = (white_c - black_c) / 64.0;
            }
            // sprintf(send_data[i].num_s, "%d", recv_data[i].num_i);
        }

        // send data
        for (int idx : to_respond) {
            retval = write(client_socks[idx], &send_data[idx], sizeof(output_t));
            if (retval < 0){
                fprintf(stderr, "write error %s\n", strerror(errno));
                exit(-1);
            }
        }
        // fprintf(stdout, "---\n");
    }

    for (int i = 0; i < N_THREAD; i++) {
        close(client_socks[i]);
    }
    close(listen_sock);
}