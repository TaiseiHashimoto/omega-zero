#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include <cstdarg>
#include <chrono>
#include <unistd.h>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

#include "server.hpp"
#include "model.hpp"
#include "board.hpp"


#define PIPE_READ  0
#define PIPE_WRITE 1


namespace {

struct sockaddr_un server_addr;

int connect_to_clients(int n_thread, int pipe_fd, std::vector<int>& client_socks) {
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

    int retval = write(pipe_fd, "DONE", 4);  // send done message to parent process
    if (retval < 0){
        fprintf(stderr, "write error %s\n", strerror(errno));
        exit(-1);
    }

    for (int i = 0; i < n_thread; i++) {
        client_socks[i] = accept(listen_sock, NULL, NULL);
        if(client_socks[i] < 0){
            fprintf(stderr, "accept error %s\n", strerror(errno));
            exit(-1);
        }
    }
    return listen_sock;
}

void run_server(int n_thread, int pipe_fd, const char *model_fname, short int device_idx) {
    printf("server start\n");
    unlink(UNIXDOMAIN_PATH);  // remove file
    init_model(model_fname, device_idx);

    std::vector<int> client_socks(n_thread);
    int listen_sock = connect_to_clients(n_thread, pipe_fd, client_socks);
    printf("accepted %d clients\n", n_thread);

    // initialization for select()
    int maxfd = 0;
    fd_set fds_org;
    FD_ZERO(&fds_org);
    for (int i = 0; i < n_thread; i++) {
        FD_SET(client_socks[i], &fds_org);
        maxfd = std::max(client_socks[i], maxfd);
    }

    // static double total_time = 0;

    int retval;
    int n_disc = 0;

    input_t *recv_data = new input_t[n_thread];
    output_t *send_data = new output_t[n_thread];

    static int total_count = 0;  // total count of inference
    std::chrono::system_clock::time_point start, end;
    float elapsed;  // msec
    float work_time = 1.0;  // msec
    float occupancy_rate = 0.5;

    while (true) {  // loop until all clients disconnect
        std::vector<int> to_respond;
        int n_recv = 0;
        int timeout_count = 0;

        // receive data
        while (n_recv + n_disc < n_thread && timeout_count < MAX_TIMEOUT) {
            fd_set fds = fds_org;
            if (n_recv == 0) {    // no waiting client
                retval = select(maxfd+1, &fds, NULL, NULL, NULL);
            } else {
                struct timeval tv;
                tv.tv_sec = 0;
                tv.tv_usec = TIMEOUT_USEC;  // TODO: better way?
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

            for (int i = 0; i < n_thread; i++) {
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

        if (n_disc == n_thread) {  // all clients disconnected
            break;
        }

        occupancy_rate = occupancy_rate * 0.99 + (float)n_recv / n_thread * 0.01;

        start = std::chrono::system_clock::now();
        inference(n_thread, recv_data, send_data);

        // send data
        for (int idx : to_respond) {
            retval = write(client_socks[idx], &send_data[idx], sizeof(output_t));
            if (retval < 0){
                fprintf(stderr, "write error %s\n", strerror(errno));
                exit(-1);
            }
        }

        end = std::chrono::system_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-3;
        work_time = work_time * 0.99 + elapsed * 0.01;

        total_count += 1;
        if (total_count % 1000 == 0) {
            printf("%d: occupancy_rate=%.3f, work_time=%.3f\n", total_count, occupancy_rate, work_time);
        }
    }

    delete[] recv_data;
    delete[] send_data;

    for (int i = 0; i < n_thread; i++) {
        close(client_socks[i]);
    }
    close(listen_sock);
}

}  // namespace


pid_t create_server_process(int n_thread, const char *model_fname, short int device_idx) {
    // initialize server address
    memset(&server_addr, 0, sizeof(struct sockaddr_un));
    server_addr.sun_family = AF_UNIX;
    strcpy(server_addr.sun_path, UNIXDOMAIN_PATH);

    // create pipe to receive sign of preparation completion
    int pipe_c2p[2];
    if (pipe(pipe_c2p) < 0) {
        fprintf(stderr, "pipe error %s\n", strerror(errno));
        exit(-1);
    }

    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "fork error %s\n", strerror(errno));
        close(pipe_c2p[PIPE_READ]);
        close(pipe_c2p[PIPE_WRITE]);
        exit(-1);
    } else if (pid == 0) {  // child process
        run_server(n_thread, pipe_c2p[PIPE_WRITE], model_fname, device_idx);
        printf("server exit\n");
        exit(0);
    }

    // wait for server to start
    char buf[4];
    int retval = read(pipe_c2p[PIPE_READ], buf, 4);
    if (retval <= 0) {
        fprintf(stderr, "read error %s\n", strerror(errno));
        exit(-1);
    }
    if (strcmp(buf, "DONE") != 0) {
        fprintf(stderr, "message error \"%s\" != \"DONE\"\n", buf);
    }
    // printf("received \"%s\" from server\n", buf);
    close(pipe_c2p[0]);
    close(pipe_c2p[1]);

    return pid;
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


void request(int server_sock, const Board board, const Side side, const std::vector<bool>& legal_flags, std::vector<float>& priors, float& value) {
    // thread_local int call_count = 0;
    // thread_local float wait_time = 1.0;  // msec

    int retval;
    input_t send_data;
    send_data.black_board = board.get_black_board();
    send_data.white_board = board.get_white_board();
    send_data.side = side;
    std::copy(legal_flags.begin(), legal_flags.end(), std::begin(send_data.legal_flags));

    // auto start = std::chrono::system_clock::now();
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
    assert(retval > 0);

    // auto end = std::chrono::system_clock::now();
    // float elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-3;
    // wait_time = wait_time * 0.99 + elapsed * 0.01;
    // if (call_count % 100000 == 0) {
    //     printf("client wait_time=%f\n", wait_time);
    // }
    // call_count += 1;

    std::copy(std::begin(recv_data.priors), std::end(recv_data.priors), priors.begin());
    value = recv_data.value;
}
