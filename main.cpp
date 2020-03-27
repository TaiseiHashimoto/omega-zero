#include <iostream>
#include <unistd.h>

#include "mcts.hpp"
#include "network.hpp"


int main(int argc, char* argv[])
{
    pid_t server_pid = create_server_process();
    (void)server_pid;

    collect_mldata("game0.dat");

    return 0;
}
