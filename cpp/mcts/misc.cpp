#include <iostream>
#include <cassert>
#include <random>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <unistd.h>
#include <libgen.h>

#include "board.hpp"
#include "node.hpp"


void p() {
    std::cout << "\n";
}
void p(GameNode* node) {
    std::cout << *node;
}

void random_dirichlet(std::default_random_engine& engine, const float alpha, std::vector<float> &output) {
    std::gamma_distribution<double> gamma(alpha, 1.0);
    double sum_g = 0;
    for (unsigned int i = 0; i < output.size(); i++) {
        float g = gamma(engine);
        output[i] = g;
        sum_g += g;
    }
    for (unsigned int i = 0; i < output.size(); i++) {
        output[i] /= sum_g;
    }
}

Action parse_action(std::string input) {
    if (input.size() == 4) {
        if (input == "pass") {
            return SpetialAction::PASS;
        } else if (input == "back") {
            return SpetialAction::BACK;
        }
    } else if (input.size() == 2) {
        int col = input[0] - 'a';
        int row = input[1] - '1';
        if (col < 0 || col >= 8 || row < 0 || row >= 8) {
            return SpetialAction::INVALID;
        }
        return col + row * 8;
    }
    return SpetialAction::INVALID;
}

// count bit = 1
int bit_count(uint64_t x) {
    x = ((x & 0xaaaaaaaaaaaaaaaa) >> 1) 
      +  (x & 0x5555555555555555); 
    x = ((x & 0xcccccccccccccccc) >> 2) 
      +  (x & 0x3333333333333333); 
    x = ((x & 0xf0f0f0f0f0f0f0f0) >> 4) 
      +  (x & 0x0f0f0f0f0f0f0f0f); 
    x = ((x & 0xff00ff00ff00ff00) >> 8) 
      +  (x & 0x00ff00ff00ff00ff); 
    x = ((x & 0xffff0000ffff0000) >> 16) 
      +  (x & 0x0000ffff0000ffff); 
    x = ((x & 0xffffffff00000000) >> 32) 
      +  (x & 0x00000000ffffffff);
    return (int)x;
}

void get_exp_path(const char *prog_name, int exp_id, char *output) {
    // Assume that program is in ROOT/cpp/bin/
    char *retval = realpath(prog_name, output);
    if (retval == NULL) {
        fprintf(stderr, "MISC  realpath error %s\n", strerror(errno));
        exit(-1);
    }
    char *root_path = dirname(dirname(dirname(output)));
    sprintf(output, "%s/exp/%d", root_path, exp_id);
}
