#pragma once

#include <iostream>
#include <vector>
#include <random>

#include "board.hpp"
#include "node.hpp"


void p();
void p(GameNode* node);
template<class Head, class... Body>
void p(Head head, Body... body) {
    std::cout << head;
    p(body...);
}

template<class T>
void safe_delete(T*& p) {
    delete p;
    p = nullptr;
}

void random_dirichlet(std::default_random_engine &engine, const float alpha, std::vector<float> &output);

Action parse_action(std::string input);

int bit_count(uint64_t x);

void get_root_path(const char *program_name, char *output);
