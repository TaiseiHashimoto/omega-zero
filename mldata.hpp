#pragma once


#include "board.hpp"
#include "node.hpp"


typedef struct {
    BitBoard black_bitboard;
    BitBoard white_bitboard;
    Side side;
    Action action;
    float Q;
    float result;
    bool legal_flags[64];
    float posteriors[64];
} entry_t;

void pack_data(Node *node, float result, entry_t& output);
void save_game(std::vector<Node*>& game_path, float result, const char* file_name);
