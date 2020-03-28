#include <iostream>
#include <fstream>

#include "mldata.hpp"
#include "node.hpp"
#include "board.hpp"


void pack_data(Node* node, float result, entry_t &entry) {
    Board board = node->board();
    const std::vector<bool>& legal_flags = node->legal_flags();
    const std::vector<float>& posteriors = node->posteriors();

    entry.black_bitboard = board.get_black_board();
    entry.white_bitboard = board.get_white_board();
    entry.side = node->side();
    entry.action = node->action();
    entry.Q = node->Q();
    entry.result = result;
    std::copy(legal_flags.begin(), legal_flags.end(), std::begin(entry.legal_flags));
    std::copy(posteriors.begin(), posteriors.end(), std::begin(entry.posteriors));
}

void save_game(std::vector<Node*> &game_path, float result, const char* file_name) {
    // result: soft result from black side
    // std::ofstream file(file_name, std::ios::app);
    std::ofstream file(file_name);
    for (auto node : game_path) {
        entry_t entry;
        pack_data(node, result, entry);
        file.write(reinterpret_cast<char*>(&entry), sizeof(entry));
        result *= -1;  // side alternates every time
    }
    file.close();
}
