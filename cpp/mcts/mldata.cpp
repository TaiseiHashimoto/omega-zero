#include <iostream>
#include <fstream>

#include "mldata.hpp"
#include "node.hpp"
#include "board.hpp"


void pack_data(GameNode* node, float result, entry_t &entry) {
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

void save_game(std::vector<GameNode*> &history, float result, const char* file_name) {
    // result: soft result from black side
    printf("file_name = %s %d\n", file_name, history.size());
    std::ofstream file(file_name, std::ios::binary | std::ios::app);
    for (auto node : history) {
        entry_t entry;
        pack_data(node, result, entry);
        file.write(reinterpret_cast<char*>(&entry), sizeof(entry_t));
        result *= -1;  // side alternates every time
    }
    file.close();
}
