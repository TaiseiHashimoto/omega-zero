#include <array>
#include <cassert>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <bitset>

#include "board.hpp"
#include "misc.hpp"


Side flip_side(const Side side)
{
    switch (side) {
    case Side::BLACK:
        return Side::WHITE;
    case Side::WHITE:
        return Side::BLACK;
    default:
        assert(false);
        return side;
    }
}

std::ostream& operator<<(std::ostream& os, const Side side)
{
    switch (side) {
    case Side::BLACK:
        os << "black";
        break;
    case Side::WHITE:
        os << "white";
        break;
    default:
        assert(false);
    }
    return os;
}

namespace
{

BitBoard transfer(const BitBoard pos, int dir) {
    switch(dir) {
    case 0: // 左
        return (pos >> 1) & 0x7f7f7f7f7f7f7f7f;
    case 1: // 右
        return (pos << 1) & 0xfefefefefefefefe;
    case 2: // 上
        return (pos >> 8) & 0x00ffffffffffffff;
    case 3: // 下
        return (pos << 8) & 0xffffffffffffff00;
    case 4: // 左上
        return (pos >> 9) & 0x007f7f7f7f7f7f7f;
    case 5: // 右上
        return (pos >> 7) & 0x00fefefefefefefe;
    case 6: // 左下
        return (pos << 7) & 0x7f7f7f7f7f7f7f00;
    case 7: // 右下
        return (pos << 9) & 0xfefefefefefefe00;
    default:
        assert(false);
        return 0;
    }
}

}  // namespace

std::ostream& operator<<(std::ostream& os, Action action)
{
    if (action == SpetialAction::PASS) {
        os << "Pass";
    } else if (action == SpetialAction::BACK) {
        os << "BACK";
    } else if (action == SpetialAction::INVALID) {
        os << "INVALID";
    } else if (action >= 0 && action < 64) {
        os << static_cast<char>('a' + action % 8) << static_cast<char>('1' + action / 8);
    } else {
        fprintf(stderr, "unknown action (%d)\n", action);
        exit(-1);
    }
    return os;
}

Board::Board() {
    reset();
}

CellState Board::loc(int col, int row) const {
    Action position = (Action)(col + row * 8);
    BitBoard pos = (BitBoard)1 << position;
    if (m_black_board & pos) {
        return CellState::BLACK;
    } else if (m_white_board & pos) {
        return CellState::WHITE;
    }
    return CellState::EMPTY;
}

void Board::reset() {
    m_disk_num = 4;
    m_white_board = (BitBoard)1 << (3 + 3 * 8);
    m_black_board = (BitBoard)1 << (3 + 4 * 8);
    m_black_board |= (BitBoard)1 << (4 + 3 * 8);
    m_white_board |= (BitBoard)1 << (4 + 4 * 8);
}

bool Board::is_legal_action(Action action, Side side) const {
    if (action == SpetialAction::BACK) {
        return m_disk_num >= 6;
    } else if (action == SpetialAction::INVALID) {
        return false;
    }

    BitBoard legal_board = make_legal_board(side);

    if (action == SpetialAction::PASS) {
        return legal_board == 0;
    }

    BitBoard pos = (BitBoard)1 << action;
    if (pos & legal_board) {
        return true;
    }
    return false;
}

std::vector<Action> Board::get_all_legal_actions(const Side side) const {
    BitBoard legal_board = make_legal_board(side);
    std::vector<Action> legal_actions;
    for (Action pos = 0; pos < HEIGHT * WIDTH; pos++) {
        if ((legal_board >> pos) & 1) {
            legal_actions.push_back(pos);
        }
    }
    return legal_actions;
}

void Board::place_disk(Action action, Side side) {
    assert(is_legal_action(action, side));
    if (action == SpetialAction::PASS) {
        return;
    }

    BitBoard pos = (BitBoard)1 << action;
    BitBoard rev = 0;
    BitBoard player_board = get_player_board(side);
    BitBoard opponent_board = get_opponent_board(side);

    for (int dir = 0; dir < 8; dir++) {
        BitBoard rev_ = 0;
        BitBoard mask = transfer(pos, dir);
        while (mask != 0 && (mask & opponent_board)) {
            rev_ |= mask;
            mask = transfer(mask, dir);
        }
        if (mask & player_board) {    // reach my stone
            rev |= rev_;
        }
    }

    player_board ^= pos | rev;
    opponent_board ^= rev;
    set_boards(player_board, opponent_board, side);

    m_disk_num += 1;
    return;
}

int Board::count(CellState target) const {
    if (target == CellState::EMPTY) {
        BitBoard empty_board = ~(m_black_board | m_white_board);
        return bit_count(empty_board);
    } else if (target == CellState::BLACK) {
        return bit_count(m_black_board);
    } else if (target == CellState::WHITE) {
        return bit_count(m_white_board);
    }
    assert(false);
    return 0;
}

int Board::get_disk_num() const {
    return m_disk_num;
}

bool Board::is_full() const {
    return m_disk_num == 64;
}

float Board::get_result(Side side) const {
    int count_b = this->count(CellState::BLACK);
    int count_w = this->count(CellState::WHITE);
    // 1 if black win, -1 if white win, 0 if draw
    // float result = (count_b > count_w) - (count_b < count_w);
    // if (side == Side::BLACK) {
    //     return result;
    // } else {
    //     return -result;
    // }
    return (count_b - count_w) / 64.0;
}

BitBoard Board::get_black_board() const {
    return m_black_board;
}

BitBoard Board::get_white_board() const
{
    return m_white_board;
}

BitBoard Board::get_player_board(Side side) const
{
    switch(side) {
    case Side::BLACK:
        return m_black_board;
    case Side::WHITE:
        return m_white_board;
    default:
        assert(false);
        return 0;
    }
}

BitBoard Board::get_opponent_board(Side side) const {
    return get_player_board(flip_side(side));
}

BitBoard Board::make_legal_board(Side side) const {
    BitBoard player_board = get_player_board(side);
    BitBoard opponent_board = get_opponent_board(side);

    BitBoard hor_watch_board = opponent_board & 0x7e7e7e7e7e7e7e7e;
    BitBoard ver_watch_board = opponent_board & 0x00ffffffffffff00;
    BitBoard all_watch_board = opponent_board & 0x007e7e7e7e7e7e00;
    BitBoard empty_board = ~(player_board | opponent_board);
    BitBoard tmp;
    BitBoard legal_board = 0;

    // 左
    tmp = hor_watch_board & (player_board >> 1);
    tmp |= hor_watch_board & (tmp >> 1);
    tmp |= hor_watch_board & (tmp >> 1);
    tmp |= hor_watch_board & (tmp >> 1);
    tmp |= hor_watch_board & (tmp >> 1);
    tmp |= hor_watch_board & (tmp >> 1);
    legal_board |= empty_board & (tmp >> 1);

    // 右
    tmp = hor_watch_board & (player_board << 1);
    tmp |= hor_watch_board & (tmp << 1);
    tmp |= hor_watch_board & (tmp << 1);
    tmp |= hor_watch_board & (tmp << 1);
    tmp |= hor_watch_board & (tmp << 1);
    tmp |= hor_watch_board & (tmp << 1);
    legal_board |= empty_board & (tmp << 1);

    // 上
    tmp = ver_watch_board & (player_board >> 8);
    tmp |= ver_watch_board & (tmp >> 8);
    tmp |= ver_watch_board & (tmp >> 8);
    tmp |= ver_watch_board & (tmp >> 8);
    tmp |= ver_watch_board & (tmp >> 8);
    tmp |= ver_watch_board & (tmp >> 8);
    legal_board |= empty_board & (tmp >> 8);

    // 下
    tmp = ver_watch_board & (player_board << 8);
    tmp |= ver_watch_board & (tmp << 8);
    tmp |= ver_watch_board & (tmp << 8);
    tmp |= ver_watch_board & (tmp << 8);
    tmp |= ver_watch_board & (tmp << 8);
    tmp |= ver_watch_board & (tmp << 8);
    legal_board |= empty_board & (tmp << 8);

    // 左斜め上
    tmp = all_watch_board & (player_board >> 9);
    tmp |= all_watch_board & (tmp >> 9);
    tmp |= all_watch_board & (tmp >> 9);
    tmp |= all_watch_board & (tmp >> 9);
    tmp |= all_watch_board & (tmp >> 9);
    tmp |= all_watch_board & (tmp >> 9);
    legal_board |= empty_board & (tmp >> 9);

    // 右斜め上
    tmp = all_watch_board & (player_board >> 7);
    tmp |= all_watch_board & (tmp >> 7);
    tmp |= all_watch_board & (tmp >> 7);
    tmp |= all_watch_board & (tmp >> 7);
    tmp |= all_watch_board & (tmp >> 7);
    tmp |= all_watch_board & (tmp >> 7);
    legal_board |= empty_board & (tmp >> 7);

    // 左斜め下
    tmp = all_watch_board & (player_board << 7);
    tmp |= all_watch_board & (tmp << 7);
    tmp |= all_watch_board & (tmp << 7);
    tmp |= all_watch_board & (tmp << 7);
    tmp |= all_watch_board & (tmp << 7);
    tmp |= all_watch_board & (tmp << 7);
    legal_board |= empty_board & (tmp << 7);

    // 右斜め下
    tmp = all_watch_board & (player_board << 9);
    tmp |= all_watch_board & (tmp << 9);
    tmp |= all_watch_board & (tmp << 9);
    tmp |= all_watch_board & (tmp << 9);
    tmp |= all_watch_board & (tmp << 9);
    tmp |= all_watch_board & (tmp << 9);
    legal_board |= empty_board & (tmp << 9);

    return legal_board;
}


void Board::set_boards(BitBoard player_board, BitBoard opponent_board, Side side)
{
    if (side == Side::BLACK) {
        m_black_board = player_board;
        m_white_board = opponent_board;
    } else {
        m_black_board = opponent_board;
        m_white_board = player_board;
    }
}


std::ostream& operator<<(std::ostream& os, const Board& board)
{
    os << "  a b c d e f g h";
    for (int y = 0; y < Board::HEIGHT; ++y) {
        os << '\n' << (y + 1);
        for (int x = 0; x < Board::WIDTH; ++x) {
            os << ' ';
            CellState state = board.loc(x, y);
            switch (state) {
            case CellState::EMPTY:
                os << '-';
                break;
            case CellState::BLACK:
                os << 'B';
                break;
            case CellState::WHITE:
                os << 'W';
                break;
            }
        }
    }
    os << "\n";
    return os;
}
