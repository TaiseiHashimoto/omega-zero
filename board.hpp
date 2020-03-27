#pragma once

#include <ostream>
#include <vector>


enum class Side : std::uint8_t
{
    BLACK,
    WHITE
};

enum class PlayerType : std::uint8_t
{
    HUMAN,
    COMPUTER
};

Side flip_side(Side side);

std::ostream& operator<<(std::ostream& os, Side side);

enum class CellState : std::uint8_t
{
    EMPTY,
    BLACK,
    WHITE
};
typedef uint64_t BitBoard;
typedef uint8_t Action;

std::ostream& operator<<(std::ostream& os, Action action);


struct SpetialAction {
    static const int PASS = 101;
    static const int BACK = 102;
    static const int INVALID = 201;
};


class Board
{
public:
    static const int WIDTH = 8;
    static const int HEIGHT = 8;

    Board();

    CellState loc(int col, int row) const;

    void reset();
    void back();

    bool is_legal_action(Action action, Side side) const;
    std::vector<Action> get_all_legal_actions(Side side) const;

    void place_disk(Action action, Side side);

    int count(CellState target) const;
    int get_disk_num() const;
    float get_result(Side side) const;
    bool is_full() const;

    BitBoard get_black_board() const;
    BitBoard get_white_board() const;
    BitBoard get_player_board(Side side) const;
    BitBoard get_opponent_board(Side side) const;
    BitBoard make_legal_board(Side side) const;
    void set_boards(BitBoard player_board, BitBoard opponent_board, Side side);

private:
    int m_disk_num;  // 現在の石数
    BitBoard m_black_board;
    BitBoard m_white_board;
};

std::ostream& operator<<(std::ostream& os, const Board& board);
