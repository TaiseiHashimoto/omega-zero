#pragma once

#include <ostream>
#include <random>
#include <map>
#include <list>
#include <tuple>
#include <mutex>

#include "board.hpp"


class GameNode
{
public:
    GameNode(Board board, Side side, float prior, GameNode* parent = NULL);
    ~GameNode();


    // getter
    const Board& board() const;
    Side side() const;
    GameNode* parent() const;
    const std::vector<GameNode*>& children() const;
    std::vector<GameNode*>& children_();
    int N() const;
    float Q() const;
    float prior() const;
    float value() const;
    bool pass() const;
    bool terminal() const;
    Action action() const;
    const std::vector<Action>& legal_actions() const;
    const std::vector<bool>& legal_flags() const;
    const std::vector<float>& posteriors() const;
    bool expanded() const;

    void expand(int server_sock);
    void add_children(const std::vector<float>& priors);
    GameNode* select_child() const;
    void backpropagete(float value, GameNode* stop_node);
    GameNode* next_node(float tau, std::default_random_engine& engine);
    void add_exploration_noise(std::default_random_engine& engine);
    void set_prior(const float prior);

private:
    Board m_board;
    Side m_side;
    GameNode* m_parent;
    std::vector<GameNode*> m_children;
    int m_N;
    float m_Q;
    float m_prior;
    float m_value;
    bool m_pass;
    bool m_terminal;
    Action m_action;
    std::vector<Action> m_legal_actions;
    std::vector<bool> m_legal_flags;
    std::vector<float> m_posteriors;
};

std::ostream& operator<<(std::ostream& os, const GameNode& node);
