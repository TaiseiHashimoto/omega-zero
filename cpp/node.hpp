#pragma once

#include <ostream>
#include <random>

#include "board.hpp"


class Node
{
public:
    Node(Board board, Side side, float prior, Node* parent = NULL);
    ~Node();


    // getter
    const Board& board() const;
    Side side() const;
    Node* parent() const;
    std::vector<Node*>& children();
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
    Node* select_child() const;
    void backpropagete(float value, Node* stop_node);
    Node* next_node(std::default_random_engine &engine);
    void add_exploration_noise(std::default_random_engine &engine);
    void set_prior(const float prior);

private:
    Board m_board;
    Side m_side;
    Node* m_parent;
    std::vector<Node*> m_children;
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

std::ostream& operator<<(std::ostream& os, const Node& node);
