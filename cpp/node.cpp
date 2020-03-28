#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <random>

#include "node.hpp"
#include "mcts.hpp"
#include "network.hpp"
#include "misc.hpp"


Node::Node(Board board, Side side, float prior, Node* parent) {
    m_board = board;
    m_side = side;
    m_parent = parent;
    m_N = 0;
    m_Q = 0;
    m_prior = prior;
    m_value = 0;
    m_pass = false;
    m_terminal = false;  // m_terminal must be initialized as false
    m_action = SpetialAction::INVALID;
    m_legal_flags.resize(64, false);
    m_posteriors.resize(64, 0);
}

Node::~Node() {
    for (auto& child : m_children) {
        safe_delete(child);
    }
}

int Node::N() const {
    return m_N;
}

float Node::Q() const {
    return m_Q;
}

float Node::prior() const {
    return m_prior;
}

float Node::value() const {
    return m_value;
}

const Board& Node::board() const {
    return m_board;
}

const std::vector<Action>& Node::legal_actions() const {
    return m_legal_actions;
}

const std::vector<bool>& Node::legal_flags() const {
    return m_legal_flags;
}

const std::vector<float>& Node::posteriors() const {
    return m_posteriors;
}

Node* Node::parent() const {
    return m_parent;
}

Side Node::side() const {
    return m_side;
}

std::vector<Node*>& Node::children()  {
    return m_children;
}

bool Node::expanded() const {
    return m_children.size() > 0;
}

bool Node::pass() const {
    return m_pass;
}

bool Node::terminal() const {
    return m_terminal;
}

Action Node::action() const {
    return m_action;
}


void Node::expand(int server_sock) {
    // get all legal actions and check pass
    m_legal_actions = m_board.get_all_legal_actions(m_side);
    for (auto action : m_legal_actions) {
        m_legal_flags[action] = true;
    }
    m_pass = (m_legal_actions.size() == 0);  // no legal action

    // terminal if double pass or no empty cell
    // m_terminal = (m_board.get_disk_num() >= 6);
    m_terminal = (m_parent && (m_pass && m_parent->pass())) || m_board.is_full();
    if (m_terminal) {
        m_value = m_board.get_result(m_side);  // substitute result for NN output
        return;
    }

    std::vector<float> priors(64);  // softmax-ed priors;
    // send request to server and receive response
    // input: m_board, m_side, m_legal_flags / output: priors, m_value
    request(server_sock, m_board, m_side, m_legal_flags, priors, m_value);

    add_children(priors);
}

void Node::add_children(const std::vector<float>& priors) {
    if (m_pass) {
        Board new_board(m_board);
        Node* child_node = new Node(new_board, flip_side(m_side), 1.0, this);
        m_children.push_back(child_node);
    } else {
        for (unsigned int i = 0; i < m_legal_actions.size(); i++) {
            auto action = m_legal_actions[i];
            Board new_board(m_board);
            new_board.place_disk(action, m_side);
            Node* child_node = new Node(new_board, flip_side(m_side), priors[action], this);
            m_children.push_back(child_node);
        }
    }
}

void Node::backpropagete(float value, Node* stop_node) {
    m_Q = (m_Q * m_N + value) / (m_N + 1);
    m_N += 1;

    // p("backpropagete");
    // p(this);

    if (this != stop_node) {
        m_parent->backpropagete(-value, stop_node);  // flip value for opponent
    }
}

Node* Node::select_child() const {
    float max_score = -1;  // -1 <= score <= 1
    Node* selected;
    // printf("selecting...\n");
    // int i = 0;
    for (const auto& child : m_children) {
        float value_score = -child->Q();  // flip opponent's value
        // TODO: log term necessary?
        float prior_score = child->prior() * std::sqrt(m_N) / (child->N() + 1);
        float score = value_score + C_PUCT * prior_score;

        // std::cout << m_legal_actions[i] << ":(" << value_score << "," << prior_score << ") ";
        // i++;

        if (max_score < score) {
            max_score = score;
            selected = child;
        }
    }
    // p();
    return selected;
}

// return next node, set m_action and m_posteriors
Node* Node::next_node(std::default_random_engine &engine) {
    if (m_pass) {
        assert(m_children.size() == 1);
        m_action = SpetialAction::PASS;
        return m_children[0];
    }

    std::vector<float> ratios;
    float ratio_sum = 0;
    for (const auto& child : m_children) {
        int count = child->N();
        float ratio = static_cast<float>(count);  // tau not used because tau = 1
        ratios.push_back(ratio);
        ratio_sum += ratio;
    }

    unsigned int selected;
    unsigned int n_children = m_children.size();

    assert(ratio_sum >= 1.0);

    // select child according to visited count (ratio)
    std::uniform_real_distribution<float> uniform(0., ratio_sum);
    float rnd = uniform(engine);
    for (unsigned int i = 0; i < n_children; i++) {
        if (rnd <= ratios[i]) {
            selected = i;
            break;
        }
        rnd -= ratios[i];
    }
    assert(selected >= 0 && selected < ratios.size());
    m_action = m_legal_actions[selected];

    // calculate posterior
    for (unsigned int i = 0; i < n_children; i++) {
        Action action = m_legal_actions[i];
        m_posteriors[action] = ratios[i] / ratio_sum;
    }

    // TODO: delete this
    for (auto post : m_posteriors) {
        assert(post >= 0 && post <= 1);
    }

    return m_children[selected];
}

void Node::add_exploration_noise(std::default_random_engine &engine) {
    int n = m_children.size();
    std::vector<float> noise(n);
    // TODO: delete case 0
    if (DIRICHLET_ALPHA == 0) {
        std::fill(noise.begin(), noise.end(), 0);
    } else {
        random_dirichlet(engine, DIRICHLET_ALPHA, noise);
    }

    for (int i = 0; i < n; i++) {
        float org_prior = m_children[i]->prior();
        float new_prior =  org_prior * (1 - EXPLORATION_FRAC) + noise[i] * EXPLORATION_FRAC;
        m_children[i]->set_prior(new_prior);
    }
}

void Node::set_prior(const float prior) {
    m_prior = prior;
}


std::ostream& operator<<(std::ostream& os, const Node& node) {
    os << node.board();
    if (node.expanded()) {
        if (node.legal_actions().size() > 0) {
            for (const auto& action : node.legal_actions()) {
                os << action << " ";
            }
        } else {
            os << "Pass ";
        }
        os << node.side() << "\n";
    } else {
        os << "not expanded  ";
    }
    os << "N=" << node.N() << " Q=" << node.Q() << " v=" << node.value() << " t=" << node.terminal() << "\n";
    return os;
}
