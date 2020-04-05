#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <numeric>
#include <algorithm>

#include "node.hpp"
#include "mcts.hpp"
#include "server.hpp"
#include "misc.hpp"
#include "config.hpp"


GameNode::GameNode(Board board, Side side, float prior, GameNode* parent) {
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

GameNode::~GameNode() {
    for (auto& child : m_children) {
        safe_delete(child);
    }
}

int GameNode::N() const {
    return m_N;
}

float GameNode::Q() const {
    return m_Q;
}

float GameNode::prior() const {
    return m_prior;
}

float GameNode::value() const {
    return m_value;
}

const Board& GameNode::board() const {
    return m_board;
}

const std::vector<Action>& GameNode::legal_actions() const {
    return m_legal_actions;
}

const std::vector<bool>& GameNode::legal_flags() const {
    return m_legal_flags;
}

const std::vector<float>& GameNode::posteriors() const {
    return m_posteriors;
}

GameNode* GameNode::parent() const {
    return m_parent;
}

Side GameNode::side() const {
    return m_side;
}

const std::vector<GameNode*>& GameNode::children() const {
    return m_children;
}

std::vector<GameNode*>& GameNode::children_() {
    return m_children;
}

bool GameNode::expanded() const {
    return m_children.size() > 0;
}

bool GameNode::pass() const {
    return m_pass;
}

bool GameNode::terminal() const {
    return m_terminal;
}

Action GameNode::action() const {
    return m_action;
}


void GameNode::expand(int server_sock) {
    // get all legal actions and check pass
    m_legal_actions = m_board.get_all_legal_actions(m_side);
    for (auto action : m_legal_actions) {
        m_legal_flags[action] = true;
    }
    m_pass = (m_legal_actions.size() == 0);  // no legal action

    // terminal if double pass or no empty cell
    m_terminal = (m_parent && (m_pass && m_parent->pass())) || m_board.is_full();
    if (m_terminal) {
        m_value = m_board.get_result(m_side);  // substitute result for NN output
        return;
    }

    std::vector<float> priors(64);  // softmax-ed priors;
    request(server_sock, m_board, m_side, m_legal_flags, priors, m_value);

    add_children(priors);
}

void GameNode::add_children(const std::vector<float>& priors) {
    if (m_pass) {
        Board new_board(m_board);
        GameNode* child_node = new GameNode(new_board, flip_side(m_side), 1.0, this);
        m_children.push_back(child_node);
    } else {
        for (unsigned int i = 0; i < m_legal_actions.size(); i++) {
            auto action = m_legal_actions[i];
            Board new_board(m_board);
            new_board.place_disk(action, m_side);
            GameNode* child_node = new GameNode(new_board, flip_side(m_side), priors[action], this);
            m_children.push_back(child_node);
        }
    }
}

void GameNode::backpropagete(float value, GameNode* stop_node) {
    m_Q = (m_Q * m_N + value) / (m_N + 1);
    m_N += 1;
    // p("backpropagete");
    // p(this);
    if (this != stop_node) {
        m_parent->backpropagete(-value, stop_node);  // flip value for opponent
    }
}

GameNode* GameNode::select_child() const {
    const auto& config = get_config();

    float max_score = -1;  // -1 <= score
    GameNode* selected = nullptr;
    // printf("NODE  selecting...\n");
    // int i = 0;
    for (const auto& child : m_children) {
        float value_score = -child->Q();  // flip opponent's value
        // TODO: log term necessary?
        float prior_score = child->prior() * std::sqrt(m_N) / (child->N() + 1);
        float score = value_score + config.c_puct * prior_score;
        // std::cout << "NODE  " << m_legal_actions[i] << ":(" << value_score << "," << prior_score << ") ";
        // i++;
        if (max_score <= score) {
            max_score = score;
            selected = child;
        }
    }
    assert(selected != nullptr);
    // p();
    return selected;
}

// return next node, set m_action and m_posteriors
GameNode* GameNode::next_node(float tau, std::default_random_engine& engine) {
    if (m_pass) {
        assert(m_children.size() == 1);
        m_action = SpetialAction::PASS;
        return m_children[0];
    }

    bool stochastic = (tau > 0.01);
    float tau_inv = stochastic ? 1.0 / tau : 1.0;

    std::vector<float> ratios;
    float ratio_sum = 0;
    float ratio_max = 0;
    unsigned int ratio_max_idx = 64;
    for (unsigned int i = 0; i < m_children.size(); i++) {
        float ratio = std::pow((float)m_children[i]->N(), tau_inv);
        ratios.push_back(ratio);
        ratio_sum += ratio;
        if (ratio > ratio_max) {
            ratio_max = ratio;
            ratio_max_idx = i;
        }
    }

    assert(ratio_sum >= 1.0);

    unsigned int selected=64;  // TODO: delete initialization

    if (stochastic) {
        // select child according to visited count (ratio)
        std::uniform_real_distribution<float> uniform(0., ratio_sum);
        float rnd = uniform(engine);
        for (unsigned int i = 0; i < m_children.size(); i++) {
            if (rnd <= ratios[i]) {
                selected = i;
                break;
            }
            rnd -= ratios[i];
        }
        // calculate posterior
        for (unsigned int i = 0; i < m_children.size(); i++) {
            Action action = m_legal_actions[i];
            m_posteriors[action] = ratios[i] / ratio_sum;
        }
    } else {
        selected = ratio_max_idx;
        Action action = m_legal_actions[selected];
        m_posteriors[action] = 1.0;
    }

    assert(selected < m_children.size());
    m_action = m_legal_actions[selected];

    return m_children[selected];
}

void GameNode::add_exploration_noise(std::default_random_engine& engine) {
    const auto& config = get_config();

    int n = m_children.size();
    std::vector<float> noise(n);
    if (config.d_alpha == 0) {
        std::fill(noise.begin(), noise.end(), 0);
    } else {
        random_dirichlet(engine, config.d_alpha, noise);
    }

    // std::cout << "NODE  " << "noise = [";
    for (int i = 0; i < n; i++) {
        float org_prior = m_children[i]->prior();
        float new_prior =  org_prior * (1 - config.e_frac) + noise[i] * config.e_frac;
        m_children[i]->set_prior(new_prior);
        // std::cout << noise[i] << " ";
    }
    // std::cout << "]" << std::endl;
}

void GameNode::set_prior(const float prior) {
    m_prior = prior;
}

std::ostream& operator<<(std::ostream& os, const GameNode& node) {
    os << node.board();
    os << node.side() << std::endl;
    if (node.expanded()) {
        unsigned int n_legal_actions = node.legal_actions().size();
        if (n_legal_actions > 0) {
            std::vector<int> idxs(n_legal_actions);
            const auto& children = node.children();
            std::iota(idxs.begin(), idxs.end(), 0);
            std::sort(idxs.begin(), idxs.end(),
                [&children](int idx1, int idx2) {
                    return children[idx1]->prior() > children[idx2]->prior();
                });
            for (int idx : idxs) {
                os << node.legal_actions()[idx] << "(" << children[idx]->prior() << ") ";
            }
        } else {
            os << "pass ";
        }
    } else {
        os << "not expanded  ";
    }
    os << "\nN=" << node.N() << " Q=" << node.Q() << " v=" << node.value() << " t=" << node.terminal() << std::endl;

    int count_b = node.board().count(CellState::BLACK);
    int count_w = node.board().count(CellState::WHITE);
    os << "disk: " << "black=" << count_b << " white=" << count_w << std::endl;
    return os;
}
