#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <random>

#include "node.hpp"
#include "mcts.hpp"
#include "server.hpp"
#include "misc.hpp"


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

std::vector<GameNode*>& GameNode::children()  {
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
    // send request to server and receive response
    // input: m_board, m_side, m_legal_flags / output: priors, m_value
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
    float max_score = -1;  // -1 <= score
    GameNode* selected = nullptr;
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
    assert(selected != nullptr);
    // p();
    return selected;
}

// return next node, set m_action and m_posteriors
GameNode* GameNode::next_node(std::default_random_engine &engine) {
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

    unsigned int selected=1000;  // TODO: delete initialization
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
    assert(selected < ratios.size());
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

void GameNode::add_exploration_noise(std::default_random_engine &engine) {
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

void GameNode::set_prior(const float prior) {
    m_prior = prior;
}


std::ostream& operator<<(std::ostream& os, const GameNode& node) {
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
