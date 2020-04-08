#include <torch/torch.h>
#include <iostream>
#include <vector>

#include "model.hpp"
#include "server.hpp"
#include "config.hpp"


// /**
//     DO NOT use assignment operations because they can cause unintended change of parameters
// **/

ResBlockImpl::ResBlockImpl(int n_filter) {
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(n_filter, n_filter, 3).padding(1).bias(false)
    ));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(n_filter));
    conv2 = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(n_filter, n_filter, 3).padding(1).bias(false)
    ));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(n_filter));
}

torch::Tensor ResBlockImpl::forward(const torch::Tensor& x) {
    torch::Tensor identity = x;
    torch::Tensor y = x;
    y = torch::relu(bn1(conv1(y)));
    y = bn2(conv2(y));
    y = torch::relu(y + identity);
    return y;
}


OmegaNetImpl::OmegaNetImpl(int board_size, int n_action, int n_res_block, int res_filter, int policy_filter, int value_filter, int value_hidden) {
    // input channel : black / white / side
    conv = register_module("conv", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, res_filter, 3).padding(1).bias(false)),
        torch::nn::BatchNorm2d(res_filter),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
    ));

    res_blocks = register_module("res_blocks", torch::nn::Sequential());
    for (int i = 0; i < n_res_block; i++) {
        res_blocks->push_back(ResBlock(res_filter));
    }

    policy_head = register_module("policy_head", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(res_filter, policy_filter, 1).padding(0).bias(false)),
        torch::nn::BatchNorm2d(policy_filter),
        torch::nn::Flatten(),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(policy_filter * board_size * board_size, n_action)
    ));

    value_head = register_module("value_head", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(res_filter, value_filter, 1).padding(0).bias(false)),
        torch::nn::BatchNorm2d(value_filter),
        torch::nn::Flatten(),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(value_filter * board_size * board_size, value_hidden),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(value_hidden, 1)
    ));
}

std::tuple<torch::Tensor, torch::Tensor> OmegaNetImpl::forward(const torch::Tensor& black_board, const torch::Tensor& white_board, const torch::Tensor& side, const torch::Tensor& legal_flags) {
    torch::Tensor side_board = torch::ones_like(black_board) * side.view({-1, 1, 1});
    torch::Tensor player_board = black_board * (1 - side_board) + white_board * side_board;
    torch::Tensor opponent_board = black_board * side_board + white_board * (1 - side_board);

    torch::Tensor x = torch::stack({player_board, opponent_board}, 1);
    x = res_blocks->forward(conv->forward(x));

    torch::Tensor policy_logit = policy_head->forward(x);
    torch::Tensor policy = policy_logit + (legal_flags + 1e-45).log();
    policy = torch::softmax(policy, /*dim=*/ 1);

    torch::Tensor value_pred = torch::tanh(value_head->forward(x));
    value_pred = value_pred.squeeze(1);

    return std::make_tuple(policy, value_pred);
}


namespace {

torch::Device device{torch::kCPU};
OmegaNet omega_net{nullptr};
torch::Tensor position_board;

torch::Tensor make_all_variations(const torch::Tensor& board) {
    return torch::stack({
        board,
        board.flip(1),
        board.flip(2),
        board.permute({0, 2, 1}),
        board.flip(1).permute({0, 2, 1}).flip(1),
        torch::rot90(board, 1, {1, 2}),
        torch::rot90(board, 2, {1, 2}),
        torch::rot90(board, 3, {1, 2})
    }, /*dim=*/0);
}

torch::Tensor make_variation(const torch::Tensor& board, const torch::Tensor& norm_idx) {
    torch::Tensor board_vars = make_all_variations(board);
    torch::Tensor batch_range = torch::arange(board.size(0));
    return torch::index(board_vars, {norm_idx, batch_range});
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> normalize_board(const torch::Tensor& black_board, const torch::Tensor& white_board) {
    torch::Tensor black_board_vars = make_all_variations(black_board);
    torch::Tensor white_board_vars = make_all_variations(white_board);

    torch::Tensor score = ((black_board_vars + white_board_vars) * position_board).sum(-1).sum(-1);
    torch::Tensor norm_idx = torch::argmax(score, /*dim=*/0);

    torch::Tensor batch_range = torch::arange(black_board.size(0));
    return std::make_tuple(torch::index(black_board_vars, {norm_idx, batch_range}), torch::index(white_board_vars, {norm_idx, batch_range}), norm_idx);
}

torch::Tensor flip_idx(const torch::Tensor& norm_idx) {
    torch::Tensor idx5 = (norm_idx == 5).nonzero().view({-1});
    torch::Tensor idx7 = (norm_idx == 7).nonzero().view({-1});
    return norm_idx.index_fill(0, idx5, 7).index_fill(0, idx7, 5);
}

}

void init_model() {
    const auto& config = get_config();

    if (torch::cuda::is_available()) {
        device = {torch::kCUDA, (short int)config.device_id};
    }
    std::cout << "using " << device << std::endl;

    omega_net = OmegaNet(config.board_size, config.n_action, config.n_res_block, config.res_filter, config.policy_filter, config.value_filter, config.value_hidden);
    try {
        printf("load model %s\n", basename(config.model_fname));
        torch::load(omega_net, config.model_fname);
    } catch (const c10::Error& e) {
        fprintf(stderr, "error loading the model %s\n", config.model_fname);
        exit(-1);
    }
    omega_net->to(device);
    omega_net->eval();

    position_board = torch::linspace(0., 1., 64).view({8, 8}).to(device);
}

void inference(const input_t *recv_data, output_t *send_data) {
    const auto& config = get_config();

    float *black_board_arr = new float[config.n_thread * 64];  // TODO: ok?
    float *white_board_arr = new float[config.n_thread * 64];
    float *side_arr = new float[config.n_thread];
    float *legal_flags_arr = new float[config.n_thread * 64];
    for (int i = 0; i < config.n_thread; i++) {
        for (int j = 0; j < 64; j++) {
            black_board_arr[i*64+j] = static_cast<float>((recv_data[i].black_board >> j) & 1);
            white_board_arr[i*64+j] = static_cast<float>((recv_data[i].white_board >> j) & 1);
            legal_flags_arr[i*64+j] = static_cast<float>(recv_data[i].legal_flags[j]);
        }
        side_arr[i] = static_cast<float>(recv_data[i].side);
    }

    torch::Tensor black_board_b = torch::from_blob(black_board_arr, {config.n_thread, 8, 8}).to(device);
    torch::Tensor white_board_b = torch::from_blob(white_board_arr, {config.n_thread, 8, 8}).to(device);
    torch::Tensor side_b = torch::from_blob(side_arr, {config.n_thread}).to(device);
    torch::Tensor legal_flags_b = torch::from_blob(legal_flags_arr, {config.n_thread, 64}).to(device);

    // normalize board direction
    torch::Tensor norm_idx;
    std::tie(black_board_b, white_board_b, norm_idx) = normalize_board(black_board_b, white_board_b);
    legal_flags_b = make_variation(legal_flags_b.view({-1, 8, 8}), norm_idx).view({-1, 64});

    torch::Tensor policy_b, value_pred_b;
    {
        torch::NoGradGuard no_grad;
        std::tie(policy_b, value_pred_b) = omega_net->forward(black_board_b, white_board_b, side_b, legal_flags_b);
    }

    policy_b = make_variation(policy_b.view({-1, 8, 8}), flip_idx(norm_idx)).view({-1, 64});

    policy_b = policy_b.to(torch::kCPU);
    value_pred_b = value_pred_b.to(torch::kCPU);

    float *value_pred_arr = (float*)value_pred_b.data_ptr();
    for (int i = 0; i < config.n_thread; i++) {
        memcpy(send_data[i].priors, policy_b[i].data_ptr(), sizeof(float)*64);
        send_data[i].value = value_pred_arr[i];
    }

    delete[] black_board_arr;
    delete[] white_board_arr;
    delete[] side_arr;
}
