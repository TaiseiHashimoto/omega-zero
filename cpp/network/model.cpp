#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <experimental/filesystem>

#include "model.hpp"
#include "server.hpp"
#include "config.hpp"


// /**
//     DO NOT use assignment operations because they can cause unintended change of parameters
// **/

ResBlockImpl::ResBlockImpl(int n_filter) {
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(n_filter, n_filter, 3).padding(1)
    ));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(n_filter));
    conv2 = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(n_filter, n_filter, 3).padding(1)
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


OmegaNetImpl::OmegaNetImpl(int board_size, int n_action, int n_res_block, int res_filter, int head_filter, int value_hidden) {
    // input channel : black / white / side
    conv = register_module("conv", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, res_filter, 3).padding(1)),
        torch::nn::BatchNorm2d(res_filter),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
    ));

    res_blocks = register_module("res_blocks", torch::nn::Sequential());
    for (int i = 0; i < n_res_block; i++) {
        res_blocks->push_back(ResBlock(res_filter));
    }

    policy_head = register_module("policy_head", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(res_filter, head_filter, 1).padding(0)),
        torch::nn::BatchNorm2d(head_filter),
        torch::nn::Flatten(),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(head_filter * board_size * board_size, n_action)
    ));

    value_head = register_module("value_head", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(res_filter, head_filter, 1).padding(0)),
        torch::nn::BatchNorm2d(head_filter),
        torch::nn::Flatten(),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(head_filter * board_size * board_size, value_hidden),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(value_hidden, 1)
    ));
}

std::tuple<torch::Tensor, torch::Tensor> OmegaNetImpl::forward(const torch::Tensor& black_board, const torch::Tensor& white_board, const torch::Tensor& side, const torch::Tensor& legal_flags) {
    torch::Tensor side_board = torch::ones_like(black_board) * side.view({-1, 1, 1});
    torch::Tensor x = torch::stack({black_board, white_board, side_board}, 1);
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

int extract_generation(std::string model_fname) {
    int start = -1;
    // skip .pt (length: 3)
    int end = model_fname.length() - 4;
    for (int i = end; i >= 0; i--) {
        if (model_fname[i] == '_') {
            start = i + 1;
            break;
        }
    }
    assert(start >= 0);
    std::string generation_str = model_fname.substr(start, end - start + 1);
    return std::stoi(generation_str);
}

} // namespace

void init_model() {
    const auto& config = get_config();

    if (torch::cuda::is_available() && config.device_id >= 0) {
        device = {torch::kCUDA, (short int)config.device_id};
    }
    std::cout << "MODEL  using " << device << std::endl;

    omega_net = OmegaNet(config.board_size, config.n_action, config.n_res_block, config.res_filter, config.head_filter, config.value_hidden);
    omega_net->eval();
}

int load_model(int current_generation) {
    const auto& config = get_config();

    char model_fname[200];
    int new_generation = -1;

    if (config.generation >= 0) {
        // generation is spacified
        new_generation = config.generation;
        sprintf(model_fname, "%s/model_jit_%d.pt", config.model_dname, config.generation);
    } else {
        // find the newest model
        std::string model_fname_str;
        for (const auto & entry : std::experimental::filesystem::directory_iterator(config.model_dname)) {
            std::string entry_path = entry.path().string();
            if (entry_path.find("_jit_") == std::string::npos) {
                continue;
            }
            int generation = extract_generation(entry_path);
            if (generation > new_generation) {
                new_generation = generation;
                model_fname_str = entry.path().string();
            }
        }

        if (new_generation > current_generation) {  // new model found
            strcpy(model_fname, model_fname_str.c_str());
        } else {
            std::cout << "MODEL  new model not created yet" << std::endl;
            return current_generation;
        }
    }

    try {
        printf("MODEL  load model %s\n", basename(model_fname));
        torch::load(omega_net, model_fname);
    } catch (const c10::Error& e) {
        fprintf(stderr, "MODEL  error loading the model\n");
        exit(-1);
    }

    omega_net->to(device);
    return new_generation;
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

    torch::Tensor policy_b, value_pred_b;
    {
        torch::NoGradGuard no_grad;
        std::tie(policy_b, value_pred_b) = omega_net->forward(black_board_b, white_board_b, side_b, legal_flags_b);
    }

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
