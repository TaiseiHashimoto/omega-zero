#include <torch/torch.h>
#include <tuple>

#include "server.hpp"


struct ResBlockImpl : torch::nn::Module {
    ResBlockImpl(int n_filter);

    torch::Tensor forward(const torch::Tensor& x);

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
};

TORCH_MODULE(ResBlock);

struct OmegaNetImpl : torch::nn::Module {
    OmegaNetImpl(int board_size, int n_action, int n_res_block, int res_filter, int head_filter, int value_hidden);

    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& black_board, const torch::Tensor& white_board, const torch::Tensor& side, const torch::Tensor& legal_flags);

    torch::nn::Sequential conv{nullptr};
    torch::nn::Sequential res_blocks{nullptr};
    torch::nn::Sequential policy_head{nullptr};
    torch::nn::Sequential value_head{nullptr};
};

TORCH_MODULE(OmegaNet);

void init_model();
void inference(const input_t *recv_data, output_t *send_data);
