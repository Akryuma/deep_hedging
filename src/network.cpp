#include "network.hpp"

DeepHedgingNetImpl::DeepHedgingNetImpl(int64_t input_dim, int64_t hidden_dim, int64_t output_dim) {
    fc1_ = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
    fc2_ = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
    fc3_ = register_module("fc3", torch::nn::Linear(hidden_dim, output_dim));
}

torch::Tensor DeepHedgingNetImpl::forward(torch::Tensor x) {
    x = torch::relu(fc1_->forward(x));
    x = torch::relu(fc2_->forward(x));
    x = torch::sigmoid(fc3_->forward(x));
    return x;
}

torch::Tensor cvar_loss(const torch::Tensor& pnl, float alpha) {
    torch::Tensor losses = - pnl.flatten();
    int64_t tail_len = static_cast<int64_t>((1.0-alpha) * losses.size(0));
    auto top_results = torch::topk(losses, tail_len, /*dim=*/0, /*largest=*/true, /*sorted=*/true);
    return std::get<0>(top_results).mean();
}