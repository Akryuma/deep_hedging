#pragma once
#include <torch/torch.h>
#include <utility>

class DeepHedgingNetImpl: public torch::nn::Module {
public:
    DeepHedgingNetImpl(int64_t input_dim, int64_t hidden_dim, int64_t output_dim);

    torch::Tensor forward(torch::Tensor x);

private:
    // Place holders for the linear layers
    torch::nn::Linear fc1_{nullptr};
    torch::nn::Linear fc2_{nullptr};
    torch::nn::Linear fc3_{nullptr};
};


TORCH_MODULE(DeepHedgingNet);

torch::Tensor cvar_loss(const torch::Tensor& pnl, float alpha = 0.95);