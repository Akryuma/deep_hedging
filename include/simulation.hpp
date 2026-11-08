#pragma once
#include <torch/torch.h>
#include <utility>

struct HestonParameters {
    double S0;     // Initial asset price
    double r;      // Drift rate
    double kappa;  // Mean reversion speed
    double theta;  // Long-term variance
    double eta;  // Volatility of variance
    double rho;    // Correlation between asset and variance
    double v0;     // Initial variance
};

class HestonSimulation {
public:
    HestonSimulation(const HestonParameters& params, int num_time_steps, int num_paths, double T, torch::Device device);

    std::pair<torch::Tensor, torch::Tensor> generate_paths();

    double dt;
private:
    HestonParameters params_;
    int num_time_steps_;
    int num_paths_;
    double T_;
    double sqrt_dt_;
    torch::Tensor Z1_half_;
    torch::Tensor Z2_half_;
    torch::Tensor S_;
    torch::Tensor v_;
};     