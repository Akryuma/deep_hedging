#include "utils.hpp"
#include "simulation.hpp"
#include <cmath>

HestonSimulation::HestonSimulation(const HestonParameters& params, int num_time_steps, int num_paths, double T, torch::Device device)
: params_(params), num_time_steps_(num_time_steps), num_paths_(num_paths), T_(T) {
    dt = utils::get_dt(num_time_steps_, T_);
    sqrt_dt_ = std::sqrt(dt);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    int half_num_paths = (num_paths_ + 1) / 2;
    
    // Allocation unique de la mémoire
    Z1_half_ = torch::empty({half_num_paths, num_time_steps_}, options);
    Z2_half_ = torch::empty({half_num_paths, num_time_steps_}, options);
    S_ = torch::empty({num_paths_, num_time_steps_ + 1}, options);
    v_ = torch::empty({num_paths_, num_time_steps_ + 1}, options);
}

std::pair<torch::Tensor, torch::Tensor> HestonSimulation::generate_paths() {
    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // Brownians
    // int half_num_paths = (num_paths + 1) / 2;

    // torch::Tensor Z1_half = torch::randn({half_num_paths, num_time_steps_}, options);
    // torch::Tensor Z2_half = torch::randn({half_num_paths, num_time_steps_}, options);
    Z1_half_.normal_(0.0, 1.0);
    Z2_half_.normal_(0.0, 1.0);
    torch::Tensor Z1_anti = -Z1_half_;
    torch::Tensor Z2_anti = -Z2_half_;
    torch::Tensor Z1 = torch::cat({Z1_half_, Z1_anti}, 0);
    torch::Tensor Z2 = torch::cat({Z2_half_, Z2_anti}, 0);
    Z1 = Z1.slice(0, 0, num_paths_);
    Z2 = Z2.slice(0, 0, num_paths_);

    torch::Tensor W1 = sqrt_dt_ * Z1;
    torch::Tensor W2 = sqrt_dt_ * (params_.rho * Z1 + std::sqrt(1 - params_.rho * params_.rho) * Z2);

    // Fill asset prices and variances
    // torch::Tensor S = torch::full({num_paths, num_time_steps_ + 1}, params_.S0, options);
    // torch::Tensor v = torch::full({num_paths, num_time_steps_ + 1}, params_.v0, options);
    S_.fill_(params_.S0);
    v_.fill_(params_.v0);

    for (int i = 0; i < num_time_steps_; ++i) {
        torch::Tensor v_t = v_.select(1, i);
        torch::Tensor S_t = S_.select(1, i);

        torch::Tensor v_next = torch::relu(v_t + params_.kappa * (params_.theta - v_t) * dt + params_.eta * torch::sqrt(v_t) * W2.select(1, i));
        S_.select(1, i + 1) = S_t + params_.r * S_t * dt + torch::sqrt(v_t) * S_t * W1.select(1, i);
        v_.select(1, i + 1) = v_next;
    }

    return {S_, v_};
}