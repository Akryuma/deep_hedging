#pragma once
#include <torch/torch.h>
#include <utility>
#include <cmath>

torch::Tensor european_option_payoff(torch::Tensor S, double K) {
    return torch::relu(S - K);
}
torch::Tensor compute_bs_price(torch::Tensor S, double K, double r, torch::Tensor V, double T, torch::Device device) {
    torch::Tensor sqrt_T = torch::sqrt(torch::full({1}, T, device));
    torch::Tensor d1 = (torch::log(S / K) + (r + 0.5 * V) * T) / (torch::sqrt(V) * sqrt_T);
    torch::Tensor d2 = d1 - torch::sqrt(V) * sqrt_T;
    
    // N(x) = 0.5 * (1.0 + erf(x / sqrt(2)))
    torch::Tensor Nd1 = 0.5 * (1.0 + torch::erf(d1 / std::sqrt(2.0)));
    torch::Tensor Nd2 = 0.5 * (1.0 + torch::erf(d2 / std::sqrt(2.0)));
    
    torch::Tensor call_price = S * Nd1 - K * std::exp(-r * T) * Nd2;
    
    return call_price;
}

torch::Tensor compute_bs_delta(torch::Tensor S, double K, double r, torch::Tensor V, torch::Tensor tau) {
    torch::Tensor tau_min = torch::clamp_min(tau, 1e-7);
    torch::Tensor log_forward_moneyness = torch::log(S / K) + r * tau_min;
    // Avoid division by zero; the zero-variance case is handled separately below.
    torch::Tensor safe_variance = torch::where(V == 0, torch::ones_like(V), V);
    torch::Tensor d1 = (log_forward_moneyness + 0.5 * V * tau_min) / (torch::sqrt(safe_variance) * torch::sqrt(tau_min));
    torch::Tensor delta = 0.5 * (1.0 + torch::erf(d1 / M_SQRT2));
    torch::Tensor zero_variance_delta = torch::where(log_forward_moneyness > 0, torch::ones_like(S),
                                        torch::where(log_forward_moneyness < 0, torch::zeros_like(S), torch::full_like(S, 0.5)));
    return torch::where(V == 0, zero_variance_delta, delta);
}
