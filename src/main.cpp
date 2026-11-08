#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <format>
#include <limits>
#include <filesystem>
#include "simulation.hpp"
#include "network.hpp"
#include "finance.hpp"
#include "export_utils.hpp"
#include "parser.hpp"
#include "schedule.hpp"

int main(int argc, char* argv[]) {
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "[INFO] Using GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
        torch::globalContext().setBenchmarkCuDNN(true);
    }
    torch::manual_seed(0);

    // Parsing arguments
    ParsedArgs args = parse_arguments(argc, argv);

    // ----- Generate Heston paths -----
    std::cout << "--- Heston Monte Carlo Engine ---" << std::endl;

    HestonParameters params{
        .S0 = 100.0,
        .r = 0., // zero rates
        .kappa = 2.0,
        .theta = 0.04,
        .eta = 1.1,
        .rho = -0.1,
        .v0 = 0.04
    };

    int num_time_steps = 365;
    double T = 1.0;
    int num_paths = args.batch;

    HestonSimulation simulation(params, num_time_steps, num_paths, T, device);    

    double r_dt = params.r * simulation.dt;
    double exp_r_dt = std::exp(r_dt);

    // ----- Market -----
    double cost_rate = args.cost_rate;

    // ----- Payoff -----
    double K = params.S0; // strike price
    torch::Tensor initial_price = compute_bs_price(torch::full({num_paths, 1}, params.S0, device), K, params.r, torch::full({num_paths, 1}, params.v0, device), T, device);

    // ----- Paths -----
    std::filesystem::create_directories("../results/outputs/models");
    std::string best_model_temp_path = std::format("../results/outputs/models/model_best_cost_rate_{}.pt", cost_rate);
    std::string ref_model_path = "../results/outputs/models/model_best_cost_rate_0.pt";

    // ----- Evaluation paths -----
    int eval_num_paths = num_paths * 10;
    HestonSimulation simulation_eval(params, num_time_steps, eval_num_paths, T, device);
    auto [S_eval, V_eval] = simulation_eval.generate_paths();
    auto log_S_eval = torch::log(S_eval);
    auto underlying_change_eval = S_eval.slice(1, 1, num_time_steps + 1) - S_eval.slice(1, 0, num_time_steps)* exp_r_dt;
    auto sqrt_V_eval = torch::sqrt(V_eval);
    torch::Tensor payoff_eval = european_option_payoff(S_eval.slice(1, num_time_steps, num_time_steps + 1), K);
    torch::Tensor initial_price_eval = compute_bs_price(torch::full({eval_num_paths, 1}, params.S0, device), K, params.r, torch::full({eval_num_paths, 1}, params.v0, device), T, device);

    // ================
    // TRAINING LOOP
    // ================
    std::cout << "--- Training Deep Hedging Network ---" << std::endl;
    double learning_rate = args.lr;
    int64_t kNumberOfEpochs = args.epochs;
    float alpha = 0.95;

    DeepHedgingNet model(4, 32, 1);
    model->to(device);

    // Transfer learning
    if (cost_rate > 0.0) {
        torch::load(model, ref_model_path);
        torch::save(model, best_model_temp_path);
        learning_rate = 1e-4;
    }
    
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
    double best_validation_loss = std::numeric_limits<double>::infinity();
    int plateau_counter = 0;

    double best_loss = std::numeric_limits<double>::infinity();

    std::vector<double> loss_history;
    torch::Tensor pnl;

    for (int64_t epoch = 0; epoch < kNumberOfEpochs; ++epoch) {
        optimizer.zero_grad();
        // Generation of new paths to avoid overfitting
        auto [S, V] = simulation.generate_paths();
        auto log_S = torch::log(S);
        auto underlying_change = S.slice(1, 1, num_time_steps+1) - S.slice(1, 0, num_time_steps) * exp_r_dt;
        auto sqrt_V = torch::sqrt(V);

        // Time evolution
        torch::Tensor payoff = european_option_payoff(S.slice(1, num_time_steps, num_time_steps + 1), K);

        torch::Tensor portfolio = initial_price.clone();
        torch::Tensor held_position = torch::zeros({num_paths, 1}, device); // position before rebalancing

        for (int64_t t = 0; t <= num_time_steps; ++t) {
            // At t = 0, there is no previous interval to accrue.
            if (t > 0) {
                portfolio = portfolio * exp_r_dt + held_position * underlying_change.slice(1, t-1, t);
            }

            torch::Tensor target_position;
            if (t < num_time_steps) {
                torch::Tensor input = torch::cat({log_S.slice(1, t, t+1),
                                        sqrt_V.slice(1, t, t+1),
                                        torch::full({num_paths, 1},T - (t * simulation.dt), device),
                                        held_position}, 1);
                target_position = model->forward(input);
            } else {
                target_position = torch::zeros({num_paths, 1}, device); // liquidation at t = N
            }
            torch::Tensor transaction_cost = cost_rate * torch::abs(target_position - held_position) * S.slice(1, t, t+1);

            portfolio = portfolio - transaction_cost;
            held_position = target_position;
        }

        pnl = portfolio - payoff;
        torch::Tensor train_loss = cvar_loss(pnl, alpha);
        train_loss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        optimizer.step();

        // Evaluation
        double current_loss;
        {
            torch::NoGradGuard no_grad;
            torch::Tensor portfolio_eval = initial_price_eval.clone();
            torch::Tensor held_position_eval = torch::zeros({eval_num_paths, 1}, device);

            for (int64_t t = 0; t <= num_time_steps; ++t) {
                if (t > 0) {
                    portfolio_eval = portfolio_eval * exp_r_dt + held_position_eval * underlying_change_eval.slice(1, t-1, t);
                }

                torch::Tensor target_position_eval;
                if (t < num_time_steps) {
                    torch::Tensor input = torch::cat({log_S_eval.slice(1, t, t+1),
                                            sqrt_V_eval.slice(1, t, t+1),
                                            torch::full({eval_num_paths, 1}, T - (t * simulation.dt), device),
                                            held_position_eval}, 1);
                    target_position_eval = model->forward(input);
                } else {
                    target_position_eval = torch::zeros({eval_num_paths, 1}, device);
                }
                torch::Tensor transaction_cost = cost_rate * torch::abs(target_position_eval - held_position_eval) * S_eval.slice(1, t, t+1);

                portfolio_eval = portfolio_eval - transaction_cost;
                held_position_eval = target_position_eval;
            }

            torch::Tensor pnl_eval = portfolio_eval - payoff_eval;
            current_loss = cvar_loss(pnl_eval, alpha).item<double>();
        }

        reduce_lr_on_plateau(current_loss, optimizer, best_validation_loss, plateau_counter);

        if (current_loss < best_loss) {
            best_loss = current_loss;
            torch::save(model, best_model_temp_path); 
        }

        // History
        loss_history.push_back(current_loss);
        if (epoch % 100 == 0 || epoch == kNumberOfEpochs - 1) {
            std::cout << "Epoch [" << epoch + 1 << "/" << kNumberOfEpochs 
                      << "], Loss: " << current_loss 
                      << ", LR: "<< optimizer.param_groups()[0].options().get_lr() 
                      << std::endl;
        }
    }

    std::filesystem::create_directories("../results/outputs/losses");
    export_csv::export_loss_to_csv(loss_history, std::format("../results/outputs/losses/loss_history_cost_rate_{}.csv", cost_rate));

    return 0;
}
