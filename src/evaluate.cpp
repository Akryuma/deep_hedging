#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <format>
#include <limits>
#include <filesystem>
#include <algorithm>
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
    torch::manual_seed(1); // different from the training

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
    std::string best_model_temp_path = std::format("../results/outputs/models/model_best_cost_rate_{}.pt", cost_rate);
    std::string ref_model_path = "../results/outputs/models/model_best_cost_rate_0.pt";


    // ----- Evaluation paths -----
    int eval_num_paths = num_paths * 10;
    HestonSimulation simulation_eval(params, num_time_steps, eval_num_paths, T, device);
    torch::Tensor initial_price_eval = compute_bs_price(torch::full({eval_num_paths, 1}, params.S0, device), K, params.r, torch::full({eval_num_paths, 1}, params.v0, device), T, device);

    float alpha = 0.95;

    DeepHedgingNet model(4, 32, 1);
    model->to(device);
    torch::load(model, best_model_temp_path);

    // ===============================================================
    // BENCHMARK BLACK SCHOLES, REFERENCE (COST RATE = 0) & EVALUATION
    // ===============================================================
    {
        torch::NoGradGuard no_grad; // No gradient tracking anymore
        std::filesystem::create_directories("../results/outputs/pnl");
        std::filesystem::create_directories("../results/outputs/metrics");
        std::filesystem::create_directories("../results/outputs/deltas");
        int pricing_num_paths = eval_num_paths;
        // New test paths, generated after selecting the best model.
        auto [S_final, V_final] = simulation_eval.generate_paths();
        auto& S = S_final;
        auto& V = V_final;
        auto log_S = torch::log(S_final);
        auto underlying_change = S_final.slice(1, 1, num_time_steps + 1) - S_final.slice(1, 0, num_time_steps) * exp_r_dt;
        auto sqrt_V = torch::sqrt(V_final);
        torch::Tensor payoff = european_option_payoff(S_final.slice(1, num_time_steps, num_time_steps + 1), K);
        auto& initial_price = initial_price_eval;
        HestonSimulation& simulation_pricing = simulation_eval;

        // Reference model
        DeepHedgingNet model_ref(4, 32, 1);
        model_ref->to(device);

        if (cost_rate > 0.0) {
            try {
                torch::load(model_ref, ref_model_path);
            } catch (const c10::Error& e) {
                std::cerr << "[ERROR] Impossible to load model from" << ref_model_path 
                          << ". Need to train model with cost_rate = 0.0 first." << std::endl;
                return -1;
            }
        } else {
            model_ref = model;
        }

        torch::Tensor portfolio_nn = initial_price.clone();
        torch::Tensor portfolio_nn_raw = initial_price.clone();
        torch::Tensor held_position_nn = torch::zeros({pricing_num_paths, 1}, device);

        torch::Tensor portfolio_ref = initial_price.clone();
        torch::Tensor portfolio_ref_raw = initial_price.clone();
        torch::Tensor held_position_ref = torch::zeros({pricing_num_paths, 1}, device);
        
        torch::Tensor portfolio_bs = initial_price.clone();
        torch::Tensor portfolio_bs_raw = initial_price.clone();
        torch::Tensor held_position_bs = torch::zeros({pricing_num_paths, 1}, device);

        torch::Tensor turnover_bs = torch::zeros({pricing_num_paths, 1}, device);
        torch::Tensor turnover_nn = torch::zeros({pricing_num_paths, 1}, device);
        torch::Tensor turnover_ref = torch::zeros({pricing_num_paths, 1}, device);
        torch::Tensor total_cost_bs = torch::zeros({pricing_num_paths, 1}, device);
        torch::Tensor total_cost_nn = torch::zeros({pricing_num_paths, 1}, device);
        torch::Tensor total_cost_ref = torch::zeros({pricing_num_paths, 1}, device);
        
        
        std::ofstream path_log(std::format("../results/outputs/deltas/single_path_deltas_cost_rate_{}.csv", cost_rate));
        path_log << "TimeStep,Price,Delta_BS,Transaction_Cost_BS,Delta_NN,Transaction_Cost_NN,Delta_Ref,Transaction_Cost_Ref\n";

        int logged_num_paths = std::min(1000, pricing_num_paths);
        std::ofstream deltas_log(std::format("../results/outputs/deltas/sample_path_deltas_cost_rate_{}.csv", cost_rate));
        deltas_log << "Path,TimeStep,Price,Variance,Delta_BS,Delta_Ref,Delta_NN\n";
        deltas_log << std::setprecision(9);

        for (int64_t t = 0; t <= num_time_steps; ++t) {
            double time_to_mat = T - (t * simulation_pricing.dt);

            if (t > 0) {
                portfolio_bs = portfolio_bs * exp_r_dt + held_position_bs * underlying_change.slice(1, t-1, t);
                portfolio_bs_raw = portfolio_bs_raw * exp_r_dt + held_position_bs * underlying_change.slice(1, t-1, t);
                portfolio_nn = portfolio_nn * exp_r_dt + held_position_nn * underlying_change.slice(1, t-1, t);
                portfolio_nn_raw = portfolio_nn_raw * exp_r_dt + held_position_nn * underlying_change.slice(1, t-1, t);
                portfolio_ref = portfolio_ref * exp_r_dt + held_position_ref * underlying_change.slice(1, t-1, t);
                portfolio_ref_raw = portfolio_ref_raw * exp_r_dt + held_position_ref * underlying_change.slice(1, t-1, t);
            }

            // 1. Black Scholes
            torch::Tensor target_position_bs;
            if (t < num_time_steps) {
                torch::Tensor tau = torch::full({pricing_num_paths, 1}, time_to_mat, torch::TensorOptions().device(device));
                target_position_bs = compute_bs_delta(S.slice(1, t, t+1), K, params.r, V.slice(1, t, t+1), tau);
            } else {
                target_position_bs = torch::zeros({pricing_num_paths, 1}, device);
            }
            torch::Tensor trade_size_bs = torch::abs(target_position_bs - held_position_bs);
            torch::Tensor transaction_cost_bs = cost_rate * trade_size_bs * S.slice(1, t, t+1);
            turnover_bs += trade_size_bs;
            total_cost_bs += transaction_cost_bs;

            portfolio_bs = portfolio_bs - transaction_cost_bs;
            held_position_bs = target_position_bs;

            // 2. Final model
            torch::Tensor target_position_nn;
            if (t < num_time_steps) {
                torch::Tensor input = torch::cat({
                    log_S.slice(1, t, t+1),
                    sqrt_V.slice(1, t, t+1),
                    torch::full({pricing_num_paths, 1}, time_to_mat, device),
                    held_position_nn
                }, 1);
                target_position_nn = model->forward(input);
            } else {
                target_position_nn = torch::zeros({pricing_num_paths, 1}, device);
            }
            torch::Tensor trade_size_nn = torch::abs(target_position_nn - held_position_nn);
            torch::Tensor transaction_cost_nn = cost_rate * trade_size_nn * S.slice(1, t, t+1);
            turnover_nn += trade_size_nn;
            total_cost_nn += transaction_cost_nn;

            portfolio_nn = portfolio_nn - transaction_cost_nn;
            held_position_nn = target_position_nn;

            // 3. Ref model
            torch::Tensor target_position_ref;
            if (t < num_time_steps) {
                torch::Tensor input_ref = torch::cat({
                    log_S.slice(1, t, t+1),
                    sqrt_V.slice(1, t, t+1),
                    torch::full({pricing_num_paths, 1}, time_to_mat, device),
                    held_position_ref
                }, 1);
                target_position_ref = model_ref->forward(input_ref);
            } else {
                target_position_ref = torch::zeros({pricing_num_paths, 1}, device);
            }
            torch::Tensor trade_size_ref = torch::abs(target_position_ref - held_position_ref);
            torch::Tensor transaction_cost_ref = cost_rate * trade_size_ref * S.slice(1, t, t+1);
            turnover_ref += trade_size_ref;
            total_cost_ref += transaction_cost_ref;

            portfolio_ref = portfolio_ref - transaction_cost_ref;
            held_position_ref = target_position_ref;

            // Logging
            torch::Tensor sample_deltas = torch::cat({
                S.slice(0, 0, logged_num_paths).slice(1, t, t+1),
                V.slice(0, 0, logged_num_paths).slice(1, t, t+1),
                target_position_bs.slice(0, 0, logged_num_paths),
                target_position_ref.slice(0, 0, logged_num_paths),
                target_position_nn.slice(0, 0, logged_num_paths)
            }, 1).to(torch::kCPU).contiguous();
            auto deltas = sample_deltas.accessor<float, 2>();
            for (int path = 0; path < logged_num_paths; ++path) {
                deltas_log << path << "," << t;
                for (int column = 0; column < 5; ++column) {
                    deltas_log << "," << deltas[path][column];
                }
                deltas_log << "\n";
            }

            float s_val = S.slice(1, t, t+1)[0][0].item<float>();
            float delta_bs_val = target_position_bs[0][0].item<float>();
            float tc_bs_val = transaction_cost_bs[0][0].item<float>();
            float delta_nn_val = target_position_nn[0][0].item<float>();
            float tc_nn_val = transaction_cost_nn[0][0].item<float>();
            float delta_ref_val = target_position_ref[0][0].item<float>();
            float tc_ref_val = transaction_cost_ref[0][0].item<float>();
            path_log << t << "," << s_val << "," 
                     << delta_bs_val << "," << tc_bs_val << "," 
                     << delta_nn_val << "," << tc_nn_val << ","
                     << delta_ref_val << "," << tc_ref_val << "\n";
        }
        path_log.close();
        deltas_log.close();

        // One row per test path, including the initial trade and final liquidation.
        torch::Tensor path_metrics = torch::cat({turnover_bs, turnover_nn, turnover_ref,
                                                total_cost_bs, total_cost_nn, total_cost_ref}, 1).to(torch::kCPU).contiguous();
        auto metrics = path_metrics.accessor<float, 2>();
        std::string metrics_path = std::format("../results/outputs/metrics/final_path_metrics_cost_rate_{}.csv", cost_rate);
        std::ofstream metrics_log(metrics_path);
        metrics_log << "Path,Turnover_BS,Turnover_NN,Turnover_Ref,Total_Cost_BS,Total_Cost_NN,Total_Cost_Ref\n";
        metrics_log << std::setprecision(9);
        for (int64_t path = 0; path < pricing_num_paths; ++path) {
            metrics_log << path;
            for (int column = 0; column < 6; ++column) {
                metrics_log << "," << metrics[path][column];
            }
            metrics_log << "\n";
        }
        metrics_log.close();
        std::cout << "[INFO] Exported path metrics to " << metrics_path << std::endl;

        torch::Tensor means = path_metrics.mean(0);
        torch::Tensor quantiles = torch::quantile(path_metrics, torch::tensor({0.05f, 0.5f, 0.95f}), 0);
        std::vector<std::string> metric_names = {"Turnover_BS", "Turnover_NN", "Turnover_Ref",
                                                 "Total_Cost_BS", "Total_Cost_NN", "Total_Cost_Ref"};
        std::ofstream summary_log(std::format("../results/outputs/metrics/final_path_metrics_summary_cost_rate_{}.csv", cost_rate));
        summary_log << "Metric,Mean,P05,Median,P95\n";
        summary_log << std::setprecision(9);
        for (int column = 0; column < 6; ++column) {
            summary_log << metric_names[column] << "," << means[column].item<float>()
                        << "," << quantiles[0][column].item<float>()
                        << "," << quantiles[1][column].item<float>()
                        << "," << quantiles[2][column].item<float>() << "\n";
        }
        summary_log.close();

        torch::Tensor pnl_nn = portfolio_nn - payoff;
        torch::Tensor pnl_nn_raw = portfolio_nn_raw - payoff;
        torch::Tensor pnl_bs = portfolio_bs - payoff;
        torch::Tensor pnl_bs_raw = portfolio_bs_raw - payoff;
        torch::Tensor pnl_ref = portfolio_ref - payoff;
        torch::Tensor pnl_ref_raw = portfolio_ref_raw - payoff;

        torch::Tensor loss_bs = cvar_loss(pnl_bs, alpha);
        torch::Tensor loss_nn = cvar_loss(pnl_nn, alpha);
        torch::Tensor loss_ref = cvar_loss(pnl_ref, alpha);
        
        std::cout << "DeepHedgingNet CVaR Loss: " << loss_nn.item<double>() << std::endl;
        std::cout << "BS CVaR Loss: " << loss_bs.item<double>() << std::endl;
        std::cout << "Ref (trained on cost_rate=0) CVaR Loss: " << loss_ref.item<double>() << std::endl;


        export_csv::export_pnl_to_csv(pnl_bs, std::format("../results/outputs/pnl/final_pnl_bs_cost_rate_{}.csv", cost_rate));
        export_csv::export_pnl_to_csv(pnl_nn, std::format("../results/outputs/pnl/final_pnl_nn_cost_rate_{}.csv", cost_rate));
        export_csv::export_pnl_to_csv(pnl_ref, std::format("../results/outputs/pnl/final_pnl_ref_cost_rate_{}.csv", cost_rate));
        export_csv::export_pnl_to_csv(pnl_nn_raw, std::format("../results/outputs/pnl/final_pnl_nn_raw_cost_rate_{}.csv", cost_rate));
        export_csv::export_pnl_to_csv(pnl_bs_raw, std::format("../results/outputs/pnl/final_pnl_bs_raw_cost_rate_{}.csv", cost_rate));
        export_csv::export_pnl_to_csv(pnl_ref_raw, std::format("../results/outputs/pnl/final_pnl_ref_raw_cost_rate_{}.csv", cost_rate));

    }
    return 0;
}
