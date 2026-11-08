#pragma once
#include <string>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

struct ParsedArgs {
    int epochs = 1000;
    int batch = 50000;          
    double lr = 0.001;        
    double cost_rate = 0.01; 
};
inline ParsedArgs parse_arguments(int argc, char* argv[]) {
    ParsedArgs result;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ./deep_hedging_test [--epochs <val>] [--batch <val>] [--lr <val>] [--cost-rate <val>]\n";
            std::exit(EXIT_SUCCESS);
        }

        try {
            if (arg == "--epochs" && i + 1 < argc) {
                result.epochs = std::stoi(argv[++i]);
            } 
            else if (arg == "--batch" && i + 1 < argc) {
                result.batch = std::stoi(argv[++i]);
            } 
            else if (arg == "--lr" && i + 1 < argc) {
                result.lr = std::stod(argv[++i]);
            } 
            else if (arg == "--cost-rate" && i + 1 < argc) {
                result.cost_rate = std::stod(argv[++i]);
            }
            else {
                throw std::invalid_argument("Unknown argument, or missing value for: " + arg + "\n");
            }
        } 
        catch (const std::invalid_argument& e) {
            std::cerr << "Critical error: " << e.what() << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

    std::cout << "==========================================\n";
    std::cout << "[INFO] Training configuration :\n";
    std::cout << "       Epochs      = " << result.epochs << "\n";
    std::cout << "       Batch Size  = " << result.batch << "\n";
    std::cout << "       Learn. Rate = " << result.lr << "\n";
    std::cout << "       Cost Rate   = " << result.cost_rate << "\n";
    std::cout << "==========================================\n";

    return result;
}