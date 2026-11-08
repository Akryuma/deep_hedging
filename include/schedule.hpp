#pragma once
#include <torch/torch.h>
#include <algorithm>
#include <iostream>

inline void reduce_lr_on_plateau(
    double validation_loss,
    torch::optim::Optimizer& optimizer,
    double& best_validation_loss,
    int& plateau_counter,
    int patience = 50,
    double factor = 0.5,
    double min_lr = 1e-6,
    double min_improvement = 1e-4
) {
    if (validation_loss < best_validation_loss - min_improvement) {
        best_validation_loss = validation_loss;
        plateau_counter = 0;
        return;
    }

    ++plateau_counter;
    if (plateau_counter < patience) {
        return;
    }

    for (auto& param_group : optimizer.param_groups()) {
        double current_lr = param_group.options().get_lr();
        double new_lr = std::max(current_lr * factor, min_lr);
        if (new_lr < current_lr) {
            param_group.options().set_lr(new_lr);
            std::cout << "[SCHEDULER] Validation plateau. LR: "
                      << current_lr << " -> " << new_lr << std::endl;
        }
    }
    plateau_counter = 0;
}
