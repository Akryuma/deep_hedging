#pragma once
#include <torch/torch.h>
#include <iostream>
#include <limits>
#include <algorithm>

class EmaReduceLROnPlateau {
private:
    double ema_loss;
    double best_ema_loss;
    int patience_counter;
    int patience_limit;
    double lr_factor;
    double min_lr;
    double alpha;

public:
    // Constructeur avec des paramètres par défaut calibrés pour ton problème
    EmaReduceLROnPlateau(int patience = 50, double factor = 0.5, double min_learning_rate = 1e-6, double smoothing_alpha = 0.1)
        : ema_loss(-1.0), 
          best_ema_loss(std::numeric_limits<double>::infinity()), 
          patience_counter(0), 
          patience_limit(patience), 
          lr_factor(factor), 
          min_lr(min_learning_rate), 
          alpha(smoothing_alpha) {}

    // Méthode principale à appeler à chaque fin d'epoch
    void step(double current_loss, torch::optim::Optimizer& optimizer) {
        // 1. Mise à jour de l'EMA
        if (ema_loss < 0.0) {
            ema_loss = current_loss; 
        } else {
            ema_loss = alpha * current_loss + (1.0 - alpha) * ema_loss;
        }

        // 2. Détection de l'amélioration
        if (ema_loss < best_ema_loss - 1e-4) {
            best_ema_loss = ema_loss;
            patience_counter = 0; // Reset de la patience si amélioration
        } else {
            patience_counter++;
        }

        // 3. Réduction du Learning Rate si plateau atteint
        if (patience_counter >= patience_limit) {
            for (auto& param_group : optimizer.param_groups()) {
                double current_lr = param_group.options().get_lr();
                double new_lr = std::max(current_lr * lr_factor, min_lr);
                
                if (new_lr < current_lr) {
                    param_group.options().set_lr(new_lr);
                    std::cout << "[SCHEDULER] Plateau atteint (EMA = " << ema_loss 
                              << "). Reduction du LR à : " << new_lr << std::endl;
                }
            }
            patience_counter = 0; // On reset pour laisser le temps au nouveau LR d'agir
        }
    }

    // Accesseur pour logger l'EMA dans le main
    double get_ema_loss() const {
        return ema_loss;
    }
};