#pragma once
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <vector>

namespace export_csv {
    void export_loss_to_csv(const std::vector<double>& losses, const std::string& filename) {
        std::ofstream file(filename);
        file << "Epoch,Loss\n";
        for(size_t i = 0; i < losses.size(); ++i) {
            file << (i + 1) << "," << losses[i] << "\n";
        }
        file.close();
        std::cout << "[INFO] Exported loss to " << filename << std::endl;
    }

    void export_pnl_to_csv(torch::Tensor pnl, const std::string& filename) {
        std::ofstream file(filename);
        file << "PNL\n";
        
        torch::Tensor pnl_cpu = pnl.to(torch::kCPU).to(torch::kFloat32).contiguous();
        const float* pnl_data = pnl_cpu.data_ptr<float>();
        
        for(int64_t i = 0; i < pnl_cpu.numel(); ++i) {
            file << pnl_data[i] << "\n";
        }
        file.close();
        std::cout << "[INFO] Exported pnl to " << filename << std::endl;
    }
}