#pragma once
#include <utility>

namespace utils {
    inline double get_dt(int num_time_steps, double T) {
        if (T <= 1.0) {
            return T / num_time_steps;
        } else {
            return 1.0 / num_time_steps;
        }
    }
}