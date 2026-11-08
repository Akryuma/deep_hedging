#!/bin/bash

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../build"

RATES=(
    0
    0.001
    0.0025
    0.005
    0.0075
    0.01
    0.015
    0.02
    0.025
    0.03
)
# RATES=(
#     0
#     0.0025
#     0.005
#     0.01
# )

for rate in "${RATES[@]}"; do
    echo "==========================================="
    echo " Running with transaction cost rate = $rate"
    echo "==========================================="
    
    ./deep_hedging_test --cost-rate "$rate"
    
    echo ""
done
