#!/bin/bash

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../build"

RATES=(
    0.01
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
    
    ./deep_hedging_test --cost-rate "$rate" --epochs 5000 "$@"
    
    echo ""
done
