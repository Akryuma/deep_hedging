#!/bin/bash

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../build"

RATES=(
    0
    0.01
)

for rate in "${RATES[@]}"; do
    echo "==========================================="
    echo " Evaluating with transaction cost rate = $rate"
    echo "==========================================="

    ./deep_hedging_eval --cost-rate "$rate"

    echo ""
done
