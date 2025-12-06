#!/bin/bash
# Script to plot accuracy vs latency Pareto frontier for all latency types

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

# Plot all three latency types: total, prefill, decode
python3 experiments/profiling/utils/plot_accuracy_latency_pareto.py \
    --exp5_dir ./results/profiling/exp5_accuracy \
    --exp6_file ./results/profiling/exp6_latency/exp6_latency_results.json \
    --output_dir ./results/profiling \
    --latency_metric mean \
    --dpi 300

