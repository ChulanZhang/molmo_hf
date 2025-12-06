#!/bin/bash
# Plot script for Exp1, Exp2, Exp3 Accuracy Results
# Visualizes accuracy vs max_crops, top_k, and num_active_blocks
# Also includes exp3 sensitivity results (importance scores and pruning accuracy)

set -e

# Default values
RESULTS_DIR="${RESULTS_DIR:-./results/profiling}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_DIR}}"

echo "=========================================="
echo "Plotting Exp1, Exp2, Exp3 Accuracy Results"
echo "=========================================="
echo "Results directory: ${RESULTS_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Run the plotting script
python3 experiments/profiling/plot_exp1_exp2_exp3_accuracy.py \
    --results_dir "${RESULTS_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "=========================================="
echo "Plotting complete!"
echo "Figures saved to: ${OUTPUT_DIR}/figures"
echo "=========================================="


