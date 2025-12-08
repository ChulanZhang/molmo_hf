#!/bin/bash
# Plot All Profiling Experiments
# Usage: bash experiments/profiling/plot_all_experiments.sh [--results_dir DIR]

RESULTS_DIR="./results/profiling"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash experiments/profiling/plot_all_experiments.sh [--results_dir DIR]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Plotting All Profiling Experiments"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo "=========================================="
echo ""

# Experiment 1: Context Scaling (Vision Tokens)
if [ -f "$RESULTS_DIR/context_scaling/exp1_context_scaling_results.json" ]; then
    echo ">>> Plotting Exp 1: Context Scaling (Vision Tokens)..."
    python experiments/profiling/plots/plot_exp1_context_scaling.py \
        --json_file "$RESULTS_DIR/context_scaling/exp1_context_scaling_results.json"
    if [ $? -eq 0 ]; then
        echo "Exp 1 plotting completed!"
    else
        echo "Exp 1 plotting failed!"
    fi
    echo ""
else
    echo ">>> Skipping Exp 1: Results file not found"
    echo ""
fi

# Experiment 2: MoE Top-K
if [ -f "$RESULTS_DIR/moe_topk/exp2_moe_topk_results.json" ]; then
    echo ">>> Plotting Exp 2: MoE Top-K..."
    python experiments/profiling/plots/plot_exp2_moe_topk.py \
        --json_file "$RESULTS_DIR/moe_topk/exp2_moe_topk_results.json"
    if [ $? -eq 0 ]; then
        echo "Exp 2 plotting completed!"
    else
        echo "Exp 2 plotting failed!"
    fi
    echo ""
else
    echo ">>> Skipping Exp 2: Results file not found"
    echo ""
fi

# Experiment 3: Transformer Blocks Mask
if [ -f "$RESULTS_DIR/transformer_blocks_mask/exp3_transformer_blocks_mask_results.json" ]; then
    echo ">>> Plotting Exp 3: Transformer Blocks Mask..."
    python experiments/profiling/plots/plot_exp3_transformer_blocks_mask.py \
        --json_file "$RESULTS_DIR/transformer_blocks_mask/exp3_transformer_blocks_mask_results.json"
    if [ $? -eq 0 ]; then
        echo "Exp 3 plotting completed!"
    else
        echo "Exp 3 plotting failed!"
    fi
    echo ""
else
    echo ">>> Skipping Exp 3: Results file not found"
    echo ""
fi

# Experiment 4: Output Tokens Scaling
if [ -f "$RESULTS_DIR/output_tokens/exp4_output_tokens_scaling_results.json" ]; then
    echo ">>> Plotting Exp 4: Output Tokens Scaling..."
    python experiments/profiling/plots/plot_exp4_output_tokens.py \
        --json_file "$RESULTS_DIR/output_tokens/exp4_output_tokens_scaling_results.json"
    if [ $? -eq 0 ]; then
        echo "Exp 4 plotting completed!"
    else
        echo "Exp 4 plotting failed!"
    fi
    echo ""
else
    echo ">>> Skipping Exp 4: Results file not found"
    echo ""
fi

echo "=========================================="
echo "All Plotting Completed!"
echo "=========================================="
echo "Plots saved to:"
echo "  - Exp 1: $RESULTS_DIR/context_scaling/figures/"
echo "  - Exp 2: $RESULTS_DIR/moe_topk/figures/"
echo "  - Exp 3: $RESULTS_DIR/transformer_blocks_mask/figures/"
echo "  - Exp 4: $RESULTS_DIR/output_tokens/figures/"
echo "=========================================="

