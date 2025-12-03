#!/bin/bash
# Run Experiment 5: FLOPs vs Latency
# Usage: bash experiments/motivate/run_exp5.sh [--exp3_results PATH] [--exp4_results PATH]

echo "Running Exp 5: FLOPs vs Latency..."

# Configuration
EXP3_RESULTS="./results/motivation/exp3/exp3_vision_tokens_vs_latency.json"
EXP4_RESULTS="./results/motivation/exp4/exp4_language_tokens_vs_latency.json"
OUTPUT_DIR="./results/motivation/exp5"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp3_results)
            EXP3_RESULTS="$2"
            shift 2
            ;;
        --exp4_results)
            EXP4_RESULTS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Check if result files exist
if [ ! -f "$EXP3_RESULTS" ]; then
    echo "Error: Exp 3 results file not found: $EXP3_RESULTS"
    echo "Please run Exp 3 first: bash experiments/motivate/run_exp3.sh"
    exit 1
fi

if [ ! -f "$EXP4_RESULTS" ]; then
    echo "Error: Exp 4 results file not found: $EXP4_RESULTS"
    echo "Please run Exp 4 first: bash experiments/motivate/run_exp4.sh"
    exit 1
fi

echo "=========================================="
echo "Experiment 5: FLOPs vs Latency"
echo "=========================================="
echo "Exp 3 Results: $EXP3_RESULTS"
echo "Exp 4 Results: $EXP4_RESULTS"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python experiments/motivate/exp5_flops_vs_latency.py \
    --exp3_results "$EXP3_RESULTS" \
    --exp4_results "$EXP4_RESULTS" \
    --output_dir "$OUTPUT_DIR" \
    --x_axis tokens

echo ""
echo "Exp 5 completed! Results saved to $OUTPUT_DIR"

