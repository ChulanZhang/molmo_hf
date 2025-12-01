#!/bin/bash
# Run Experiment 3: Vision Tokens vs Latency
# Usage: bash experiments/motivate/run_exp3.sh [GPU_ID] [--max_grid_size N] [--num_runs N]

export CUDA_VISIBLE_DEVICES=${1:-0}
shift # Remove the first argument (GPU_ID) from $@
echo "Running Exp 3: Vision Tokens vs Latency on GPU $CUDA_VISIBLE_DEVICES..."

# Configuration
MODEL_PATH="checkpoints"
OUTPUT_DIR="./results/motivation/exp3"
DEVICE="cuda"
MAX_GRID_SIZE=12
NUM_RUNS=10
USE_HOOK="--use_hook_for_llm_prefill"  # Default: use hooks for direct measurement

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_grid_size)
            MAX_GRID_SIZE="$2"
            shift 2
            ;;
        --num_runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --use_hook_for_llm_prefill)
            USE_HOOK="--use_hook_for_llm_prefill"
            shift
            ;;
        --no-hook)
            USE_HOOK=""  # Disable hook, use subtraction method
            shift
            ;;
        --model_path)
            MODEL_PATH="$2"
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

echo "=========================================="
echo "Experiment 3: Vision Tokens vs Latency"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Max Grid Size: $MAX_GRID_SIZE"
echo "Num Runs: $NUM_RUNS (per resolution)"
if [ -n "$USE_HOOK" ]; then
    echo "LLM Prefill Measurement: Direct (using hooks) [default]"
else
    echo "LLM Prefill Measurement: Subtraction method"
fi
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python experiments/motivate/exp3_vision_tokens_vs_latency.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --max_grid_size "$MAX_GRID_SIZE" \
    --num_runs "$NUM_RUNS" \
    $USE_HOOK

echo ""
echo "Exp 3 completed! Results saved to $OUTPUT_DIR"

