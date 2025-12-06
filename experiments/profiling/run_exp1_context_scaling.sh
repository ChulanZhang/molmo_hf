#!/bin/bash
# Run Profiling Experiment 1: Context Scaling (Vision Tokens)
# Usage: bash experiments/profiling/run_exp1_context_scaling.sh [GPU_ID] [--num_samples N] [--max_grid_size M] [--num_runs R]

export CUDA_VISIBLE_DEVICES=${1:-0}
shift # Remove the first argument (GPU_ID) from $@
echo "Running Profiling Exp 1: Context Scaling (Vision Tokens) on GPU $CUDA_VISIBLE_DEVICES..."

# Configuration
MODEL_PATH="checkpoints"
OUTPUT_DIR="./results/profiling/context_scaling"
NUM_SAMPLES=12
MAX_GRID_SIZE=12
NUM_RUNS=1

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_grid_size)
            MAX_GRID_SIZE="$2"
            shift 2
            ;;
        --num_runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash experiments/profiling/run_exp1_context_scaling.sh [GPU_ID] [--num_samples N] [--max_grid_size M] [--num_runs R] [--model_path PATH] [--output_dir DIR]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Profiling Experiment 1: Context Scaling (Vision Tokens)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Number of tiling configs: $NUM_SAMPLES"
echo "Max grid size: $MAX_GRID_SIZE"
echo "Runs per config: $NUM_RUNS"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python experiments/profiling/knob1_tokens/exp_context_scaling.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --max_grid_size "$MAX_GRID_SIZE" \
    --num_runs "$NUM_RUNS"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Exp 1 completed! Results saved to $OUTPUT_DIR"
else
    echo ""
    echo "Exp 1 failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

