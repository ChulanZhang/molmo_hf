#!/bin/bash
# Run Profiling Experiment 4: Output Tokens Scaling
# Usage: bash experiments/profiling/run_exp4_output_tokens.sh [GPU_ID] [--num_samples N] [--max_new_tokens N1 N2 ...]

export CUDA_VISIBLE_DEVICES=${1:-0}
shift # Remove the first argument (GPU_ID) from $@
echo "Running Profiling Exp 4: Output Tokens Scaling on GPU $CUDA_VISIBLE_DEVICES..."

# Configuration
MODEL_PATH="checkpoints"
OUTPUT_DIR="./results/profiling/output_tokens"
NUM_SAMPLES=50
MAX_NEW_TOKENS=()  # Empty means use default

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
        --max_new_tokens)
            shift # Remove --max_new_tokens
            MAX_NEW_TOKENS=()
            # Collect all values until next option or end
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                MAX_NEW_TOKENS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash experiments/profiling/run_exp4_output_tokens.sh [GPU_ID] [--num_samples N] [--max_new_tokens N1 N2 ...] [--model_path PATH] [--output_dir DIR]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Profiling Experiment 4: Output Tokens Scaling"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Samples: $NUM_SAMPLES"
if [ ${#MAX_NEW_TOKENS[@]} -eq 0 ]; then
    echo "Max new tokens: (default - [1, 5, 10, 20, 50, 100, 200])"
    MAX_TOKENS_ARG=""
else
    echo "Max new tokens: ${MAX_NEW_TOKENS[*]}"
    MAX_TOKENS_ARG="--max_new_tokens ${MAX_NEW_TOKENS[*]}"
fi
echo "Output: $OUTPUT_DIR"
echo "=========================================="

if [ ${#MAX_NEW_TOKENS[@]} -eq 0 ]; then
    python experiments/profiling/knob4_output_tokens/exp_output_tokens_scaling.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_samples "$NUM_SAMPLES"
else
    python experiments/profiling/knob4_output_tokens/exp_output_tokens_scaling.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_samples "$NUM_SAMPLES" \
        --max_new_tokens "${MAX_NEW_TOKENS[@]}"
fi

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Exp 4 completed! Results saved to $OUTPUT_DIR"
else
    echo ""
    echo "Exp 4 failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

