#!/bin/bash
# Run Exp2 Accuracy: MoE Top-K Analysis
# Measures accuracy on full VQA v2 validation set with different MoE top_k values
#
# Usage:
#   Single GPU:  bash experiments/profiling/run_exp2_accuracy.sh [GPU_ID] [OPTIONS]
#   Multi-GPU:   torchrun --nproc-per-node=N experiments/profiling/knob2_topk/exp2_accuracy.py [OPTIONS]
#
# Examples:
#   # Single GPU
#   bash experiments/profiling/run_exp2_accuracy.sh 0
#   bash experiments/profiling/run_exp2_accuracy.sh 0 --top_k_values 4 8 12 16
#   
#   # Multi-GPU (call Python script directly)
#   torchrun --nproc-per-node=4 experiments/profiling/knob2_topk/exp2_accuracy.py \
#       --model_path checkpoints \
#       --output_dir ./results/profiling/exp2_accuracy \
#       --batch_size 8 \
#       --top_k_values 4 8 12 16

# Configuration
MODEL_PATH="${MODEL_PATH:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/profiling/exp2_accuracy}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
TOP_K_VALUES="${TOP_K_VALUES:-}"  # Default: test all from 4 to 64, step 4 (16 groups: [4, 8, 12, ..., 64])

# Parse arguments
if [[ "$1" =~ ^[0-9]+$ ]]; then
    # First argument is GPU ID (single GPU mode)
    export CUDA_VISIBLE_DEVICES=$1
    shift
    SINGLE_GPU=true
else
    # Multi-GPU mode - but this script is for single GPU only
    # For multi-GPU, users should call Python script directly with torchrun
    SINGLE_GPU=true  # Default to single GPU mode
fi

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --top_k_values)
            shift
            TOP_K_VALUES=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                TOP_K_VALUES+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage:"
            echo "  Single GPU:  bash experiments/profiling/run_exp2_accuracy.sh [GPU_ID] [OPTIONS]"
            echo "  Multi-GPU:   torchrun --nproc-per-node=N experiments/profiling/knob2_topk/exp2_accuracy.py [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_path PATH          Model path (default: checkpoints)"
            echo "  --output_dir DIR          Output directory (default: ./results/profiling/exp2_accuracy)"
            echo "  --batch_size N             Batch size per GPU (default: 8)"
            echo "  --max_new_tokens N         Max tokens to generate (default: 16)"
            echo "  --top_k_values K1 K2 ...  List of top_k values to test (default: 4,8,12,...,64)"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Exp2 Accuracy: MoE Top-K Analysis"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
if [ ${#TOP_K_VALUES[@]} -eq 0 ]; then
    echo "Top-K values: (default: 4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64)"
else
    echo "Top-K values: ${TOP_K_VALUES[*]}"
fi
if [ "$SINGLE_GPU" = true ]; then
    echo "Mode: Single GPU (GPU $CUDA_VISIBLE_DEVICES)"
fi
echo "=========================================="

# Build Python command
PYTHON_CMD="python experiments/profiling/knob2_topk/exp2_accuracy.py \
    --model_path \"$MODEL_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --batch_size $BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS"

if [ ${#TOP_K_VALUES[@]} -gt 0 ]; then
    PYTHON_CMD="$PYTHON_CMD --top_k_values ${TOP_K_VALUES[*]}"
fi

echo ""
echo "Running command:"
echo "$PYTHON_CMD"
echo ""

eval $PYTHON_CMD

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Exp2 Accuracy completed! Results saved to $OUTPUT_DIR"
else
    echo ""
    echo "❌ Exp2 Accuracy failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

