#!/bin/bash
# Run Exp3 Accuracy: Transformer Blocks Mask
# Measures accuracy on full VQA v2 validation set with different numbers of active transformer blocks
# Strategy: Sequential activation from layer 1, starting from 8 layers minimum
#   - 8 layers: activates layers 1-8 (indices 0-7)
#   - 9 layers: activates layers 1-9 (indices 0-8)
#   - 10 layers: activates layers 1-10 (indices 0-9), etc.
#
# Usage:
#   Single GPU:  bash experiments/profiling/run_exp3_accuracy.sh [GPU_ID] [OPTIONS]
#   Multi-GPU:   torchrun --nproc-per-node=N experiments/profiling/knob3_layers/exp3_accuracy.py [OPTIONS]
#
# Examples:
#   # Single GPU - test all from 8 to total_blocks (default)
#   bash experiments/profiling/run_exp3_accuracy.sh 0
#   
#   # Single GPU - test specific layer counts
#   bash experiments/profiling/run_exp3_accuracy.sh 0 --num_active_blocks 8 9 10 12 16
#   
#   # Multi-GPU (call Python script directly)
#   torchrun --nproc-per-node=4 experiments/profiling/knob3_layers/exp3_accuracy.py \
#       --model_path checkpoints \
#       --output_dir ./results/profiling/exp3_accuracy \
#       --batch_size 8 \
#       --num_active_blocks 8 9 10 12 16

# Configuration
MODEL_PATH="${MODEL_PATH:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/profiling/exp3_accuracy}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
NUM_ACTIVE_BLOCKS="${NUM_ACTIVE_BLOCKS:-}"  # Default: test all from 8 to total_blocks (sequential activation)

# Parse arguments
if [[ "$1" =~ ^[0-9]+$ ]]; then
    # First argument is GPU ID (single GPU mode)
    export CUDA_VISIBLE_DEVICES=$1
    shift
    SINGLE_GPU=true
else
    # Multi-GPU mode (torchrun)
    SINGLE_GPU=false
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
        --num_active_blocks)
            shift
            NUM_ACTIVE_BLOCKS=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                NUM_ACTIVE_BLOCKS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage:"
            echo "  Single GPU:  bash experiments/profiling/run_exp3_accuracy.sh [GPU_ID] [OPTIONS]"
            echo "  Multi-GPU:   torchrun --nproc-per-node=N experiments/profiling/knob3_layers/exp3_accuracy.py [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_path PATH              Model path (default: checkpoints)"
            echo "  --output_dir DIR               Output directory (default: ./results/profiling/exp3_accuracy)"
            echo "  --batch_size N                 Batch size per GPU (default: 8)"
            echo "  --max_new_tokens N             Max tokens to generate (default: 16)"
            echo "  --num_active_blocks N1 N2 ...  List of active block counts to test (default: test all from 8 to total_blocks, sequential activation)"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Exp3 Accuracy: Transformer Blocks Mask"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
if [ ${#NUM_ACTIVE_BLOCKS[@]} -eq 0 ]; then
    echo "Active blocks: (default: test all from 8 to total_blocks, sequential activation)"
else
    echo "Active blocks: ${NUM_ACTIVE_BLOCKS[*]} (sequential: layers 1-N)"
fi
echo "Strategy: Sequential activation from layer 1, minimum 8 layers"
if [ "$SINGLE_GPU" = true ]; then
    echo "Mode: Single GPU (GPU $CUDA_VISIBLE_DEVICES)"
fi
echo "=========================================="

# Build Python command
PYTHON_CMD="python experiments/profiling/knob3_layers/exp3_accuracy.py \
    --model_path \"$MODEL_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --batch_size $BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS"

if [ ${#NUM_ACTIVE_BLOCKS[@]} -gt 0 ]; then
    PYTHON_CMD="$PYTHON_CMD --num_active_blocks ${NUM_ACTIVE_BLOCKS[*]}"
fi

echo ""
echo "Running command:"
echo "$PYTHON_CMD"
echo ""

eval $PYTHON_CMD

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Exp3 Accuracy completed! Results saved to $OUTPUT_DIR"
else
    echo ""
    echo "❌ Exp3 Accuracy failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

