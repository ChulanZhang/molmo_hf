#!/bin/bash
# Run Profiling Experiment 3: Transformer Blocks Mask
# Usage: 
#   Count-based: bash experiments/profiling/run_exp3_transformer_blocks_mask.sh [GPU_ID] [--num_samples N] [--num_active_blocks N1 N2 ...]
#   Specific indices: bash experiments/profiling/run_exp3_transformer_blocks_mask.sh [GPU_ID] [--num_samples N] [--active_block_indices I1 I2 ...] [--active_block_indices I3 I4 ...]

export CUDA_VISIBLE_DEVICES=${1:-0}
shift # Remove the first argument (GPU_ID) from $@
echo "Running Profiling Exp 3: Transformer Blocks Mask on GPU $CUDA_VISIBLE_DEVICES..."

# Configuration
MODEL_PATH="checkpoints"
OUTPUT_DIR="./results/profiling/transformer_blocks_mask"
NUM_SAMPLES=100
NUM_ACTIVE_BLOCKS=()  # Empty means use default (various fractions)
ACTIVE_BLOCK_INDICES=()  # Empty means use count-based mode

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
        --num_active_blocks)
            shift # Remove --num_active_blocks
            NUM_ACTIVE_BLOCKS=()
            # Collect all values until next option or end
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                NUM_ACTIVE_BLOCKS+=("$1")
                shift
            done
            ;;
        --active_block_indices)
            shift # Remove --active_block_indices
            # Collect indices for this configuration
            INDICES=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                INDICES+=("$1")
                shift
            done
            # Store as a group (we'll pass multiple --active_block_indices to Python)
            if [ ${#INDICES[@]} -gt 0 ]; then
                ACTIVE_BLOCK_INDICES+=("${INDICES[*]}")  # Store as space-separated string
            fi
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage:"
            echo "  Count-based: bash experiments/profiling/run_exp3_transformer_blocks_mask.sh [GPU_ID] [--num_samples N] [--num_active_blocks N1 N2 ...]"
            echo "  Specific indices: bash experiments/profiling/run_exp3_transformer_blocks_mask.sh [GPU_ID] [--num_samples N] [--active_block_indices I1 I2 ...] [--active_block_indices I3 I4 ...]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Profiling Experiment 3: Transformer Blocks Mask"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Samples: $NUM_SAMPLES"
if [ ${#ACTIVE_BLOCK_INDICES[@]} -gt 0 ]; then
    echo "Mode: Specific block indices"
    echo "Block configurations:"
    for idx_group in "${ACTIVE_BLOCK_INDICES[@]}"; do
        echo "  - [${idx_group}]"
    done
    BLOCK_INDICES_ARG=""
    for idx_group in "${ACTIVE_BLOCK_INDICES[@]}"; do
        BLOCK_INDICES_ARG="$BLOCK_INDICES_ARG --active_block_indices $idx_group"
    done
elif [ ${#NUM_ACTIVE_BLOCKS[@]} -eq 0 ]; then
    echo "Active blocks: (default - various fractions)"
    BLOCK_INDICES_ARG=""
else
    echo "Mode: Count-based (first N blocks)"
    echo "Active block counts: ${NUM_ACTIVE_BLOCKS[*]}"
    BLOCK_INDICES_ARG="--num_active_blocks ${NUM_ACTIVE_BLOCKS[*]}"
fi
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Build Python command
PYTHON_CMD="python experiments/profiling/knob3_layers/exp_transformer_blocks_mask.py \
    --model_path \"$MODEL_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --num_samples \"$NUM_SAMPLES\""

if [ ${#ACTIVE_BLOCK_INDICES[@]} -gt 0 ]; then
    # Specific indices mode
    for idx_group in "${ACTIVE_BLOCK_INDICES[@]}"; do
        PYTHON_CMD="$PYTHON_CMD --active_block_indices $idx_group"
    done
elif [ ${#NUM_ACTIVE_BLOCKS[@]} -gt 0 ]; then
    # Count-based mode
    PYTHON_CMD="$PYTHON_CMD --num_active_blocks ${NUM_ACTIVE_BLOCKS[*]}"
fi

eval $PYTHON_CMD

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Exp 3 completed! Results saved to $OUTPUT_DIR"
else
    echo ""
    echo "Exp 3 failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
