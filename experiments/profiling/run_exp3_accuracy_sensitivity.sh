#!/bin/bash
# Run Exp3 Accuracy Sensitivity: Layer Importance Analysis
# Two-stage approach:
#   1. Sensitivity Analysis: Ablate each layer to compute importance scores
#   2. Importance-based Pruning: Remove layers from least to most important
#
# Usage:
#   Single GPU:  bash experiments/profiling/run_exp3_accuracy_sensitivity.sh [GPU_ID] [OPTIONS]
#   Multi-GPU:   torchrun --nproc-per-node=N experiments/profiling/knob3_layers/exp3_accuracy_sensitivity.py [OPTIONS]
#
# Examples:
#   # Single GPU - full run (sensitivity + pruning)
#   bash experiments/profiling/run_exp3_accuracy_sensitivity.sh 0
#   
#   # Single GPU - skip sensitivity analysis (use saved importance scores)
#   bash experiments/profiling/run_exp3_accuracy_sensitivity.sh 0 \
#       --skip_sensitivity \
#       --importance_scores_file ./results/profiling/exp3_accuracy_sensitivity/layer_importance_scores.json
#   
#   # Multi-GPU (call Python script directly)
#   torchrun --nproc-per-node=4 experiments/profiling/knob3_layers/exp3_accuracy_sensitivity.py \
#       --model_path checkpoints \
#       --output_dir ./results/profiling/exp3_accuracy_sensitivity \
#       --batch_size 8 \
#       --num_samples 5000

# Configuration
MODEL_PATH="${MODEL_PATH:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/profiling/exp3_accuracy_sensitivity}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
MIN_LAYERS="${MIN_LAYERS:-8}"
SKIP_SENSITIVITY="${SKIP_SENSITIVITY:-false}"
IMPORTANCE_SCORES_FILE="${IMPORTANCE_SCORES_FILE:-}"

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
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --min_layers)
            MIN_LAYERS="$2"
            shift 2
            ;;
        --skip_sensitivity)
            SKIP_SENSITIVITY="true"
            shift
            ;;
        --importance_scores_file)
            IMPORTANCE_SCORES_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage:"
            echo "  Single GPU:  bash experiments/profiling/run_exp3_accuracy_sensitivity.sh [GPU_ID] [OPTIONS]"
            echo "  Multi-GPU:   torchrun --nproc-per-node=N experiments/profiling/knob3_layers/exp3_accuracy_sensitivity.py [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_path PATH              Model path (default: checkpoints)"
            echo "  --output_dir DIR               Output directory (default: ./results/profiling/exp3_accuracy_sensitivity)"
            echo "  --batch_size N                 Batch size per GPU (default: 8)"
            echo "  --max_new_tokens N             Max tokens to generate (default: 16)"
            echo "  --num_samples N                Number of samples for evaluation (default: 5000)"
            echo "  --min_layers N                 Minimum number of layers to test (default: 8)"
            echo "  --skip_sensitivity             Skip sensitivity analysis, load from file"
            echo "  --importance_scores_file PATH  Path to saved importance scores JSON file"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Exp3 Accuracy Sensitivity: Layer Importance"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Number of samples: $NUM_SAMPLES"
echo "Minimum layers: $MIN_LAYERS"
if [ "$SKIP_SENSITIVITY" = "true" ]; then
    echo "Sensitivity analysis: SKIPPED (using saved scores)"
    if [ -n "$IMPORTANCE_SCORES_FILE" ]; then
        echo "Importance scores file: $IMPORTANCE_SCORES_FILE"
    fi
else
    echo "Sensitivity analysis: ENABLED"
fi
if [ "$SINGLE_GPU" = true ]; then
    echo "Mode: Single GPU (GPU $CUDA_VISIBLE_DEVICES)"
fi
echo "=========================================="

# Build Python command
PYTHON_CMD="python experiments/profiling/knob3_layers/exp3_accuracy_sensitivity.py \
    --model_path \"$MODEL_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --batch_size $BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num_samples $NUM_SAMPLES \
    --min_layers $MIN_LAYERS"

if [ "$SKIP_SENSITIVITY" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --skip_sensitivity"
    if [ -n "$IMPORTANCE_SCORES_FILE" ]; then
        PYTHON_CMD="$PYTHON_CMD --importance_scores_file \"$IMPORTANCE_SCORES_FILE\""
    fi
fi

echo ""
echo "Running command:"
echo "$PYTHON_CMD"
echo ""

eval $PYTHON_CMD

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Exp3 Sensitivity completed! Results saved to $OUTPUT_DIR"
    echo "   - Layer importance scores: $OUTPUT_DIR/layer_importance_scores.json"
    echo "   - Final results: $OUTPUT_DIR/exp3_accuracy_sensitivity_results.json"
else
    echo ""
    echo "❌ Exp3 Sensitivity failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

