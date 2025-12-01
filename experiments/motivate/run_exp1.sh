#!/bin/bash
# Run Experiment 1: Latency Distribution
# Usage: bash experiments/motivate/run_exp1.sh [GPU_ID] [--num_samples N]

export CUDA_VISIBLE_DEVICES=${1:-0}
shift # Remove the first argument (GPU_ID) from $@
echo "Running Exp 1: Latency Distribution on GPU $CUDA_VISIBLE_DEVICES..."

# Configuration
MODEL_PATH="checkpoints"
DATASET="coco_2014_vqa"
SPLIT="validation"
NUM_SAMPLES=9999999
OUTPUT_DIR="./results/motivation/exp1"
DEVICE="cuda"

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
        *)
            shift
            ;;
    esac
done

echo "=========================================="
echo "Experiment 1: Latency Distribution"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET ($SPLIT)"
echo "Samples: $NUM_SAMPLES"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python experiments/motivate/exp1_latency_distribution.py \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo ""
echo "Exp 1 completed! Results saved to $OUTPUT_DIR"

