#!/bin/bash
# Run Experiment 6: Crop Overlap Analysis
# Usage: bash experiments/motivate/run_exp6.sh [GPU_ID] [--num_samples N] [--max_size SIZE] [--analyze_top_k K]

export CUDA_VISIBLE_DEVICES=${1:-0}
shift # Remove the first argument (GPU_ID) from $@
echo "Running Exp 6: Crop Overlap Analysis on GPU $CUDA_VISIBLE_DEVICES..."

# Configuration
MODEL_PATH="checkpoints"
DATASET="coco_2014_vqa"
SPLIT="validation"
NUM_SAMPLES=100
MAX_SIZE=500
ANALYZE_TOP_K=5
OUTPUT_DIR="./results/motivation/exp6"
DEVICE="cuda"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --max_size)
            MAX_SIZE="$2"
            shift 2
            ;;
        --analyze_top_k)
            ANALYZE_TOP_K="$2"
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
echo "Experiment 6: Crop Overlap Analysis"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET ($SPLIT)"
echo "Search: $NUM_SAMPLES samples, max_size=$MAX_SIZE"
echo "Analyze: Top $ANALYZE_TOP_K smallest images"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python experiments/motivate/exp6_crop_overlap_analysis.py \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --num_samples "$NUM_SAMPLES" \
    --max_size "$MAX_SIZE" \
    --analyze_top_k "$ANALYZE_TOP_K" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Exp 6 completed! Results saved to $OUTPUT_DIR"
else
    echo ""
    echo "Exp 6 failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

