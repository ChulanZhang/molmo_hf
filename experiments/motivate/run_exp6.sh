#!/bin/bash
# Run Experiment 6: Crop Overlap Analysis
# Usage: bash experiments/motivate/run_exp6.sh [GPU_ID] [--exp2_results PATH] [--num_small_crops N] [--num_large_crops N]

export CUDA_VISIBLE_DEVICES=${1:-0}
shift # Remove the first argument (GPU_ID) from $@
echo "Running Exp 6: Crop Overlap Analysis on GPU $CUDA_VISIBLE_DEVICES..."

# Configuration
MODEL_PATH="checkpoints"
DATASET="coco_2014_vqa"
SPLIT="validation"
EXP2_RESULTS="./results/motivation/exp2/exp2_component_profiling.json"
NUM_IMAGES=5
TARGET_MAX_CROPS=12
OUTPUT_DIR="./results/motivation/exp6"
DEVICE="cuda"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp2_results)
            EXP2_RESULTS="$2"
            shift 2
            ;;
        --num_images)
            NUM_IMAGES="$2"
            shift 2
            ;;
        --target_max_crops)
            TARGET_MAX_CROPS="$2"
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
if [ -n "$EXP2_RESULTS" ]; then
    echo "Source: Exp2 results from $EXP2_RESULTS"
    echo "Select: $NUM_IMAGES images (evenly distributed across crop count range)"
    if [ -n "$TARGET_MAX_CROPS" ]; then
        echo "Target max crops: $TARGET_MAX_CROPS"
    fi
else
    echo "Source: Small images search"
fi
echo "Output: $OUTPUT_DIR"
echo "=========================================="

if [ -n "$EXP2_RESULTS" ]; then
    python experiments/motivate/exp6_crop_overlap_analysis.py \
        --model_path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --exp2_results "$EXP2_RESULTS" \
        --num_images "$NUM_IMAGES" \
        --target_max_crops "$TARGET_MAX_CROPS" \
        --save_crop_images \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE"
else
    python experiments/motivate/exp6_crop_overlap_analysis.py \
        --model_path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --num_samples 100 \
        --max_size 500 \
        --analyze_top_k 5 \
        --save_crop_images \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE"
fi

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Exp 6 completed! Results saved to $OUTPUT_DIR"
else
    echo ""
    echo "Exp 6 failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

