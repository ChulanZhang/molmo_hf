#!/bin/bash
# Run Phase 1: Dataset Profiling (Exp 1 & 3)
# Usage: bash experiments/motivate/run_phase1.sh [GPU_ID]

export CUDA_VISIBLE_DEVICES=${1:-0}
echo "Running Phase 1 on GPU $CUDA_VISIBLE_DEVICES..."

# Configuration
MODEL_PATH="checkpoints"
DATASET="coco_2014_vqa"
SPLIT="validation"
NUM_SAMPLES=9999999
OUTPUT_DIR="./results"
DEVICE="cuda"

# Run (using module execution to handle imports correctly)
python -m experiments.motivate.run_unified_experiments \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR/phase1-full" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --num_samples "$NUM_SAMPLES" \
    --device "$DEVICE" \
    --phase1_only
