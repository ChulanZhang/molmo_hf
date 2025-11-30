#!/bin/bash
# Run Phase 2: Controlled Scaling (Exp 2, 4a, 5)
# Usage: bash experiments/motivate/run_phase2.sh [GPU_ID]

export CUDA_VISIBLE_DEVICES=${1:-0}
echo "Running Phase 2 on GPU $CUDA_VISIBLE_DEVICES..."

# Configuration
MODEL_PATH="hf:allenai/MolmoE-1B-0924"
DATASET="coco_2014_vqa"
SPLIT="validation"
NUM_SAMPLES=1000 # Not used for scaling but kept for consistency
OUTPUT_DIR="./results"
DEVICE="cuda"

# Run (using module execution to handle imports correctly)
python -m experiments.motivate.run_unified_experiments \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR/phase2" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --num_samples "$NUM_SAMPLES" \
    --device "$DEVICE" \
    --phase2_only
