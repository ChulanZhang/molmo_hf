#!/bin/bash
# Training script for One-Stage Controller
# 
# This script provides a convenient way to train the one-stage controller
# with commonly used configurations.

# Default values
RESULTS_DIR="results/core_exp_h100"
DATASET_NAMES="text_vqa"
MODEL_PATH="checkpoints/molmo"
OUTPUT_DIR="checkpoints/one_stage_controller"
BATCH_SIZE=1
NUM_EPOCHS=100
LR=1e-4
GROUP_SIZE=5
IMPORTANCE_SCORES_FILE="results/layer_importance_scores_exp3_recommended.json"
USE_WANDB=false
USE_MULTI_GPU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --dataset_names)
            DATASET_NAMES="$2"
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
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --group_size)
            GROUP_SIZE="$2"
            shift 2
            ;;
        --importance_scores_file)
            IMPORTANCE_SCORES_FILE="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB=true
            shift
            ;;
        --use_multi_gpu)
            USE_MULTI_GPU=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="python experiments/controller/train_joint_controller.py \
    --results_dir ${RESULTS_DIR} \
    --dataset_names ${DATASET_NAMES} \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --group_size ${GROUP_SIZE} \
    --importance_scores_file ${IMPORTANCE_SCORES_FILE}"

# Add optional flags
if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb"
fi

if [ "$USE_MULTI_GPU" = true ]; then
    CMD="$CMD --use_multi_gpu"
fi

# Print command and execute
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

