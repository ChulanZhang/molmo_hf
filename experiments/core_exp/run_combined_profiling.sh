#!/bin/bash
# Combined Profiling: Accuracy and Latency with Vision Tokens Control
# Tests combinations of vision tokens (target), MoE top_k, and transformer blocks.
# Records detailed data for E1, E2, E3 analysis.

set -e

# Note: This script should be run from the project root directory
# If you run it from elsewhere, make sure to adjust paths accordingly

# Default values
MODEL_PATH="${MODEL_PATH:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/core_exp}"
DATASET_NAME="${DATASET_NAME:-coco_2014_vqa}"
SPLIT="${SPLIT:-validation}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

# Sampling strategy options:
# - "full": All combinations
# - "balanced": Balanced coverage (recommended)
# - "stratified": Min, middle, max from each dimension
# - "boundary": More comprehensive boundary sampling
# - "lhs": Latin Hypercube Sampling
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-balanced}"

# Dataset sampling: Number of samples to use (None = use all)
# Recommended: 1000-2000 for combined profiling
# This reduces experiment time while maintaining statistical significance
NUM_SAMPLES="${NUM_SAMPLES:-1000}"

# Number of runs per sample for latency averaging
# Recommended: 3-5 for stable measurements
NUM_RUNS_PER_SAMPLE="${NUM_RUNS_PER_SAMPLE:-3}"

# Use PyTorch profiler for detailed operator-level analysis
# If enabled, can use fewer samples (e.g., 100-200) for detailed analysis
USE_PROFILER="${USE_PROFILER:-false}"

# Primary knob: image sizes (HxW). Each size maps to tiling → num_crops → vision tokens.
# Suggested sizes (resize target):
#   560x336  -> tiling 1x2  -> ~384 actual tokens
#   560x784  -> tiling 2x3  -> ~744 actual tokens
#   784x784  -> tiling 3x3  -> ~1044 actual tokens
IMAGE_SIZE_LIST="${IMAGE_SIZE_LIST:-560x336 560x784 784x784}"

# Default top_k list (based on previous experiments)
TOP_K_LIST="${TOP_K_LIST:-4 8 12}"

# Default num_active_blocks list (based on previous experiments)
NUM_ACTIVE_BLOCKS_LIST="${NUM_ACTIVE_BLOCKS_LIST:-12 13 14 15 16}"

# Check if running with torchrun (multi-GPU)
if command -v torchrun &> /dev/null && [ -n "${RANK:-}" ]; then
    echo "Running in multi-GPU mode with torchrun"
    echo "Note: torchrun should be called directly with the Python script:"
    echo "  torchrun --nproc-per-node=4 experiments/core_exp/combined_profiling.py \\"
    echo "    --model_path ${MODEL_PATH} \\"
    echo "    --output_dir ${OUTPUT_DIR} \\"
    echo "    --sampling_strategy ${SAMPLING_STRATEGY} \\"
    echo "    --num_samples ${NUM_SAMPLES} \\"
    echo "    --num_runs_per_sample ${NUM_RUNS_PER_SAMPLE} \\"
    echo "    --vision_tokens_list ${VISION_TOKENS_LIST}"
    if [ "$USE_PROFILER" = "true" ]; then
        echo "    --use_profiler"
    fi
    echo ""
    echo "For single-GPU mode, run this script directly:"
    echo "  ./run_combined_profiling.sh"
    exit 0
fi

# Single-GPU mode
echo "=========================================="
echo "Combined Profiling: Accuracy and Latency"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Dataset: ${DATASET_NAME}/${SPLIT}"
echo "Batch size: 1 (fixed for accurate per-sample measurement)"
echo "Number of samples: ${NUM_SAMPLES}"
echo "Runs per sample: ${NUM_RUNS_PER_SAMPLE}"
echo "Sampling strategy: ${SAMPLING_STRATEGY}"
echo "Use profiler: ${USE_PROFILER}"
echo "Image size list: ${IMAGE_SIZE_LIST} (primary knob)"
echo "=========================================="
echo ""

# Build command
CMD="python experiments/core_exp/combined_profiling.py \
    --model_path \"${MODEL_PATH}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --dataset_name \"${DATASET_NAME}\" \
    --split \"${SPLIT}\" \
    --max_new_tokens \"${MAX_NEW_TOKENS}\" \
    --sampling_strategy \"${SAMPLING_STRATEGY}\" \
    --num_samples \"${NUM_SAMPLES}\" \
    --num_runs_per_sample \"${NUM_RUNS_PER_SAMPLE}\" \
    --image_size_list ${IMAGE_SIZE_LIST} \
    --top_k_list ${TOP_K_LIST} \
    --num_active_blocks_list ${NUM_ACTIVE_BLOCKS_LIST}"

# Add profiler flag if enabled
if [ "$USE_PROFILER" = "true" ]; then
    CMD="$CMD --use_profiler"
fi

# Run the experiment
eval $CMD "$@"

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="

