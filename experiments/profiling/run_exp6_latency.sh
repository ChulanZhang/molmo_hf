#!/bin/bash
# Exp6 Latency: Combined Control Knobs Latency Analysis
# Tests combinations of max_crops, top_k, and num_active_blocks
# Fixed batch_size=1 for accurate per-sample latency measurement

set -e

# Default values
MODEL_PATH="${MODEL_PATH:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/profiling/exp6_latency}"
DATASET_NAME="${DATASET_NAME:-coco_2014_vqa}"
SPLIT="${SPLIT:-validation}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"

# Sampling strategy options:
# - "full": All combinations (6×8×5=240 combinations)
# - "stratified": Min, middle, max from each dimension (3×3×3=27 combinations)
# - "boundary": More comprehensive boundary sampling (3×5×3=45 combinations)
# - "balanced": Balanced coverage (3×4×3=36 combinations) [RECOMMENDED]
# - "custom_sparse": Every 2nd value (3×4×3=36 combinations)
# - "lhs": Latin Hypercube Sampling (customizable number)

SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-balanced}"

# Default knob ranges (updated based on exp3 sensitivity results)
# max_crops: [2, 4, 6, 8, 10] (5 values, step 2, max 10)
# top_k: [4, 8, 12] (3 values, max 12)
# num_active_blocks: [12, 13, 14, 15, 16] (5 values, min 12, max 16)

# Check if running with torchrun (multi-GPU)
if command -v torchrun &> /dev/null && [ -n "${RANK:-}" ]; then
    # Multi-GPU mode: torchrun will call the Python script directly
    # We need to pass all arguments to the Python script
    echo "Running in multi-GPU mode with torchrun"
    echo "Note: torchrun should be called directly with the Python script:"
    echo "  torchrun --nproc-per-node=4 experiments/profiling/knob5_combined/exp6_accuracy.py \\"
    echo "    --model_path ${MODEL_PATH} \\"
    echo "    --output_dir ${OUTPUT_DIR} \\"
    echo "    --sampling_strategy ${SAMPLING_STRATEGY} \\"
    echo "    --num_samples ${NUM_SAMPLES}"
    echo ""
    echo "For single-GPU mode, run this script directly:"
    echo "  ./run_exp6_latency.sh"
    exit 0
fi

# Single-GPU mode
echo "=========================================="
echo "Exp6 Latency: Combined Control Knobs"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Dataset: ${DATASET_NAME}/${SPLIT}"
echo "Batch size: 1 (fixed for accurate latency measurement)"
echo "Number of samples: ${NUM_SAMPLES}"
echo "Sampling strategy: ${SAMPLING_STRATEGY}"
echo "=========================================="
echo ""

# Run the experiment
python experiments/profiling/knob5_combined/exp6_accuracy.py \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${SPLIT}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --num_samples "${NUM_SAMPLES}" \
    --sampling_strategy "${SAMPLING_STRATEGY}" \
    "$@"

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="

