#!/bin/bash
# Script to run only the last experiment (max_crops=10, top_k=12, num_active_blocks=16)
# This experiment failed to save results due to a distributed communication timeout

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

cd "$PROJECT_ROOT"

# Activate environment if activate_env.sh exists
if [ -f "activate_env.sh" ]; then
    source activate_env.sh
fi

echo "=========================================="
echo "Running last experiment: max_crops=10, top_k=12, num_active_blocks=16"
echo "=========================================="
echo "Model path: checkpoints"
echo "Output dir: ./results/profiling/exp5_accuracy"
echo "Dataset: coco_2014_vqa/validation"
echo "Batch size: 64 (with auto-adjust)"
echo "=========================================="
echo ""

# Run only the last experiment configuration
# max_crops=10, top_k=12, num_active_blocks=16
torchrun --nproc-per-node=4 \
    experiments/profiling/knob5_combined/exp5_accuracy.py \
    --model_path checkpoints \
    --output_dir ./results/profiling/exp5_accuracy \
    --dataset_name coco_2014_vqa \
    --split validation \
    --batch_size 64 \
    --max_new_tokens 16 \
    --max_crops 10 \
    --top_k 12 \
    --num_active_blocks 16 \
    --sampling_strategy balanced \
    --auto_adjust_batch_size

echo ""
echo "=========================================="
echo "Last experiment completed!"
echo "Results saved to: ./results/profiling/exp5_accuracy"
echo "=========================================="

