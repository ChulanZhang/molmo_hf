#!/bin/bash
# Script to run combined profiling on multiple datasets
# Supports both short-answer VQA datasets and long-answer datasets

set -e

# Note: This script should be run from the project root directory
# If you run it from elsewhere, make sure to adjust paths accordingly

# Default values
MODEL_PATH="${MODEL_PATH:-checkpoints}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./results/core_exp_h100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"  # Default for short-answer datasets
NUM_SAMPLES="${NUM_SAMPLES:-12}"  # Default: 1000 samples
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-balanced}"
NUM_RUNS_PER_SAMPLE="${NUM_RUNS_PER_SAMPLE:-1}"

# Auto-detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    else
        NUM_GPUS=1
    fi
fi
NUM_GPUS="${NUM_GPUS_OVERRIDE:-${NUM_GPUS}}"

# Primary knob: vision tokens (target). select_tiling will automatically choose the best tiling
# based on each image's aspect ratio, ensuring minimal distortion.
# Vision tokens → num_crops → adaptive tiling selection per image
# Recommended values:
#   432 tokens  -> 2 crops  (small images)
#   720 tokens  -> 4 crops  (medium images)
#   1008 tokens -> 6 crops  (large images)
#   1440 tokens -> 9 crops  (very large images)
# Note: Using vision_tokens_list allows select_tiling to adapt to each image's aspect ratio,
#       avoiding the aspect ratio mismatch issues with fixed image_size_list.
# VISION_TOKENS_LIST="${VISION_TOKENS_LIST:-432 720 1008 1440}"
# TOP_K_LIST="${TOP_K_LIST:-8 12}"
# NUM_ACTIVE_BLOCKS_LIST="${NUM_ACTIVE_BLOCKS_LIST:-14 16}"
VISION_TOKENS_LIST="${VISION_TOKENS_LIST:-1440}"
TOP_K_LIST="${TOP_K_LIST:-12}"
NUM_ACTIVE_BLOCKS_LIST="${NUM_ACTIVE_BLOCKS_LIST:-16}"
# Control whether to upscale small images to fill target canvas before tiling
RESIZE_TO_FILL="${RESIZE_TO_FILL:-true}"

# Dataset configurations
# Format: "dataset_name:split:max_new_tokens"
# max_new_tokens: Upper limit for generation (actual length determined by EOS token)
#   - Short-answer VQA: 16-32 (sufficient, EOS will stop earlier if answer is short)
#   - Long-answer datasets: 256-512 (allows longer answers, EOS will stop when done)
# Note: Since use_eos_token=True, max_new_tokens is just an upper limit.
#       Model will stop early when EOS token is generated, so setting larger values is safe.
DATASETS=(
    "coco_2014_vqa:validation:16"      # Short answers, but set higher to avoid truncation
    "text_vqa:validation:64"
    "okvqa:validation:16"
    "science_qa_img:validation:16"
    "st_qa:validation:32"             # Long answers, set high to allow full generation
    "doc_qa:validation:32"
    "tally_qa:test:16"
    "mmmu:validation:16"              # Very long answers (explanations, detailed descriptions)
    "coco_caption:validation:64"      # Image captioning: captions typically 10-20 words (~15-40 tokens)
)

# Function to run combined profiling
run_combined_profiling() {
    local dataset_name=$1
    local split=$2
    local max_new_tokens=$3
    
    # Output dir will be adjusted by Python code to include dataset name
    local output_dir="${BASE_OUTPUT_DIR}"
    
    echo "=========================================="
    echo "Running Combined Profiling on ${dataset_name}"
    echo "=========================================="
    echo "Dataset: ${dataset_name}"
    echo "Split: ${split}"
    echo "Output dir: ${output_dir} (will be adjusted to include dataset name)"
    echo "Batch size: 1 (FIXED for accurate per-sample measurement)"
    echo "Num samples: ${NUM_SAMPLES} (total across all ranks)"
    echo "Max new tokens: ${max_new_tokens} (upper limit, EOS token will stop early)"
    echo "Number of GPUs: ${NUM_GPUS}"
    echo "Vision tokens: ${VISION_TOKENS_LIST}"
    echo "Top K: ${TOP_K_LIST}"
    echo "Active blocks: ${NUM_ACTIVE_BLOCKS_LIST}"
    echo ""
    echo "Note: use_eos_token=True, so max_new_tokens is just an upper limit."
    echo "      Model will stop when EOS token is generated, even if max_new_tokens is larger."
    echo ""
    
    # Ensure log directory exists
    local log_dir="${BASE_OUTPUT_DIR}/logs"
    mkdir -p "${log_dir}"
    local log_file="${log_dir}/combined_profiling_${dataset_name}_$(date +%Y%m%d_%H%M%S).log"
    echo "Saving terminal log to: ${log_file}"
    
    # Build resize_to_fill flag based on RESIZE_TO_FILL variable
    RESIZE_FLAG=""
    if [ "${RESIZE_TO_FILL}" = "true" ]; then
        RESIZE_FLAG="--resize_to_fill"
    fi
    
    torchrun --nproc-per-node=${NUM_GPUS} experiments/core_exp/acc_lat_profiling.py \
        --model_path "${MODEL_PATH}" \
        --output_dir "${output_dir}" \
        --dataset_name "${dataset_name}" \
        --split "${split}" \
        --max_new_tokens "${max_new_tokens}" \
        --vision_tokens_list ${VISION_TOKENS_LIST} \
        --top_k_list ${TOP_K_LIST} \
        --num_active_blocks_list ${NUM_ACTIVE_BLOCKS_LIST} \
        --sampling_strategy "${SAMPLING_STRATEGY}" \
        --num_samples "${NUM_SAMPLES}" \
        --num_runs_per_sample "${NUM_RUNS_PER_SAMPLE}" \
        ${RESIZE_FLAG} 2>&1 | tee "${log_file}"
    
    echo ""
    echo "Combined profiling completed for ${dataset_name}"
    echo "Results saved to: ${output_dir}/${dataset_name//_/-}"
    echo "=========================================="
    echo ""
}

# Main execution
echo "=========================================="
echo "Multi-Dataset Combined Profiling"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Base output dir: ${BASE_OUTPUT_DIR}"
echo "Batch size: 1 (FIXED for accurate per-sample measurement)"
echo "Num samples: ${NUM_SAMPLES}"
echo "Sampling strategy: ${SAMPLING_STRATEGY}"
echo "Number of GPUs: ${NUM_GPUS} (auto-detected, override with NUM_GPUS_OVERRIDE)"
echo ""
echo "Knob ranges:"
echo "  Vision tokens: ${VISION_TOKENS_LIST}"
echo "  Top K: ${TOP_K_LIST}"
echo "  Active blocks: ${NUM_ACTIVE_BLOCKS_LIST}"
echo "  Resize to fill: ${RESIZE_TO_FILL}"
echo ""
echo "Note: Using vision_tokens_list allows select_tiling to adapt to each image's aspect ratio"
echo "      for minimal distortion. Results will be saved in dataset-specific subdirectories"
echo "=========================================="
echo ""

# Parse command line arguments
SPECIFIC_DATASET="${1:-}"  # Optional: run only specific dataset

for dataset_config in "${DATASETS[@]}"; do
    IFS=':' read -r dataset_name split max_new_tokens <<< "$dataset_config"
    
    # Skip if specific dataset is requested and doesn't match
    if [ -n "${SPECIFIC_DATASET}" ] && [ "${dataset_name}" != "${SPECIFIC_DATASET}" ]; then
        continue
    fi
    
    run_combined_profiling "${dataset_name}" "${split}" "${max_new_tokens}"
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results saved in dataset-specific subdirectories under:"
echo "  ${BASE_OUTPUT_DIR}/<dataset-name>/"
echo "=========================================="

