#!/bin/bash
# Script to run exp5 (accuracy) and exp6 (latency) on datasets requiring LONG answers
# These datasets need larger max_new_tokens (128-256) compared to short-answer VQA datasets (16)
# Datasets: DocVQA, ST-VQA, InfoQA, and other datasets that may require detailed explanations

set -e

# Default values
MODEL_PATH="${MODEL_PATH:-checkpoints}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./results/profiling}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"  # Larger value for long-answer datasets (default: 128, can be 256)
NUM_SAMPLES="${NUM_SAMPLES:-1000}"  # Default: 1000 samples for exp6
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-balanced}"

# Auto-detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    # Fallback: check CUDA_VISIBLE_DEVICES or default to 1
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    else
        NUM_GPUS=1
    fi
fi

# Use detected number of GPUs, but allow override via environment variable
NUM_GPUS="${NUM_GPUS_OVERRIDE:-${NUM_GPUS}}"

# Dataset configurations for LONG-ANSWER datasets
# Format: "dataset_name:split:experiment"
# experiment can be "exp5", "exp6", or "both"
# These datasets typically require longer answers (explanations, detailed descriptions, etc.)
DATASETS=(
    "mmmu:validation:both"              # MMMU - multi-domain dataset requiring detailed answers
    # "doc_qa:validation:both"            # Document VQA - may require longer answers
    # "st_qa:validation:both"            # Scene Text QA - may require longer answers
    # "info_qa:validation:both"           # InfoQA - may require longer answers
    # Add more long-answer datasets here as needed
)

# Function to run exp5
run_exp5() {
    local dataset_name=$1
    local split=$2
    # Output dir will be adjusted by Python code to include dataset name
    # Format: ${BASE_OUTPUT_DIR}/exp5_accuracy_long_${dataset_name}
    local output_dir="${BASE_OUTPUT_DIR}/exp5_accuracy_long"
    
    echo "=========================================="
    echo "Running Exp5 (Accuracy) on ${dataset_name} (LONG-ANSWER)"
    echo "=========================================="
    echo "Dataset: ${dataset_name}"
    echo "Split: ${split}"
    echo "Output dir: ${output_dir} (will be adjusted to include dataset name)"
    echo "Batch size: ${BATCH_SIZE} (with auto-adjustment)"
    echo "Max new tokens: ${MAX_NEW_TOKENS} (for long answers)"
    echo "Number of GPUs: ${NUM_GPUS}"
    echo ""
    
    # Ensure log directory exists
    local log_dir="${BASE_OUTPUT_DIR}/logs"
    mkdir -p "${log_dir}"
    local log_file="${log_dir}/exp5_long_${dataset_name}_$(date +%Y%m%d_%H%M%S).log"
    echo "Saving terminal log to: ${log_file}"
    
    torchrun --nproc-per-node=${NUM_GPUS} experiments/profiling/knob5_combined/exp5_accuracy.py \
        --model_path "${MODEL_PATH}" \
        --output_dir "${output_dir}" \
        --dataset_name "${dataset_name}" \
        --split "${split}" \
        --batch_size "${BATCH_SIZE}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --max_crops 2 4 6 8 10 \
        --top_k 4 8 12 \
        --num_active_blocks 12 13 14 15 16 \
        --sampling_strategy "${SAMPLING_STRATEGY}" \
        --auto_adjust_batch_size 2>&1 | tee "${log_file}"
    
    echo ""
    echo "Exp5 completed for ${dataset_name}"
    echo "Results saved to: ${output_dir}_${dataset_name//_/-}"
    echo "=========================================="
    echo ""
}

# Function to run exp6
run_exp6() {
    local dataset_name=$1
    local split=$2
    # Output dir will be adjusted by Python code to include dataset name
    # Format: ${BASE_OUTPUT_DIR}/exp6_latency_long_${dataset_name}
    local output_dir="${BASE_OUTPUT_DIR}/exp6_latency_long"
    
    echo "=========================================="
    echo "Running Exp6 (Latency) on ${dataset_name} (LONG-ANSWER)"
    echo "=========================================="
    echo "Dataset: ${dataset_name}"
    echo "Split: ${split}"
    echo "Output dir: ${output_dir} (will be adjusted to include dataset name)"
    echo "Batch size: 1 (FIXED for accurate per-sample latency measurement)"
    echo "Num samples: ${NUM_SAMPLES} (total across all ranks)"
    echo "Max new tokens: ${MAX_NEW_TOKENS} (for long answers)"
    echo "Number of GPUs: ${NUM_GPUS}"
    echo ""
    
    # Ensure log directory exists
    local log_dir="${BASE_OUTPUT_DIR}/logs"
    mkdir -p "${log_dir}"
    local log_file="${log_dir}/exp6_long_${dataset_name}_$(date +%Y%m%d_%H%M%S).log"
    echo "Saving terminal log to: ${log_file}"
    
    torchrun --nproc-per-node=${NUM_GPUS} experiments/profiling/knob5_combined/exp6_accuracy.py \
        --model_path "${MODEL_PATH}" \
        --output_dir "${output_dir}" \
        --dataset_name "${dataset_name}" \
        --split "${split}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --max_crops 2 4 6 8 10 \
        --top_k 4 8 12 \
        --num_active_blocks 12 13 14 15 16 \
        --num_samples "${NUM_SAMPLES}" \
        --sampling_strategy "${SAMPLING_STRATEGY}" 2>&1 | tee "${log_file}"
    
    echo ""
    echo "Exp6 completed for ${dataset_name}"
    echo "Results saved to: ${output_dir}_${dataset_name//_/-}"
    echo "=========================================="
    echo ""
}

# Main execution
echo "=========================================="
echo "Multi-Dataset Exp5/Exp6 Experiments (LONG-ANSWER DATASETS)"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Base output dir: ${BASE_OUTPUT_DIR}"
echo "Batch size (exp5): ${BATCH_SIZE} (with auto-adjustment)"
echo "Batch size (exp6): 1 (FIXED for latency measurement)"
echo "Num samples (exp6): ${NUM_SAMPLES}"
echo "Sampling strategy: ${SAMPLING_STRATEGY}"
echo "Max new tokens: ${MAX_NEW_TOKENS} (for long answers - larger than short-answer VQA)"
echo "Number of GPUs: ${NUM_GPUS} (auto-detected, override with NUM_GPUS_OVERRIDE)"
echo ""
echo "Note: Results will be saved in dataset-specific subdirectories:"
echo "  - Exp5: ${BASE_OUTPUT_DIR}/exp5_accuracy_long_<dataset_name>"
echo "  - Exp6: ${BASE_OUTPUT_DIR}/exp6_latency_long_<dataset_name>"
echo ""
echo "⚠️  IMPORTANT: These datasets require longer answers, so max_new_tokens=${MAX_NEW_TOKENS}"
echo "   (compared to 16 for short-answer VQA datasets)"
echo "=========================================="
echo ""

# Parse command line arguments
EXPERIMENT_TYPE="${1:-both}"  # "exp5", "exp6", or "both"
SPECIFIC_DATASET="${2:-}"     # Optional: run only specific dataset

for dataset_config in "${DATASETS[@]}"; do
    IFS=':' read -r dataset_name split experiment <<< "$dataset_config"
    
    # Skip if specific dataset is requested and doesn't match
    if [ -n "${SPECIFIC_DATASET}" ] && [ "${dataset_name}" != "${SPECIFIC_DATASET}" ]; then
        continue
    fi
    
    # Determine which experiments to run
    # If EXPERIMENT_TYPE is "exp5", only run exp5
    # If EXPERIMENT_TYPE is "exp6", only run exp6
    # If EXPERIMENT_TYPE is "both", run based on experiment field in dataset config
    if [ "${EXPERIMENT_TYPE}" == "exp5" ]; then
        run_exp5 "${dataset_name}" "${split}"
    elif [ "${EXPERIMENT_TYPE}" == "exp6" ]; then
        run_exp6 "${dataset_name}" "${split}"
    elif [ "${EXPERIMENT_TYPE}" == "both" ]; then
        # Run based on experiment field in dataset config
        if [ "${experiment}" == "exp5" ] || [ "${experiment}" == "both" ]; then
            run_exp5 "${dataset_name}" "${split}"
        fi
        if [ "${experiment}" == "exp6" ] || [ "${experiment}" == "both" ]; then
            run_exp6 "${dataset_name}" "${split}"
        fi
    fi
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results saved in dataset-specific directories under:"
echo "  ${BASE_OUTPUT_DIR}/"
echo ""
echo "Exp5 results: ${BASE_OUTPUT_DIR}/exp5_accuracy_long_<dataset_name>/"
echo "Exp6 results: ${BASE_OUTPUT_DIR}/exp6_latency_long_<dataset_name>/"
echo ""
echo "Note: These experiments used max_new_tokens=${MAX_NEW_TOKENS} for long answers"
echo "=========================================="
