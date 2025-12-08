#!/bin/bash
# Script to run exp5 (accuracy) on multiple datasets
# Datasets: TextVQA, OK-VQA, ScienceQA, ST-VQA, DocVQA, TallyQA

set -e

# Default values
MODEL_PATH="${MODEL_PATH:-checkpoints}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./results/profiling}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-balanced}"

# Dataset list with split information
declare -A DATASET_SPLITS=(
    ["text_vqa"]="validation"
    ["okvqa"]="validation"
    ["science_qa_img"]="validation"
    ["st_qa"]="validation"
    ["doc_qa"]="validation"
    ["tally_qa"]="test"  # TallyQA uses test set instead of validation
)

# Parse command line arguments
DATASET_NAME="${1:-}"  # Optional: run only specific dataset

if [ -z "${DATASET_NAME}" ]; then
    # Run all datasets
    DATASETS=("text_vqa" "okvqa" "science_qa_img" "st_qa" "doc_qa" "tally_qa")
else
    # Run specific dataset
    if [ -z "${DATASET_SPLITS[${DATASET_NAME}]}" ]; then
        echo "Error: Unknown dataset '${DATASET_NAME}'"
        echo "Available datasets: text_vqa, okvqa, science_qa_img, st_qa, doc_qa, tally_qa"
        exit 1
    fi
    DATASETS=("${DATASET_NAME}")
fi

echo "=========================================="
echo "Exp5 Accuracy: Multi-Dataset Experiments"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Base output dir: ${BASE_OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Sampling strategy: ${SAMPLING_STRATEGY}"
echo "Datasets to run: ${DATASETS[*]}"
echo "=========================================="
echo ""

for dataset_name in "${DATASETS[@]}"; do
    split="${DATASET_SPLITS[${dataset_name}]}"
    output_dir="${BASE_OUTPUT_DIR}/exp5_accuracy"
    
    echo "=========================================="
    echo "Running Exp5 on ${dataset_name}"
    echo "=========================================="
    echo "Dataset: ${dataset_name}"
    echo "Split: ${split}"
    echo "Output dir: ${output_dir}_${dataset_name//_/-}"
    echo ""
    
    torchrun --nproc-per-node=4 experiments/profiling/knob5_combined/exp5_accuracy.py \
        --model_path "${MODEL_PATH}" \
        --output_dir "${output_dir}" \
        --dataset_name "${dataset_name}" \
        --split "${split}" \
        --batch_size "${BATCH_SIZE}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --sampling_strategy "${SAMPLING_STRATEGY}" \
        --auto_adjust_batch_size
    
    echo ""
    echo "Exp5 completed for ${dataset_name}"
    echo "=========================================="
    echo ""
done

echo ""
echo "=========================================="
echo "All Exp5 experiments completed!"
echo "Results saved to: ${BASE_OUTPUT_DIR}/exp5_accuracy_<dataset>"
echo "=========================================="

