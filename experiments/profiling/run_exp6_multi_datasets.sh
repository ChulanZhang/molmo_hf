#!/bin/bash
# Script to run exp6 (latency) on multiple datasets
# Datasets: TextVQA, OK-VQA, ScienceQA, ST-VQA, DocVQA, TallyQA

set -e

# Default values
MODEL_PATH="${MODEL_PATH:-checkpoints}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./results/profiling}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
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
echo "Exp6 Latency: Multi-Dataset Experiments"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Base output dir: ${BASE_OUTPUT_DIR}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Sampling strategy: ${SAMPLING_STRATEGY}"
echo "Datasets to run: ${DATASETS[*]}"
echo "=========================================="
echo ""

for dataset_name in "${DATASETS[@]}"; do
    split="${DATASET_SPLITS[${dataset_name}]}"
    output_dir="${BASE_OUTPUT_DIR}/exp6_latency"
    
    echo "=========================================="
    echo "Running Exp6 on ${dataset_name}"
    echo "=========================================="
    echo "Dataset: ${dataset_name}"
    echo "Split: ${split}"
    echo "Output dir: ${output_dir}_${dataset_name//_/-}"
    echo ""
    
    torchrun --nproc-per-node=4 experiments/profiling/knob5_combined/exp6_accuracy.py \
        --model_path "${MODEL_PATH}" \
        --output_dir "${output_dir}" \
        --dataset_name "${dataset_name}" \
        --split "${split}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --num_samples "${NUM_SAMPLES}" \
        --sampling_strategy "${SAMPLING_STRATEGY}"
    
    echo ""
    echo "Exp6 completed for ${dataset_name}"
    echo "=========================================="
    echo ""
done

echo ""
echo "=========================================="
echo "All Exp6 experiments completed!"
echo "Results saved to: ${BASE_OUTPUT_DIR}/exp6_latency_<dataset>"
echo "=========================================="

