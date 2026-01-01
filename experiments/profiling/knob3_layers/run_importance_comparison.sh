#!/bin/bash
# Script to run importance score comparison for multiple datasets
# Compares train vs validation importance scores using Spearman correlation

set -e

# Default values
MODEL_PATH="${MODEL_PATH:-checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/profiling/exp3_importance_comparison}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"  # Use 10K samples for comparison (faster)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

# Datasets to compare (recommended: start with 1-2 datasets)
DATASETS=(
    "coco_2014_vqa"
    "text_vqa"
)

echo "=========================================="
echo "Importance Score Comparison: Train vs Validation"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="
echo ""

# Parse command line arguments
SPECIFIC_DATASET="${1:-}"

if [ -n "${SPECIFIC_DATASET}" ]; then
    DATASETS=("${SPECIFIC_DATASET}")
fi

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_DATASETS=()

for dataset_name in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Comparing: ${dataset_name}"
    echo "=========================================="
    
    python experiments/profiling/knob3_layers/compare_train_val_importance.py \
        --dataset_name "${dataset_name}" \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_gpus "${NUM_GPUS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_samples "${NUM_SAMPLES}" \
        --max_new_tokens "${MAX_NEW_TOKENS}"
    
    if [ $? -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✅ Successfully compared ${dataset_name}"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_DATASETS+=("${dataset_name}")
        echo "❌ Failed to compare ${dataset_name}"
    fi
    
    echo ""
done

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Successfully compared: ${SUCCESS_COUNT} dataset(s)"
if [ ${FAIL_COUNT} -gt 0 ]; then
    echo "Failed: ${FAIL_COUNT} dataset(s)"
    echo "Failed datasets: ${FAILED_DATASETS[*]}"
fi
echo "=========================================="

# Check results
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "To view results, check:"
for dataset_name in "${DATASETS[@]}"; do
    result_file="${OUTPUT_DIR}/${dataset_name}/importance_comparison_${dataset_name}.json"
    if [ -f "${result_file}" ]; then
        echo "  - ${result_file}"
    fi
done

