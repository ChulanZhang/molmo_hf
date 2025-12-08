#!/bin/bash
# Test script to verify all datasets work with exp5 and exp6
# Uses a single simple configuration and 500 samples for quick testing

set -e

# Default values
MODEL_PATH="${MODEL_PATH:-checkpoints}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./results/profiling/test}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

# Dataset list with split information
declare -A DATASET_SPLITS=(
    ["text_vqa"]="validation"
    ["okvqa"]="validation"
    ["science_qa_img"]="validation"
    ["st_qa"]="validation"
    ["doc_qa"]="validation"
    ["tally_qa"]="test"  # TallyQA uses test set instead of validation
)

# Use a simple fixed configuration for testing
MAX_CROPS=7
TOP_K=8
NUM_ACTIVE_BLOCKS=16

echo "=========================================="
echo "Multi-Dataset Test: Exp5 & Exp6"
echo "=========================================="
echo "Model path: ${MODEL_PATH}"
echo "Output dir: ${BASE_OUTPUT_DIR}"
echo "Test configuration: max_crops=${MAX_CROPS}, top_k=${TOP_K}, num_active_blocks=${NUM_ACTIVE_BLOCKS}"
echo "Num samples: ${NUM_SAMPLES} (for both exp5 and exp6)"
echo "=========================================="
echo ""

# Parse command line arguments
DATASET_NAME="${1:-}"  # Optional: test only specific dataset
EXPERIMENT_TYPE="${2:-both}"  # "exp5", "exp6", or "both"

if [ -z "${DATASET_NAME}" ]; then
    # Test all datasets
    DATASETS=("text_vqa" "okvqa" "science_qa_img" "st_qa" "tally_qa")
else
    # Test specific dataset
    if [ -z "${DATASET_SPLITS[${DATASET_NAME}]}" ]; then
        echo "Error: Unknown dataset '${DATASET_NAME}'"
        echo "Available datasets: text_vqa, okvqa, science_qa_img, st_qa, doc_qa, tally_qa"
        exit 1
    fi
    DATASETS=("${DATASET_NAME}")
fi

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_DATASETS=()

for dataset_name in "${DATASETS[@]}"; do
    split="${DATASET_SPLITS[${dataset_name}]}"
    
    echo "=========================================="
    echo "Testing ${dataset_name}"
    echo "=========================================="
    echo "Dataset: ${dataset_name}"
    echo "Split: ${split}"
    echo ""
    
    # Test Exp5 (accuracy)
    if [ "${EXPERIMENT_TYPE}" == "exp5" ] || [ "${EXPERIMENT_TYPE}" == "both" ]; then
        echo "Testing Exp5 (accuracy)..."
        output_dir="${BASE_OUTPUT_DIR}/exp5_accuracy"
        
        if torchrun --nproc-per-node=1 experiments/profiling/knob5_combined/exp5_accuracy.py \
            --model_path "${MODEL_PATH}" \
            --output_dir "${output_dir}" \
            --dataset_name "${dataset_name}" \
            --split "${split}" \
            --batch_size "${BATCH_SIZE}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --max_crops "${MAX_CROPS}" \
            --top_k "${TOP_K}" \
            --num_active_blocks "${NUM_ACTIVE_BLOCKS}" \
            --sampling_strategy "full" \
            --num_samples "${NUM_SAMPLES}" \
            --auto_adjust_batch_size 2>&1 | tee /tmp/exp5_test_${dataset_name}.log; then
            echo "✅ Exp5 test passed for ${dataset_name}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "❌ Exp5 test failed for ${dataset_name}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            FAILED_DATASETS+=("${dataset_name} (exp5)")
        fi
        echo ""
    fi
    
    # Test Exp6 (latency)
    if [ "${EXPERIMENT_TYPE}" == "exp6" ] || [ "${EXPERIMENT_TYPE}" == "both" ]; then
        echo "Testing Exp6 (latency)..."
        output_dir="${BASE_OUTPUT_DIR}/exp6_latency"
        
        if torchrun --nproc-per-node=1 experiments/profiling/knob5_combined/exp6_accuracy.py \
            --model_path "${MODEL_PATH}" \
            --output_dir "${output_dir}" \
            --dataset_name "${dataset_name}" \
            --split "${split}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --num_samples "${NUM_SAMPLES}" \
            --max_crops "${MAX_CROPS}" \
            --top_k "${TOP_K}" \
            --num_active_blocks "${NUM_ACTIVE_BLOCKS}" \
            --sampling_strategy "full" 2>&1 | tee /tmp/exp6_test_${dataset_name}.log; then
            echo "✅ Exp6 test passed for ${dataset_name}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "❌ Exp6 test failed for ${dataset_name}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            FAILED_DATASETS+=("${dataset_name} (exp6)")
        fi
        echo ""
    fi
    
    echo "=========================================="
    echo ""
done

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total tests: $((SUCCESS_COUNT + FAIL_COUNT))"
echo "✅ Passed: ${SUCCESS_COUNT}"
echo "❌ Failed: ${FAIL_COUNT}"

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo ""
    echo "Failed datasets:"
    for failed in "${FAILED_DATASETS[@]}"; do
        echo "  - ${failed}"
    done
    echo ""
    echo "Check logs in /tmp/exp*_test_*.log for details"
    exit 1
else
    echo ""
    echo "All tests passed! ✅"
    exit 0
fi

