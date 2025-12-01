#!/bin/bash
# Run Experiment 4: Language Tokens vs Latency
# Usage: bash experiments/motivate/run_exp4.sh [GPU_ID] [--num_samples N] [--max_new_tokens_list ...]

export CUDA_VISIBLE_DEVICES=${1:-0}
shift # Remove the first argument (GPU_ID) from $@
echo "Running Exp 4: Language Tokens vs Latency on GPU $CUDA_VISIBLE_DEVICES..."

# Configuration
MODEL_PATH="checkpoints"
DATASET="coco_2014_vqa"
SPLIT="validation"
NUM_SAMPLES=50
OUTPUT_DIR="./results/motivation/exp4-50samples"
DEVICE="cuda"
# MAX_NEW_TOKENS_LIST=(8 16 32)
MAX_NEW_TOKENS_LIST=(8 16 32 64 128 256 512 1024 2048 4096)

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_samples)
            NUM_SAMPLES="$2"
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
        --max_new_tokens_list)
            shift
            MAX_NEW_TOKENS_LIST=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                MAX_NEW_TOKENS_LIST+=("$1")
                shift
            done
            ;;
        *)
            shift
            ;;
    esac
done

echo "=========================================="
echo "Experiment 4: Language Tokens vs Latency"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET ($SPLIT)"
echo "Samples: $NUM_SAMPLES"
echo "Max New Tokens: ${MAX_NEW_TOKENS_LIST[*]}"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python experiments/motivate/exp4_language_tokens_vs_latency.py \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --max_new_tokens_list "${MAX_NEW_TOKENS_LIST[@]}"

echo ""
echo "Exp 4 completed! Results saved to $OUTPUT_DIR"

