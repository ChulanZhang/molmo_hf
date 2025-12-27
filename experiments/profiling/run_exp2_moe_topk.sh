#!/bin/bash
# Run Profiling Experiment 2: MoE Top-K Analysis
# Usage: bash experiments/profiling/run_exp2_moe_topk.sh [GPU_ID] [--num_samples N] [--top_k_values K1 K2 ...]

export CUDA_VISIBLE_DEVICES=${1:-0}
shift # Remove the first argument (GPU_ID) from $@
echo "Running Profiling Exp 2: MoE Top-K Analysis on GPU $CUDA_VISIBLE_DEVICES..."

# Configuration
MODEL_PATH="checkpoints"
OUTPUT_DIR="./results/profiling/moe_topk"
NUM_SAMPLES=100
TOP_K_VALUES=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64)

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
        --top_k_values)
            shift # Remove --top_k_values
            TOP_K_VALUES=()
            # Collect all values until next option or end
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                TOP_K_VALUES+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash experiments/profiling/run_exp2_moe_topk.sh [GPU_ID] [--num_samples N] [--top_k_values K1 K2 ...] [--model_path PATH] [--output_dir DIR]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Profiling Experiment 2: MoE Top-K Analysis"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Samples: $NUM_SAMPLES"
echo "Top-K values: ${TOP_K_VALUES[*]}"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python experiments/profiling/knob2_topk/exp_moe_topk.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --top_k_values "${TOP_K_VALUES[@]}"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Exp 2 completed! Results saved to $OUTPUT_DIR"
else
    echo ""
    echo "Exp 2 failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

