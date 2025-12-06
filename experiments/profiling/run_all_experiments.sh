#!/bin/bash
# Run All Profiling Experiments
# Usage: bash experiments/profiling/run_all_experiments.sh [GPU_ID] [--num_samples N]

export CUDA_VISIBLE_DEVICES=${1:-0}
shift # Remove the first argument (GPU_ID) from $@
echo "Running All Profiling Experiments on GPU $CUDA_VISIBLE_DEVICES..."

# Configuration
MODEL_PATH="checkpoints"
NUM_SAMPLES=50

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
        *)
            shift
            ;;
    esac
done

echo "=========================================="
echo "Running All Profiling Experiments"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Samples per experiment: $NUM_SAMPLES"
echo "=========================================="
echo ""

# # Experiment 1: Context Scaling
# echo ">>> Starting Exp 1: Context Scaling..."
# bash experiments/profiling/run_exp1_context_scaling.sh $CUDA_VISIBLE_DEVICES \
#     --model_path "$MODEL_PATH" \
#     --num_samples "$NUM_SAMPLES"
# if [ $? -ne 0 ]; then
#     echo "Exp 1 failed!"
#     exit 1
# fi
# echo ""

# Experiment 2: MoE Top-K
echo ">>> Starting Exp 2: MoE Top-K Analysis..."
bash experiments/profiling/run_exp2_moe_topk.sh $CUDA_VISIBLE_DEVICES \
    --model_path "$MODEL_PATH" \
    --num_samples "$NUM_SAMPLES"
if [ $? -ne 0 ]; then
    echo "Exp 2 failed!"
    exit 1
fi
echo ""

# Experiment 3: Transformer Blocks Mask
echo ">>> Starting Exp 3: Transformer Blocks Mask..."
bash experiments/profiling/run_exp3_transformer_blocks_mask.sh $CUDA_VISIBLE_DEVICES \
    --model_path "$MODEL_PATH" \
    --num_samples "$NUM_SAMPLES"
if [ $? -ne 0 ]; then
    echo "Exp 3 failed!"
    exit 1
fi
echo ""

# # Experiment 4: Output Tokens Scaling
# echo ">>> Starting Exp 4: Output Tokens Scaling..."
# bash experiments/profiling/run_exp4_output_tokens.sh $CUDA_VISIBLE_DEVICES \
#     --model_path "$MODEL_PATH" \
#     --num_samples "$NUM_SAMPLES"
# if [ $? -ne 0 ]; then
#     echo "Exp 4 failed!"
#     exit 1
# fi
# echo ""

echo "=========================================="
echo "All Profiling Experiments Completed!"
echo "=========================================="
echo "Results saved to:"
# echo "  - Exp 1: ./results/profiling/context_scaling"
echo "  - Exp 2: ./results/profiling/moe_topk"
echo "  - Exp 3: ./results/profiling/transformer_blocks_mask"
# echo "  - Exp 4: ./results/profiling/output_tokens"
echo "=========================================="

