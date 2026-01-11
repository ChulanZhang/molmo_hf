#!/bin/bash
# Prepare datasets for evaluation
# This script downloads and prepares datasets for adaptive inference evaluation

set -e

echo "=========================================="
echo "Preparing Datasets for Evaluation"
echo "=========================================="

# Set data directory (if not already set)
export MOLMO_DATA_DIR=${MOLMO_DATA_DIR:-"./data"}

# Create data directory
mkdir -p "$MOLMO_DATA_DIR"

# Function to download a dataset
download_dataset() {
    local dataset_name=$1
    echo ""
    echo "Downloading dataset: $dataset_name"
    echo "-----------------------------------"
    python scripts/download_data.py "$dataset_name" --n_procs 1
    echo "✓ $dataset_name downloaded"
}

# List of datasets to prepare
# Add or remove datasets as needed
DATASETS=(
    "textvqa"
    "okvqa"
    "coco_2014_vqa"
    "science_qa_img"
    "doc_qa"
    "st_qa"
    "tally_qa"
)

# Download datasets
for dataset in "${DATASETS[@]}"; do
    download_dataset "$dataset" || {
        echo "⚠ Warning: Failed to download $dataset"
        echo "  You may need to download it manually or check the dataset name"
    }
done

echo ""
echo "=========================================="
echo "Dataset Preparation Complete"
echo "=========================================="
echo ""
echo "Datasets are available in: $MOLMO_DATA_DIR"
echo ""
echo "You can now run evaluation with:"
echo "  python experiments/controller/evaluate_adaptive_inference.py \\"
echo "      --model_path checkpoints/molmo \\"
echo "      --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \\"
echo "      --dataset text_vqa --num_samples 100 --latency_budget 200.0"
echo ""

