#!/bin/bash
# Sync all required data for the 9 datasets
# Includes both HF caches and raw data files

set -e

SOURCE_MOLMO_DATA_DIR="/anvil/scratch/x-pwang1/data/vlm/molmo"
SOURCE_HF_HOME="/anvil/scratch/x-pwang1/data/vlm/huggingface"
TARGET_MOLMO_DATA_DIR="/anvil/projects/x-cis250705/data/vlm/molmo"
TARGET_HF_HOME="/anvil/projects/x-cis250705/data/vlm/huggingface"

echo "=========================================="
echo "Sync script for all 9 datasets"
echo "=========================================="
echo ""
echo "Source directories:"
echo "  MOLMO_DATA_DIR: ${SOURCE_MOLMO_DATA_DIR}"
echo "  HF_HOME: ${SOURCE_HF_HOME}"
echo ""
echo "Target directories:"
echo "  MOLMO_DATA_DIR: ${TARGET_MOLMO_DATA_DIR}"
echo "  HF_HOME: ${TARGET_HF_HOME}"
echo ""
echo "The following data will be synchronized:"
echo ""
echo "HF caches (8 datasets):"
echo "  1. coco_caption (~57MB)"
echo "  2. vqa_v2 (~152MB, coco_2014_vqa)"
echo "  3. tally_qa (~56MB)"
echo "  4. facebook___textvqa (~27MB, text_vqa)"
echo "  5. HuggingFaceM4___ok-vqa (~10MB, okvqa)"
echo "  6. derek-thomas___science_qa (~672MB, science_qa_img)"
echo "  7. HuggingFaceM4___document_vqa (~34GB, doc_qa)"
echo "  8. MMMU___mmmu (~3.5GB, mmmu)"
echo ""
echo "Raw data:"
echo "  1. downloads/val2014/ (~20GB, required by coco_2014_vqa, coco_caption)"
echo "  2. downloads/train2014/ (~41GB, required by tally_qa)"
echo "  3. scene-text/ (required by st_qa)"
echo ""
echo "Total size: ~38GB (HF caches) + 61GB (raw data) = ~99GB"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Sync canceled"
    exit 1
fi

echo ""
echo "Starting sync..."

# Create target directories
mkdir -p "${TARGET_HF_HOME}/datasets"
mkdir -p "${TARGET_MOLMO_DATA_DIR}/torch_datasets/downloads"
mkdir -p "${TARGET_MOLMO_DATA_DIR}/torch_datasets"

# 1. Sync HF caches
echo ""
echo "=========================================="
echo "1. Sync HF caches..."
echo "=========================================="

HF_CACHE_DATASETS=(
    "coco_caption"
    "vqa_v2"
    "tally_qa"
    "facebook___textvqa"
    "HuggingFaceM4___ok-vqa"
    "derek-thomas___science_qa"
    "HuggingFaceM4___document_vqa"
    "MMMU___mmmu"
)

for cache_name in "${HF_CACHE_DATASETS[@]}"; do
    source_cache="${SOURCE_HF_HOME}/datasets/${cache_name}"
    target_cache="${TARGET_HF_HOME}/datasets/${cache_name}"
    
    if [ -d "${source_cache}" ]; then
        echo ""
        echo "Syncing ${cache_name}..."
        rsync -avz --progress "${source_cache}/" "${target_cache}/"
        echo "✓ ${cache_name} synced"
    else
        echo ""
        echo "⚠️  ${cache_name} not found, skipping (may need first-run download)"
    fi
done

# 2. Sync raw data
echo ""
echo "=========================================="
echo "2. Sync raw data..."
echo "=========================================="

# 2.1 Sync val2014 (required by coco_2014_vqa, coco_caption)
echo ""
echo "Syncing val2014/ (required by coco_2014_vqa, coco_caption)..."
if [ -d "${SOURCE_MOLMO_DATA_DIR}/torch_datasets/downloads/val2014" ]; then
    rsync -avz --progress \
        "${SOURCE_MOLMO_DATA_DIR}/torch_datasets/downloads/val2014/" \
        "${TARGET_MOLMO_DATA_DIR}/torch_datasets/downloads/val2014/"
    echo "✓ val2014/ synced"
else
    echo "⚠️  val2014/ not found"
fi

# 2.2 Sync train2014 (required by tally_qa)
echo ""
echo "Syncing train2014/ (required by tally_qa)..."
if [ -d "${SOURCE_MOLMO_DATA_DIR}/torch_datasets/downloads/train2014" ]; then
    rsync -avz --progress \
        "${SOURCE_MOLMO_DATA_DIR}/torch_datasets/downloads/train2014/" \
        "${TARGET_MOLMO_DATA_DIR}/torch_datasets/downloads/train2014/"
    echo "✓ train2014/ synced"
else
    echo "⚠️  train2014/ not found"
fi

# 2.3 Sync scene-text (required by st_qa)
echo ""
echo "Syncing scene-text/ (required by st_qa)..."
if [ -d "${SOURCE_MOLMO_DATA_DIR}/torch_datasets/scene-text" ]; then
    rsync -avz --progress \
        "${SOURCE_MOLMO_DATA_DIR}/torch_datasets/scene-text/" \
        "${TARGET_MOLMO_DATA_DIR}/torch_datasets/scene-text/"
    echo "✓ scene-text/ synced"
else
    echo "⚠️  scene-text/ not found"
fi

echo ""
echo "=========================================="
echo "Sync finished!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update environment variables in activate_env.sh:"
echo "   export MOLMO_DATA_DIR=${TARGET_MOLMO_DATA_DIR}"
echo "   export HF_HOME=${TARGET_HF_HOME}"
echo ""
echo "2. Verify data integrity:"
echo "   source activate_env.sh"
echo "   python scripts/analyze_all_datasets_data_requirements.py"
echo ""
echo "Notes:"
echo "  - Some HF caches may be missing; first run will auto-download"
echo "  - If certain dataset caches are missing, they will auto-download from HuggingFace"
echo ""

