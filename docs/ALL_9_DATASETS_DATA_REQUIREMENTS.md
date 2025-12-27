# Data Requirements for All 9 Datasets

This document describes the data requirements for the nine datasets used in the exp5 and exp6 experiments.

## Dataset List

As configured in `experiments/profiling/run_exp5_exp6_multi_datasets.sh`, the experiments use:

1. `coco_2014_vqa:validation:exp6`
2. `text_vqa:validation:both`
3. `okvqa:validation:both`
4. `science_qa_img:validation:both`
5. `st_qa:validation:both`
6. `doc_qa:validation:both`
7. `tally_qa:test:both`
8. `mmmu:validation:both`
9. `coco_caption:validation:both`

## Dataset Categories

### HF-cache-only datasets (5)

These load entirely from the HuggingFace datasets cache; no raw files needed:

1. **text_vqa** (`facebook/textvqa`)
   - HF cache: `{HF_HOME}/datasets/facebook___textvqa/`
   - Size: ~27MB

2. **okvqa** (`HuggingFaceM4/OK-VQA`)
   - HF cache: `{HF_HOME}/datasets/HuggingFaceM4___ok-vqa/`
   - Size: ~10MB

3. **science_qa_img** (`derek-thomas/ScienceQA`)
   - HF cache: `{HF_HOME}/datasets/derek-thomas___science_qa/`
   - Size: ~672MB

4. **doc_qa** (`HuggingFaceM4/DocumentVQA`)
   - HF cache: `{HF_HOME}/datasets/HuggingFaceM4___document_vqa/`
   - Size: ~34GB

5. **mmmu** (`MMMU/MMMU`)
   - HF cache: `{HF_HOME}/datasets/MMMU___mmmu/`
   - Size: ~3.5GB

### Datasets requiring raw data (4)

These need raw files (images, JSON, etc.):

1. **coco_2014_vqa** (`Vqa2`)
   - HF cache: `{HF_HOME}/datasets/vqa_v2/` (~152MB)
   - Raw data: `{MOLMO_DATA_DIR}/torch_datasets/downloads/val2014/` (~20GB)
   - Needs: COCO 2014 validation images

2. **coco_caption** (`CocoCaption`)
   - HF cache: `{HF_HOME}/datasets/coco_caption/` (~57MB)
   - Raw data: `{MOLMO_DATA_DIR}/torch_datasets/downloads/val2014/` (~20GB)
   - Needs: COCO 2014 validation images

3. **tally_qa** (`TallyQa`)
   - HF cache: `{HF_HOME}/datasets/tally_qa/` (~56MB)
   - Raw data:
     - `{MOLMO_DATA_DIR}/torch_datasets/downloads/train2014/` (~41GB)
     - `{MOLMO_DATA_DIR}/torch_datasets/downloads/val2014/` (~20GB)
   - Needs: COCO 2014 train and validation images

4. **st_qa** (`SceneTextQa`)
   - Raw data: `{MOLMO_DATA_DIR}/torch_datasets/scene-text/`
   - Needs:
     - `train_task_3.json`
     - `test_task_3.json`
     - image files
   - Note: must be downloaded manually; not hosted on HF

## Must-sync Data

### 1. HF caches (all datasets)

Location: `{HF_HOME}/datasets/`

Sync these directories:
- `coco_caption/` (~57MB)
- `vqa_v2/` (~152MB, coco_2014_vqa)
- `tally_qa/` (~56MB)
- `facebook___textvqa/` (~27MB, text_vqa)
- `HuggingFaceM4___ok-vqa/` (~10MB, okvqa)
- `derek-thomas___science_qa/` (~672MB, science_qa_img)
- `HuggingFaceM4___document_vqa/` (~34GB, doc_qa)
- `MMMU___mmmu/` (~3.5GB, mmmu)

### 2. Raw data (datasets needing images)

Location: `{MOLMO_DATA_DIR}/torch_datasets/`

Sync:
- `downloads/val2014/` (~20GB) - required by coco_2014_vqa, coco_caption
- `downloads/train2014/` (~41GB) - required by tally_qa
- `scene-text/` - required by st_qa (JSON + images)

Do not sync:
- `downloads/test2015/` - not used by these datasets
- `downloads/*.zip` - not needed if extracted directories exist
- Other dataset directories (e.g., pixmo_datasets, info_qa)

## Sync Commands

### Full sync script

Use `scripts/sync_all_9_datasets.sh` for a full sync:

```bash
./scripts/sync_all_9_datasets.sh
```

### Manual sync

#### 1. HF caches

```bash
mkdir -p /anvil/projects/x-cis250705/data/vlm/huggingface/datasets

# Required caches
rsync -avz --progress \
  /anvil/scratch/x-pwang1/data/vlm/huggingface/datasets/coco_caption/ \
  /anvil/projects/x-cis250705/data/vlm/huggingface/datasets/coco_caption/

rsync -avz --progress \
  /anvil/scratch/x-pwang1/data/vlm/huggingface/datasets/vqa_v2/ \
  /anvil/projects/x-cis250705/data/vlm/huggingface/datasets/vqa_v2/

rsync -avz --progress \
  /anvil/scratch/x-pwang1/data/vlm/huggingface/datasets/tally_qa/ \
  /anvil/projects/x-cis250705/data/vlm/huggingface/datasets/tally_qa/

rsync -avz --progress \
  /anvil/scratch/x-pwang1/data/vlm/huggingface/datasets/facebook___textvqa/ \
  /anvil/projects/x-cis250705/data/vlm/huggingface/datasets/facebook___textvqa/
```

#### 2. Raw data

```bash
mkdir -p /anvil/projects/x-cis250705/data/vlm/molmo/torch_datasets/downloads

# val2014 (required by coco_2014_vqa, coco_caption)
rsync -avz --progress \
  /anvil/scratch/x-pwang1/data/vlm/molmo/torch_datasets/downloads/val2014/ \
  /anvil/projects/x-cis250705/data/vlm/molmo/torch_datasets/downloads/val2014/

# train2014 (required by tally_qa)
rsync -avz --progress \
  /anvil/scratch/x-pwang1/data/vlm/molmo/torch_datasets/downloads/train2014/ \
  /anvil/projects/x-cis250705/data/vlm/molmo/torch_datasets/downloads/train2014/

# scene-text (required by st_qa)
rsync -avz --progress \
  /anvil/scratch/x-pwang1/data/vlm/molmo/torch_datasets/scene-text/ \
  /anvil/projects/x-cis250705/data/vlm/molmo/torch_datasets/scene-text/
```

## Size Estimates

### Must-sync

- **HF caches**: ~38GB (coco_caption 57MB + vqa_v2 152MB + tally_qa 56MB + text_vqa 27MB + okvqa 10MB + science_qa 672MB + document_vqa 34GB + mmmu 3.5GB)
- **Raw data**: ~61GB (val2014 20GB + train2014 41GB + scene-text unknown size)
- **Total**: ~99GB

### Optional sync

- **Other HF caches**: auto-downloads on first run if missing (hundreds of MB to a few GB)
- **test2015**: not needed (~53GB)
- **Other datasets**: not needed (~355GB)

## Verification

Run the analysis script to check data integrity:

```bash
source activate_env.sh
python scripts/analyze_all_datasets_data_requirements.py
```

## Summary

For exp5 and exp6 on all nine datasets, you must sync:

1. ✅ HF caches under `{HF_HOME}/datasets/` (~292MB)
2. ✅ COCO images: `{MOLMO_DATA_DIR}/torch_datasets/downloads/val2014/` (~20GB)
3. ✅ COCO images: `{MOLMO_DATA_DIR}/torch_datasets/downloads/train2014/` (~41GB)
4. ✅ Scene-text: `{MOLMO_DATA_DIR}/torch_datasets/scene-text/` (needed by st_qa)

Not required:
- ❌ test2015 (~53GB)
- ❌ Other dataset directories (~355GB)
- ❌ Other HF caches (auto-download if missing)

Space saved if you only sync required data: ~408GB

