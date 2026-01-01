# Layer Importance Analysis

## Overview

This document describes the layer importance analysis experiments for transformer block pruning in the Molmo VLM model. The analysis measures the impact of removing individual transformer blocks on model accuracy to determine block importance scores.

## Key Concepts

### Sensitivity Analysis

Sensitivity analysis measures the accuracy drop when removing individual transformer blocks. The importance score for each block is defined as:

```
importance_score[block_i] = accuracy_baseline - accuracy_without_block_i
```

Higher scores indicate more important blocks (removing them causes larger accuracy drops).

### Dataset Selection

**Recommendation: Use training set for sensitivity analysis**

**Rationale:**
1. **Larger sample size**: Training sets typically have more samples, providing more reliable statistics
2. **Structural property**: Block importance is a structural property of the model, should be consistent across data splits
3. **Relative importance**: We care about relative importance (which blocks are more important), not absolute accuracy

**Validation**: Use `compare_train_val_importance.py` to verify consistency between train and validation splits via Spearman correlation.

## Experiments

### 1. Single-Dataset Sensitivity Analysis

**Script**: `experiments/profiling/knob3_layers/exp3_accuracy_sensitivity_v2.py`

**Features:**
- Multi-dataset support (test on 3+ datasets to verify consistency)
- Beam search for block combination exploration
- Dynamic batch size optimization
- All 16 blocks included (no forced retention of first/last blocks)

**Usage:**
```bash
torchrun --nproc-per-node=4 experiments/profiling/knob3_layers/exp3_accuracy_sensitivity_v2.py \
    --dataset_name coco_2014_vqa \
    --split train \
    --batch_size 16 \
    --num_samples 5000 \
    --max_blocks_to_remove 0  # Only sensitivity analysis, skip beam search
```

**Output**: `layer_importance_scores.json` with block indices and importance scores.

### 2. Train vs Validation Comparison

**Script**: `experiments/profiling/knob3_layers/compare_train_val_importance.py`

**Purpose**: Verify that importance scores are consistent between train and validation splits.

**Usage:**
```bash
# Single dataset
python experiments/profiling/knob3_layers/compare_train_val_importance.py \
    --dataset_name coco_2014_vqa \
    --num_samples 5000

# Multiple datasets
python experiments/profiling/knob3_layers/compare_train_val_importance.py \
    --dataset_name coco_2014_vqa text_vqa okvqa

# All supported datasets
python experiments/profiling/knob3_layers/compare_train_val_importance.py \
    --dataset_name all
```

**Output**:
- JSON comparison results
- Visualization plots (side-by-side bar charts)
- Spearman correlation coefficient

**Interpretation**:
- Correlation > 0.9: ✅ Consistent, can use training set
- Correlation ≤ 0.9: ⚠️ Inconsistent, may need validation set

### 3. Beam Search for Block Combinations

**Purpose**: Explore block combinations considering inter-block dependencies.

**Algorithm**:
1. Step 1: Remove 1 block, test all 16 possibilities, keep top 3 with least impact
2. Step 2: For each top-3 config, remove another block, test remaining possibilities, keep top 3
3. Step 3-4: Continue until max 4 blocks removed

**Usage**:
```bash
torchrun --nproc-per-node=4 experiments/profiling/knob3_layers/exp3_accuracy_sensitivity_v2.py \
    --dataset_name coco_2014_vqa \
    --split train \
    --beam_width 3 \
    --max_blocks_to_remove 4
```

## Supported Datasets

The following datasets are supported for importance analysis:

- `coco_2014_vqa` - COCO VQA
- `text_vqa` - TextVQA
- `okvqa` - OK-VQA
- `science_qa_img` - ScienceQA (image version)
- `doc_qa` - DocQA
- `chart_qa` - ChartQA
- `info_qa` - InfoQA
- `plot_qa` - PlotQA
- `figure_qa` - FigureQA
- `dv_qa` - DVQA
- `mmmu` - MMMU
- `coco_caption` - COCO Captioning
- `tally_qa` - TallyQA
- `st_qa` - SceneTextQa

## Technical Details

### Batch Size Optimization

The experiment automatically optimizes batch size based on:
- Number of active blocks (fewer blocks = larger batch size possible)
- GPU memory availability
- Conservative scaling to avoid OOM errors

### Model Compatibility

**Note**: A monkey patch is applied to fix a shape mismatch issue in `modeling_molmoe.py`:
- Issue: `image_flat` needs to be flattened before `index_add_` operation
- Fix: Automatic flattening in `index_add_` when source is 3D and dim=0
- Location: Module-level patch in `exp3_accuracy_sensitivity_v2.py`

### Output Format

Importance scores are saved as JSON:
```json
{
  "0": 0.6015,
  "1": 0.0369,
  "2": 0.0185,
  ...
}
```

Where keys are block indices and values are importance scores (accuracy drops).

## References

- Original experiment: `exp3_accuracy_sensitivity.py`
- Multi-dataset runner: `run_exp3_multi_datasets.py`
- Comparison script: `compare_train_val_importance.py`

