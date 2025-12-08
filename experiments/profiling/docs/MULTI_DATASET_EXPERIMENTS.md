# Multi-Dataset Experiments for Exp5 and Exp6

## Overview

This document describes how to run exp5 (accuracy) and exp6 (latency) experiments on multiple datasets.

## Supported Datasets

| Dataset | Dataset Name | Split | Evaluation Metric | Status |
|---------|-------------|-------|-------------------|--------|
| TextVQA | `text_vqa` | validation | `vqa_score` | ✅ Ready |
| OK-VQA | `okvqa` | validation | `vqa_score` | ✅ Ready |
| ScienceQA | `science_qa_img` | validation | `mc` (multiple choice) | ✅ Ready |
| ST-VQA | `st_qa` | validation | `ansl_em` | ✅ Ready |
| DocVQA | `doc_qa` | validation | `ansl_em` | ✅ Ready |
| TallyQA | `tally_qa` | test | `em` (exact match) | ✅ Ready |

**Note**: TallyQA uses `test` split instead of `validation` because it doesn't have a validation split.

## Changes Made

### 1. Extended `BaseExperiment.compute_accuracy()`

Added support for multiple evaluation metrics:
- `vqa_score`: Standard VQA evaluation (TextVQA, OK-VQA)
- `mc`: Multiple choice evaluation (ScienceQA)
- `em`: Exact match evaluation (TallyQA)
- `ansl_em`: ANLS + EM evaluation (ST-VQA, DocVQA)

### 2. Added `get_metric_for_dataset()` Function

Automatically selects the correct evaluation metric based on dataset name.

### 3. Modified Exp5 and Exp6

- Both experiments now accept `dataset_name` parameter
- Output directories are automatically adjusted to include dataset suffix (e.g., `exp5_accuracy_text-vqa`)
- Automatic metric selection based on dataset
- Automatic split selection (test for TallyQA, validation for others)

### 4. Result Format

Results are saved with the same detailed format as before:
- **Dataset-level results**: Summary statistics for each configuration
- **Per-sample results**: Detailed results for each individual sample including:
  - For VQA-style datasets: `sample_id`, `score`, `pred`, `answers`
  - For MC datasets (ScienceQA): `sample_id`, `score`, `pred`, `answer_idx`, `predicted_idx`, `options`
  - For other datasets: Similar format with appropriate fields

## Usage

### Option 1: Run All Datasets (Recommended)

```bash
# Run both exp5 and exp6 on all datasets
bash experiments/profiling/run_exp5_exp6_multi_datasets.sh

# Run only exp5 on all datasets
bash experiments/profiling/run_exp5_exp6_multi_datasets.sh exp5

# Run only exp6 on all datasets
bash experiments/profiling/run_exp5_exp6_multi_datasets.sh exp6
```

### Option 2: Run Specific Dataset

```bash
# Run exp5 on TextVQA only
bash experiments/profiling/run_exp5_multi_datasets.sh text_vqa

# Run exp6 on ScienceQA only
bash experiments/profiling/run_exp6_multi_datasets.sh science_qa_img
```

### Option 3: Run Individual Experiments Directly

```bash
# Exp5 on TextVQA
torchrun --nproc-per-node=4 experiments/profiling/knob5_combined/exp5_accuracy.py \
    --model_path checkpoints \
    --output_dir ./results/profiling/exp5_accuracy \
    --dataset_name text_vqa \
    --split validation \
    --batch_size 8 \
    --sampling_strategy balanced \
    --auto_adjust_batch_size

# Exp6 on ScienceQA
torchrun --nproc-per-node=4 experiments/profiling/knob5_combined/exp6_accuracy.py \
    --model_path checkpoints \
    --output_dir ./results/profiling/exp6_latency \
    --dataset_name science_qa_img \
    --split validation \
    --num_samples 5000 \
    --sampling_strategy balanced
```

## Output Directory Structure

Results are saved in separate directories for each dataset to avoid confusion:

```
results/profiling/
├── exp5_accuracy/                    # Original VQA v2 results
├── exp5_accuracy_text-vqa/           # TextVQA results
├── exp5_accuracy_okvqa/              # OK-VQA results
├── exp5_accuracy_science-qa-img/    # ScienceQA results
├── exp5_accuracy_st-qa/              # ST-VQA results
├── exp5_accuracy_doc-qa/             # DocVQA results
├── exp5_accuracy_tally-qa/           # TallyQA results
├── exp6_latency/                     # Original VQA v2 results
├── exp6_latency_text-vqa/            # TextVQA results
├── exp6_latency_okvqa/               # OK-VQA results
├── exp6_latency_science-qa-img/      # ScienceQA results
├── exp6_latency_st-qa/               # ST-VQA results
├── exp6_latency_doc-qa/              # DocVQA results
└── exp6_latency_tally-qa/            # TallyQA results
```

## Result File Format

### Exp5 Results

Each configuration generates:
- Individual file: `exp5_accuracy_results_crops{X}_topk{Y}_blocks{Z}_rank{R}.json`
- Merged file: `exp5_accuracy_results.json` (on rank 0)

Each file contains:
```json
{
  "summary": [
    {
      "max_crops": 2,
      "top_k": 4,
      "num_active_blocks": 12,
      "accuracy": 0.3839,
      "num_samples": 53589,
      "std": 0.4679,
      "duration_seconds": 1806.0,
      ...
    }
  ],
  "all_samples": [
    {
      "sample_id": 0,
      "score": 1.0,
      "pred": "down",
      "answers": ["down", "down", ...]
    },
    ...
  ],
  "config": {
    "dataset_name": "text_vqa",
    "split": "validation",
    ...
  }
}
```

### Exp6 Results

Each configuration generates:
- Individual file: `exp6_latency_crops{X}_topk{Y}_blocks{Z}_rank{R}.json`
- Merged file: `exp6_latency_results.json` (on rank 0)

Each file contains:
```json
{
  "summary": [
    {
      "max_crops": 2,
      "top_k": 4,
      "num_active_blocks": 12,
      "accuracy": 0.6155,
      "latency_total_ms": {
        "mean": 324.62,
        "P50": 292.67,
        ...
      },
      "latency_prefill_ms": {...},
      "latency_decode_ms": {...},
      ...
    }
  ],
  "all_samples": [
    {
      "sample_id": 0,
      "max_crops": 2,
      "top_k": 4,
      "num_active_blocks": 12,
      "T_total_ms": 324.62,
      "T_prefill_ms": 91.53,
      "T_decode_ms": 233.09,
      "num_vision_tokens": 1152,
      "num_language_tokens": 50,
      "num_output_tokens": 5,
      "num_total_tokens": 1207,
      "accuracy": 1.0
    },
    ...
  ],
  "config": {
    "dataset_name": "text_vqa",
    "split": "validation",
    ...
  }
}
```

## Special Notes

### ScienceQA (Multiple Choice)

- Uses `mc` metric
- Per-sample results include `answer_idx`, `predicted_idx`, and `options`
- The `select_mc_option()` function handles prediction parsing with fallback heuristics

### ST-VQA and DocVQA

- Uses `ansl_em` metric (ANLS as primary)
- Requires manual dataset download for ST-VQA (see dataset documentation)

### TallyQA

- Uses `test` split (no validation split available)
- Uses `em` (exact match) metric
- Data format uses `message_list` with `answer` field in each message

## Environment Variables

You can customize the experiments using environment variables:

```bash
export MODEL_PATH="checkpoints"
export BASE_OUTPUT_DIR="./results/profiling"
export BATCH_SIZE=8
export MAX_NEW_TOKENS=16
export NUM_SAMPLES=5000
export SAMPLING_STRATEGY="balanced"

bash experiments/profiling/run_exp5_exp6_multi_datasets.sh
```

## Troubleshooting

### ScienceQA: Missing answer_idx or options

If you see warnings about missing `answer_idx` or `options`, check that:
1. The dataset is loaded correctly
2. The data formatter is extracting these fields to metadata (should be automatic)

### ST-VQA: Dataset not found

ST-VQA requires manual download:
1. Download from https://rrc.cvc.uab.es/?ch=11
2. Extract to the expected location (check `ST_QA_SRC` in code)

### TallyQA: No validation split

TallyQA only has `train` and `test` splits. The code automatically uses `test` when `validation` is requested.

## Next Steps

After running experiments, you can:
1. Analyze results using the same analysis scripts (with dataset-specific paths)
2. Generate Pareto frontier plots for each dataset
3. Compare results across datasets

