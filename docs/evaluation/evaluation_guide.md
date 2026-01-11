# Complete Evaluation Guide for Adaptive Inference

This guide provides a complete workflow for evaluating the adaptive inference engine using lmms-eval and direct dataset evaluation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Evaluation Methods](#evaluation-methods)
5. [Results Analysis](#results-analysis)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

- Trained model checkpoint
- Trained controller checkpoint (two-stage controller)
- CUDA-capable GPU (recommended)
- Python 3.8+

## Installation

### 1. Install lmms-eval (Optional, for standard benchmark evaluation)

You can use the latest version of lmms-eval:

```bash
# Install latest version (recommended)
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
pip install -e .
cd ..

# Or use AdaLLaVA's specific version if needed:
# git checkout 80391ce3bfb5a19b32e7a19a2d9399e1378ed2dd
```

### 2. Verify Environment

```bash
# Activate environment
source activate_env.sh

# Verify model and controller checkpoints exist
ls checkpoints/molmo/
ls checkpoints/two_stage_controller/stage2/
```

## Dataset Preparation

### Automatic Preparation

Run the preparation script:

```bash
bash scripts/prepare_eval_datasets.sh
```

This will download common evaluation datasets:
- TextVQA
- OK-VQA
- COCO VQA
- ScienceQA
- DocVQA
- Scene Text VQA
- TallyQA

### Manual Preparation

For specific datasets:

```bash
python scripts/download_data.py <dataset_name> --n_procs 1
```

Example:
```bash
python scripts/download_data.py textvqa --n_procs 1
python scripts/download_data.py okvqa --n_procs 1
```

## Evaluation Methods

### Method 1: Direct Dataset Evaluation (Recommended for Quick Testing)

This method evaluates directly on datasets using the project's data pipeline.

**Basic Usage:**

```bash
python experiments/controller/evaluate_adaptive_inference.py \
    --model_path checkpoints/molmo \
    --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
    --dataset text_vqa \
    --num_samples 100 \
    --latency_budget 200.0 \
    --device cuda
```

**Full Options:**

```bash
python experiments/controller/evaluate_adaptive_inference.py \
    --model_path checkpoints/molmo \
    --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
    --dataset text_vqa \
    --split validation \
    --num_samples 100 \
    --latency_budget 200.0 \
    --max_new_tokens 128 \
    --batch_size 1 \
    --device cuda \
    --output_path ./results/logs_eval/ \
    --save_predictions
```

**Parameters:**
- `--model_path`: Path to model checkpoint (required)
- `--controller_path`: Path to controller checkpoint (required)
- `--dataset`: Dataset name (text_vqa, okvqa, coco_2014_vqa, etc.)
- `--split`: Dataset split (validation, test, train) - default: validation
- `--num_samples`: Number of samples to evaluate (None = all)
- `--latency_budget`: Latency budget in milliseconds (default: 200.0)
- `--max_new_tokens`: Maximum tokens to generate (default: 128)
- `--batch_size`: Batch size (default: 1) - **Note: Latency is measured with batch_size=1**
- `--device`: Device (cuda, cpu) - default: cuda
- `--output_path`: Output directory (default: ./results/logs_eval/)
- `--save_predictions`: Save individual predictions

**Important Note on Batch Size**: 
- Latency measurements are performed with **batch_size=1** to accurately measure per-sample latency
- This matches AdaLLaVA's evaluation approach and reflects real-world single-request scenarios
- Larger batch sizes can improve throughput but will affect per-sample latency measurements

### Method 2: Using Updated test_adaptive_inference.py

The test script now supports evaluation:

```bash
python experiments/controller/test_adaptive_inference.py \
    --model_path checkpoints/molmo \
    --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
    --evaluate \
    --dataset text_vqa \
    --num_samples 100 \
    --latency_budget 200.0 \
    --device cuda
```

### Method 3: LMms-Eval Framework (For Standard Benchmarks)

For evaluation on standard benchmarks with FLOPs computation:

```bash
python -m experiments.controller.run_lmms_eval \
    --model_path checkpoints/molmo \
    --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
    --tasks textvqa_val,mme,pope \
    --latency_budget 200.0 \
    --output_path ./logs_eval/
```

**Supported Tasks:**
- `textvqa_val`: TextVQA validation
- `mme`: Multimodal Evaluation
- `pope`: Polling-based Object Probing Evaluation
- `mmbench_en_dev`: MMBench English dev
- `scienceqa_img`: ScienceQA with images
- `vqav2_val`, `vqav2_test`: VQA v2 validation/test
- `okvqa_val`: OK-VQA validation

## Evaluation Workflows

### Workflow 1: Single Dataset Evaluation

Evaluate on one dataset with a specific latency budget:

```bash
python experiments/controller/evaluate_adaptive_inference.py \
    --model_path checkpoints/molmo \
    --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
    --dataset text_vqa \
    --num_samples 100 \
    --latency_budget 200.0
```

### Workflow 2: Multiple Latency Budgets

Evaluate with different latency constraints:

```bash
for budget in 150.0 170.0 200.0 250.0; do
    echo "Evaluating with latency budget: ${budget}ms"
    python experiments/controller/evaluate_adaptive_inference.py \
        --model_path checkpoints/molmo \
        --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
        --dataset text_vqa \
        --num_samples 100 \
        --latency_budget $budget \
        --output_path ./logs_eval_budget_${budget}/
done
```

### Workflow 3: Multiple Datasets

Evaluate on multiple datasets:

```bash
for dataset in text_vqa okvqa coco_2014_vqa; do
    echo "Evaluating on: $dataset"
    python experiments/controller/evaluate_adaptive_inference.py \
        --model_path checkpoints/molmo \
        --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
        --dataset $dataset \
        --num_samples 100 \
        --latency_budget 200.0 \
        --output_path ./logs_eval_${dataset}/
done
```

### Workflow 4: Full Benchmark Suite (LMms-Eval)

Evaluate on multiple benchmarks at once:

```bash
python -m experiments.controller.run_lmms_eval \
    --model_path checkpoints/molmo \
    --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
    --tasks textvqa_val,mme,pope,mmbench_en_dev,scienceqa_img \
    --latency_budget 200.0 \
    --output_path ./logs_eval_full/
```

## Results Analysis

### Output Files

Evaluation produces JSON files with:

1. **Metrics**: Accuracy, latency, throughput
2. **Knob Distribution**: Tier, top_k, num_active_blocks usage
3. **Predictions**: Individual predictions (if `--save_predictions` is used)

### Example Results Structure

```json
{
  "dataset": "text_vqa",
  "split": "validation",
  "num_samples": 100,
  "latency_budget": 200.0,
  "metrics": {
    "accuracy": 0.4523,
    "accuracy_std": 0.0123,
    "avg_latency_ms": 185.5,
    "latency_std_ms": 12.3,
    "throughput_samples_per_sec": 5.4
  },
  "knob_distribution": {
    "tier": {"low": 20, "medium": 50, "high": 30},
    "top_k": {4: 10, 6: 30, 8: 40, 12: 20},
    "num_active_blocks": {12: 15, 14: 35, 16: 50}
  }
}
```

### Analyzing Results

1. **Accuracy vs Latency Tradeoff**: Compare accuracy across different latency budgets
2. **Knob Usage**: Analyze which knob configurations are most common
3. **Dataset Performance**: Compare performance across different datasets

### Visualization (Future Work)

You can create visualizations from the JSON results:

```python
import json
import matplotlib.pyplot as plt

# Load results
with open('results/logs_eval/text_vqa_validation_results.json') as f:
    results = json.load(f)

# Plot accuracy vs latency budget
# (Example - implement as needed)
```

## Troubleshooting

### Common Issues

#### 1. Dataset Not Found

**Error**: `DatasetNotFoundError` or similar

**Solution**:
```bash
# Download the dataset
python scripts/download_data.py <dataset_name> --n_procs 1

# Verify MOLMO_DATA_DIR is set
echo $MOLMO_DATA_DIR
```

#### 2. Controller Checkpoint Not Loading

**Error**: `KeyError` or `RuntimeError` when loading checkpoint

**Solution**:
- Verify checkpoint path is correct
- Check checkpoint contains expected keys:
  ```python
  import torch
  ckpt = torch.load('checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt')
  print(ckpt.keys())
  ```

#### 3. Out of Memory (OOM)

**Error**: `CUDA out of memory`

**Solution**:
- Reduce `batch_size` (default: 1, try smaller if needed)
- Reduce `num_samples` for testing
- Use smaller latency budgets (fewer active blocks)
- Clear GPU cache: `torch.cuda.empty_cache()`

#### 4. lmms-eval Not Found

**Error**: `ModuleNotFoundError: No module named 'lmms_eval'`

**Solution**:
```bash
# Install lmms-eval
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
git checkout 80391ce3bfb5a19b32e7a19a2d9399e1378ed2dd
pip install -e .
```

#### 5. Slow Evaluation

**Issue**: Evaluation is very slow

**Solution**:
- Reduce `num_samples` for quick testing
- Use `batch_size=1` (already default)
- Check GPU utilization
- Consider using multiple GPUs (future work)

## Best Practices

1. **Start Small**: Test with `--num_samples 10` first
2. **Verify Checkpoints**: Ensure model and controller checkpoints are valid
3. **Monitor Resources**: Watch GPU memory and CPU usage
4. **Save Results**: Always use `--save_predictions` for detailed analysis
5. **Document Settings**: Keep track of evaluation parameters for reproducibility

## Next Steps

1. **Compare Baselines**: Evaluate static model (no adaptive inference) for comparison
2. **Sweep Latency Budgets**: Test multiple latency budgets to find optimal tradeoff
3. **Analyze Knob Usage**: Understand which configurations work best
4. **Visualize Results**: Create plots for accuracy-latency tradeoffs
5. **Report Metrics**: Document results for paper/report

## References

- [AdaLLaVA Paper](https://arxiv.org/pdf/2503.10905)
- [AdaLLaVA GitHub](https://github.com/zhuoyan-xu/AdaLLaVA)
- [LMms-Eval Documentation](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- [LMms-Eval Integration Guide](./lmms_eval_integration.md)

