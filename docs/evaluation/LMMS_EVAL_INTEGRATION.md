# LMms-Eval Integration for Adaptive Inference

This document describes how to use lmms-eval to evaluate the adaptive inference engine, following AdaLLaVA's approach.

## Overview

We integrate the adaptive inference engine with [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to evaluate on standard multimodal benchmarks. This allows us to:

1. Evaluate on multiple benchmarks consistently
2. Compute FLOPs, latency, and memory usage during evaluation
3. Compare with other models using the same evaluation framework
4. Support various latency budget constraints

## Installation

### 1. Install lmms-eval

You can use the latest version of lmms-eval (no need for a specific commit):

```bash
# Option 1: Install latest version from GitHub (recommended)
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
pip install -e .
cd ..

# Option 2: Install via pip (if available)
# pip install lmms-eval
```

**Note**: If you encounter compatibility issues, you can use AdaLLaVA's specific version:
```bash
git checkout 80391ce3bfb5a19b32e7a19a2d9399e1378ed2dd
```

### 2. Install LLM-Viewer (for FLOPs computation)

```bash
# LLM-Viewer is used to compute FLOPs during evaluation
# Follow instructions from: https://github.com/LLM-Viewer/LLM-Viewer
```

## Quick Start

### Basic Evaluation

Evaluate on a single dataset with a latency budget:

```bash
python experiments/controller/evaluate_adaptive_inference.py \
    --model_path checkpoints/molmo \
    --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
    --dataset text_vqa \
    --num_samples 100 \
    --latency_budget 200.0 \
    --device cuda
```

### Using lmms-eval Framework

For more comprehensive evaluation with multiple benchmarks:

```bash
python -m experiments.controller.run_lmms_eval \
    --model_path checkpoints/molmo \
    --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
    --tasks textvqa_val,mme,pope \
    --latency_budget 200.0 \
    --output_path ./logs_eval/
```

### Using Updated test_adaptive_inference.py

The updated test script now supports evaluation:

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

## Supported Datasets/Benchmarks

### Datasets Used by AdaLLaVA (from paper)

According to the AdaLLaVA paper, the following datasets were used for evaluation:

- **TextVQA** (`textvqa_val`): Text-based Visual Question Answering
- **VQAv2** (`vqav2_test`, `vqav2_val`): Visual Question Answering v2
- **MME** (`mme`): Multimodal Evaluation
- **POPE** (`pope`): Polling-based Object Probing Evaluation
- **MMBench** (`mmbench_en_dev`): Multimodal Benchmark
- **ScienceQA** (`scienceqa_img`): Science Question Answering
- **OK-VQA** (`okvqa_val`): Outside Knowledge VQA
- **GQA**: General Question Answering (if available in lmms-eval)

### Standard Benchmarks (via lmms-eval)

All the above datasets are supported, plus additional benchmarks available in lmms-eval.

### Direct Dataset Evaluation

- `text_vqa`: TextVQA dataset
- `okvqa`: OK-VQA dataset
- `coco_2014_vqa`: COCO VQA dataset
- `science_qa_img`: ScienceQA with images
- `doc_qa`: Document VQA
- `st_qa`: Scene Text VQA
- `tally_qa`: TallyQA dataset

## Latency Measurement

**Important**: Latency measurements are performed with **batch_size=1** to accurately measure per-sample latency. This is consistent with AdaLLaVA's evaluation approach and reflects real-world single-request scenarios.

- **Batch size for latency measurement**: Fixed at 1
- **Batch size for accuracy evaluation**: Can be adjusted (default: 1)
- **Reason**: Batch size affects latency - larger batches can improve throughput but increase per-sample latency

## Evaluation with Different Latency Budgets

To evaluate with different latency constraints (similar to AdaLLaVA):

```bash
# 85% latency budget
python -m experiments.controller.run_lmms_eval \
    --model_path checkpoints/molmo \
    --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
    --tasks textvqa_val,mme \
    --latency_budget 170.0 \
    --output_path ./logs_eval_0.85/

# 100% latency budget (baseline)
python -m experiments.controller.run_lmms_eval \
    --model_path checkpoints/molmo \
    --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
    --tasks textvqa_val,mme \
    --latency_budget 200.0 \
    --output_path ./logs_eval_1.0/
```

## Output Format

### Results File Structure

The evaluation produces JSON files with the following structure:

```json
{
  "dataset": "text_vqa",
  "split": "validation",
  "num_samples": 100,
  "latency_budget": 200.0,
  "max_new_tokens": 128,
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
  },
  "predictions": [...]
}
```

### LMms-Eval Output Format

When using lmms-eval, the output follows the standard lmms-eval format:

```json
{
  "textvqa_val": {
    "accuracy": 0.4523,
    "flops": 7239070670529.51,
    "avg_flops": 3615471530841.294,
    "prefill_time": 0.0687,
    "memory_consumption": 22598248812.697556,
    ...
  }
}
```

## Evaluation Workflow

### 1. Prepare Datasets

Most datasets are automatically downloaded via HuggingFace. For some datasets, you may need to download images separately:

```bash
# Download TextVQA images (if needed)
python scripts/download_data.py textvqa --n_procs 1
```

### 2. Run Evaluation

Choose one of the evaluation methods:

- **Direct evaluation** (faster, simpler): Use `evaluate_adaptive_inference.py`
- **LMms-eval** (more comprehensive, standard format): Use `run_lmms_eval.py`

### 3. Analyze Results

Results are saved as JSON files. You can:

- Compare accuracy across different latency budgets
- Analyze knob distribution (tier, top_k, num_active_blocks)
- Compute latency-quality tradeoffs
- Generate visualizations

## Integration with AdaLLaVA Approach

Our implementation follows AdaLLaVA's evaluation approach:

1. **Latency Budget Control**: Set latency budget as a parameter (e.g., 0.85 = 85% of baseline)
2. **Multiple Benchmarks**: Support evaluation on multiple benchmarks in one run
3. **FLOPs Computation**: Integrate with LLM-Viewer for FLOPs computation (when available)
4. **Standard Format**: Output results in standard lmms-eval format

## Troubleshooting

### Issue: lmms-eval not found

**Solution**: Install lmms-eval following the installation instructions above.

### Issue: Dataset not found

**Solution**: 
1. Check dataset name spelling
2. Download dataset using `scripts/download_data.py`
3. Set `MOLMO_DATA_DIR` environment variable

### Issue: Out of memory

**Solution**:
1. Reduce `batch_size` (default: 1)
2. Reduce `num_samples` for testing
3. Use smaller latency budgets (fewer active blocks)

### Issue: Controller checkpoint not loading

**Solution**:
1. Verify checkpoint path is correct
2. Check that checkpoint contains the expected keys
3. Ensure model and controller architectures match

## Advanced Usage

### Custom Latency Budgets

You can evaluate with multiple latency budgets in a loop:

```bash
for budget in 150.0 170.0 200.0 250.0; do
    python experiments/controller/evaluate_adaptive_inference.py \
        --model_path checkpoints/molmo \
        --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
        --dataset text_vqa \
        --num_samples 100 \
        --latency_budget $budget \
        --output_path ./logs_eval_budget_${budget}/
done
```

### Batch Evaluation on Multiple Datasets

```bash
for dataset in text_vqa okvqa coco_2014_vqa; do
    python experiments/controller/evaluate_adaptive_inference.py \
        --model_path checkpoints/molmo \
        --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
        --dataset $dataset \
        --num_samples 100 \
        --latency_budget 200.0 \
        --output_path ./logs_eval_${dataset}/
done
```

## References

- [AdaLLaVA GitHub](https://github.com/zhuoyan-xu/AdaLLaVA)
- [AdaLLaVA Paper](https://arxiv.org/pdf/2503.10905)
- [LMms-Eval GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- [LLM-Viewer GitHub](https://github.com/LLM-Viewer/LLM-Viewer)

