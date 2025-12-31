# Core Experiments: Combined Profiling

This directory contains the reorganized profiling experiments for E1, E2, and E3 analysis.

## Overview

- **`acc_lat_profiling.py`**: Combined accuracy and latency profiling
  - Measures accuracy and stage-wise latency for different knob combinations
  - Uses vision tokens (target) as first knob instead of max_crops
  - Records detailed metadata needed for E1, E2, E3 analysis
  - Supports optional PyTorch profiler for detailed operator-level analysis
  - Supports multi-dataset execution

**Main scripts**:
- **`run_multi_datasets_h100.sh`**: Multi-dataset execution for H100 GPUs (recommended)
- **`run_multi_datasets_a100.sh`**: Multi-dataset execution for A100 GPUs (with memory optimizations)

## Key Features

### 1. Vision Tokens Control (Primary Knob)

The experiment uses **target vision tokens** as the primary control knob:

```python
# Target vision tokens → calculate num_crops
num_crops = (target_tokens // 144) - 1
max_crops = num_crops  # Set to num_crops to ensure exact crop count
```

**Key Innovation**: Instead of fixing image dimensions (which can cause aspect ratio mismatches), we specify a **target number of vision tokens** and let the system automatically adapt the tiling configuration to each image's aspect ratio.

**Benefits**:
- ✅ **Adaptive tiling**: Each image gets the best tiling for its aspect ratio
- ✅ **Minimal distortion**: Aspect ratio is preserved as much as possible
- ✅ **Simple configuration**: Just specify vision token values (e.g., `432 720 1008 1440`)
- ✅ **Consistent experiments**: All configs use same vision token targets
- ✅ **Better accuracy**: More vision tokens → better accuracy

**How it works**:
1. Specify target vision tokens (e.g., 1008)
2. Calculate required crops: `num_crops = (1008 // 144) - 1 = 6`
3. For each image, `select_tiling` automatically selects the best tiling based on original aspect ratio
4. Image is resized to optimal dimensions with minimal distortion

**See**: `docs/knobs/vision_tokens_knob.md` for detailed explanation and examples.

### 2. Vision Tokens Calculation

The experiment calculates actual vision tokens from batch:

```python
# Formula: Total Vision Tokens = (num_crops + 1) × 144 (theoretical)
# Actual counting: only valid tokens (excludes invalid patches marked as -100)
num_vision_tokens = (batch["image_input_idx"] >= 0).sum().item()
```

**Important**: Actual vision tokens may be **slightly less** than theoretical value `(num_crops + 1) × 144` because:
- Some patches may be marked as invalid (-100) if they exceed image boundaries
- Padding may cause some patches to be invalid
- Tiling configuration may result in partial crops

**Typical deviation**: 0-24 tokens (0-5% of theoretical value). The deviation is consistent across similar images, so it doesn't significantly affect experimental comparisons.

Records both **target** and **actual** vision tokens for comparison.

### 3. Dataset Sampling

Supports dataset sampling to reduce experiment time:

- **Default**: 1000 samples (sufficient for stable estimates)
- **With profiler**: Can use fewer samples (100-200) for detailed analysis
- **Consistent sampling**: Uses fixed seed (42) for reproducibility

**Why sampling is safe**:
- Random sampling preserves distribution
- Does not change Pareto frontier trends if sample size ≥ 500 per config
- Same samples used across all configurations for fair comparison

### 4. Stage-Wise Latency Breakdown

Records detailed stage breakdown:
- `T_vision_total`: Vision backbone processing time (ViT + Projector)
- `T_LLM_prefill`: Transformer prefill time
- `T_LLM_decode`: Per-token generation time
- `T_total`: End-to-end latency

**Uses manual timing** (no profiler) by default to avoid overhead.

### 5. Optional PyTorch Profiler

Can enable PyTorch profiler for detailed operator-level analysis:

**Mode 1: First sample only (default, faster)**
```bash
USE_PROFILER=true ./run_combined_profiling.sh
```

**Mode 2: All samples (comprehensive, slower)**
```bash
python experiments/core_exp/acc_lat_profiling.py \
    --use_profiler \
    --use_profiler_on_all_samples \
    --num_samples 200
```

**When to use**:
- For E1 detailed analysis (operator-level breakdown)
- Mode 1: Quick profiling, representative data
- Mode 2: Comprehensive profiling, all samples
- Profiler adds 2-5% overhead per sample (minimal mode)

See `PROFILER_USAGE.md` for detailed usage guide.

### 6. Detailed Metadata Recording

Records:
- **Vision tokens**: Target and actual vision tokens (per sample)
- **Configuration**: vision_tokens (primary), top_k, num_active_blocks
- **Language prompt**: Full prompt text (per sample)
- **Accuracy**: Per-sample accuracy scores
- **Stage latencies**: Per-sample stage breakdown (T_vision_total, T_LLM_prefill, T_LLM_decode, T_total)
- **Aggregate statistics**: Mean, std, P50, P95, P99 for all metrics

## Usage

### Single Dataset

```bash
# Multi-GPU (recommended)
torchrun --nproc-per-node=4 experiments/core_exp/acc_lat_profiling.py \
    --model_path checkpoints \
    --output_dir ./results/core_exp_h100 \
    --dataset_name coco_2014_vqa \
    --sampling_strategy balanced \
    --num_samples 1000 \
    --num_runs_per_sample 3 \
    --tier_list low medium high \
    --top_k_list 8 12 \
    --num_active_blocks_list 14 16
```

### Multi-Dataset (Recommended)

```bash
# Run on all configured datasets (H100)
bash experiments/core_exp/run_multi_datasets_h100.sh

# Run on specific dataset only
bash experiments/core_exp/run_multi_datasets_h100.sh coco_2014_vqa

# Run on A100 (with memory optimizations)
bash experiments/core_exp/run_multi_datasets_a100.sh coco_2014_vqa
```

### With PyTorch Profiler

```bash
# Enable profiler for detailed operator-level analysis
USE_PROFILER=true NUM_SAMPLES=200 bash experiments/core_exp/run_multi_datasets_h100.sh coco_2014_vqa

# Or via command line
python experiments/core_exp/acc_lat_profiling.py \
    --use_profiler \
    --num_samples 200 \
    --tier_list low medium high
```

### Custom Configuration

```bash
# Custom vision tokens, top_k, and blocks
python experiments/core_exp/acc_lat_profiling.py \
    --tier_list low medium \
    --top_k_list 4 8 16 32 \
    --num_active_blocks_list 8 12 16 20 24 \
    --sampling_strategy stratified \
    --num_samples 2000
```

## Output Files

### Combined Profiling

Each configuration is saved in a separate file with descriptive naming:

**Filename format** (tier-based mode):
```
<task_name>_imgsizetier-<tier>_crops<mean>_topk<k>_blocks<n>.json
```

**Examples**:
- `coco-2014-vqa_imgsizetier-low_crops2_topk8_blocks14.json`
- `coco-2014-vqa_imgsizetier-medium_crops6_topk12_blocks16.json`
- `coco-2014-vqa_imgsizetier-high_crops12_topk8_blocks14.json`

**Additional files**:
- `profiler_results_config_{config_idx}_sample_{sample_idx}.txt`: Profiler output (if `--use_profiler` enabled)

**Structure**:
```json
{
  "tier": "medium",
  "tier_range": {"min_crops": 4, "max_crops": 8},
  "selected_crops_mean": 6.2,
  "selected_crops_distribution": {"4": 10, "6": 25, "8": 15},
  "target_vision_tokens_mean": 1008.0,
  "actual_vision_tokens_mean": 1001.5,
  "max_crops": 8,
  "top_k": 12,
  "num_active_blocks": 16,
  "num_total_blocks": 16,
  "active_block_indices": [0, 1, 2, ..., 15],
  "accuracy": 0.85,
  "accuracy_std": 0.02,
  "num_samples": 1000,
  "theoretical_num_crops": 6,
  "theoretical_tiling": [3, 2],
  "theoretical_image_size": [784, 560],
  "theoretical_vision_tokens": 1008,
  "aggregate_stats": {
    "T_vision_total_mean": 47.3,
    "T_LLM_prefill_mean": 120.5,
    "T_LLM_decode_mean": 15.3,
    "T_total_mean": 183.1,
    "T_total_p95": 195.2,
    "T_total_p99": 210.5,
    "vision_tokens_mean": 1001.5,
    "vision_tokens_std": 12.3,
    "vision_tokens_diff_mean": 6.5,
    ...
  },
  "per_sample_results": [
    {
      "sample_id": 0,
      "target_vision_tokens": 1008,
      "target_crops": 6,
      "actual_vision_tokens": 1002,
      "top_k": 12,
      "num_active_blocks": 16,
      "input_text_tokens": 45,
      "output_tokens": 8,
      "theoretical_num_crops": 6,
      "theoretical_tiling": [3, 2],
      "theoretical_image_size": [784, 560],
      "theoretical_vision_tokens": 1008,
      "actual_num_crops": 6,
      "actual_tiling": [3, 2],
      "actual_image_size": [784, 560],
      "accuracy": 0.9,
      "pred": "a cat",
      "metadata": {
        "question": "What is in the image?",
        "answers": ["cat", "kitten"],
        "image_id": "12345",
        ...
      },
      "T_vision_total": 47.1,
      "T_vision_total": 47.1,
      "T_LLM_prefill": 120.3,
      "T_LLM_decode": 15.2,
      "T_total": 182.6,
      "T_decode_per_token": 1.9,
      ...
    }
  ]
}
```

## Sampling Strategies

- **`balanced`** (recommended): Balanced coverage, 3-4 values per dimension
- **`stratified`**: Min, middle, max from each dimension (27 combinations)
- **`boundary`**: More comprehensive boundary sampling
- **`full`**: All combinations (can be very large)
- **`lhs`**: Latin Hypercube Sampling

## Notes

### PyTorch Profiler

**Default**: We use manual timing (no profiler) because:
- Profiler adds 10-30% overhead (default) or 2-5% (minimal)
- Manual timing with `torch.cuda.synchronize()` has < 0.1% overhead
- More accurate for latency measurements

**Optional**: Can enable profiler with `--use_profiler` for:
- E1 detailed operator-level analysis
- Understanding which operators dominate latency
- Use fewer samples (100-200) when profiler enabled

See `docs/core_exp/PROFILER_NOTES.md` for details.

### Dataset Sampling

**Sampling is safe** because:
- Random sampling preserves distribution
- Does not change Pareto frontier trends
- Same samples used across all configurations

**Recommended sample sizes**:
- Accuracy: 2000 samples (sufficient for stable estimates)
- Latency: 1000 samples (sufficient for stable estimates)

## Integration with E1, E2, E3

### E1: Stage-Aware Latency Decomposition

Uses configuration result files (e.g., `coco-2014-vqa_imgsizetokens1008_topk8_blocks14.json`):
- **Stage breakdown**: `T_vision_total`, `T_LLM_prefill`, `T_LLM_decode` (per sample in `per_sample_results`)
- **Vision tokens**: `actual_vision_tokens` per sample
- **Scaling curves**: Prefill vs input tokens, Decode vs output tokens
- **Profiler results**: If `--use_profiler` enabled, use `profiler_results_config_*.txt` for operator-level breakdown

### E2: Knob Coupling + Pareto-Front Structure

Uses configuration result files:
- **Quality**: `accuracy` from config-level results
- **Latency**: `T_total_p95` or `T_total_p99` from `aggregate_stats`
- **Vision tokens**: `target_vision_tokens` and `actual_vision_tokens_mean`
- **Other knobs**: `top_k`, `num_active_blocks`
- **Pareto frontiers**: Combine quality and latency across all configurations
- **Coupling analysis**: Compare frontiers with fixed knobs

### E3: Latency Estimator

Uses configuration result files:
- **Training data**: Stage latencies for different configurations (from `per_sample_results`)
- **Features**: `target_vision_tokens`, `top_k`, `num_active_blocks`, `input_text_tokens`, `output_tokens`
- **Targets**: Stage latencies (`T_vision_total`, `T_LLM_prefill`, `T_LLM_decode`)
- **Per-sample data**: Use `per_sample_results` for training (rich per-sample data)

## Key Changes from Previous Experiments

### 1. Vision Tokens Control (Primary Knob)

**Before**: Used `image_size_list` or `vision_tokens_list` (fixed targets, aspect ratio mismatches)
**Now**: Uses `tier_list` (tier-based adaptive crop selection, minimal distortion)

```python
# Old approach (vision_tokens_list)
vision_tokens_list = [432, 720, 1008, 1440]
# Problems:
# - Fixed vision token targets may force square crop counts
# - Images with different aspect ratios may get suboptimal tiling
# - Limited flexibility for aspect ratio matching

# New approach (tier_list)
tier_list = ["low", "medium", "high"]
# Benefits:
# - Each image gets the best crop count within tier range for its aspect ratio
# - Minimal distortion (aspect ratio preserved)
# - Adaptive selection ensures accuracy benefits from increased vision tokens
# - Simple configuration: low (1-3 crops), medium (4-8 crops), high (9-15 crops)
```

**See**: `docs/knobs/vision_tokens_knob.md` and `docs/knobs/vision_tokens_knob_tier_implementation_summary.md` for detailed comparison and examples.

### 2. Combined Measurement

**Before**: Separate accuracy and latency experiments
**Now**: Combined experiment (batch_size=1, measures both)

### 3. Dataset Sampling

**Before**: Accuracy profiling used full dataset (slow)
**Now**: Both use sampling (faster, statistically equivalent)

## Related Documentation

- **Vision Tokens Knob**: `docs/knobs/vision_tokens_knob.md` (key reference for vision tokens control)
- **MoE Top-K Knob**: `docs/knobs/moe_topk_knob.md`
- **Transformer Blocks Knob**: `docs/knobs/transformer_blocks_knob.md`
- **E1 Documentation**: `docs/core_exp/e1_stage_aware_latency_decomposition.md`
- **E2 Documentation**: `docs/core_exp/e2_knob_coupling_pareto.md`
- **E3 Documentation**: `docs/core_exp/e3_latency_estimator.md`
- **Profiler Notes**: `docs/core_exp/PROFILER_NOTES.md`

