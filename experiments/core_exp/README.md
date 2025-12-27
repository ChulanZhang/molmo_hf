# Core Experiments: Combined Profiling

This directory contains the reorganized profiling experiments for E1, E2, and E3 analysis.

## Overview

- **`combined_profiling.py`**: Combined accuracy and latency profiling
  - Measures accuracy and stage-wise latency for different knob combinations
  - Uses vision tokens (target) as first knob instead of max_crops
  - Records detailed metadata needed for E1, E2, E3 analysis
  - Supports optional PyTorch profiler for detailed operator-level analysis
  - Supports multi-dataset execution

**Main scripts**:
- **`run_combined_profiling.sh`**: Single dataset execution
- **`run_multi_datasets.sh`**: Multi-dataset execution (recommended for batch experiments)

## Key Features

### 1. Vision Tokens Control

The experiment uses **target vision tokens** as the first knob:

```python
# Target vision tokens → calculate num_crops
num_crops = (target_tokens // 144) - 1
max_crops = 16  # Fixed upper limit for compatibility with large images
```

**Benefits**:
- Precise control over vision token count
- `max_crops=16` fixed as upper limit (ensures compatibility with large images)
- `num_crops` is the actual number of crops used (recorded in results)
- Better for E1, E2, E3 analysis (direct vision tokens control)

### 2. Vision Tokens Calculation

The experiment calculates actual vision tokens from batch:

```python
# Formula: Total Vision Tokens = (num_crops + 1) × 144
num_vision_tokens = (batch["image_input_idx"] >= 0).sum().item()
```

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
- `T_vision_encoder`: ViT processing time
- `T_projector`: Image feature projection time
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
python experiments/core_exp/combined_profiling.py \
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
- **Stage latencies**: Per-sample stage breakdown (T_vision_encoder, T_projector, T_LLM_prefill, T_LLM_decode, T_total)
- **Aggregate statistics**: Mean, std, P50, P95, P99 for all metrics

## Usage

### Single Dataset

```bash
# Single-GPU
./run_combined_profiling.sh

# Multi-GPU
torchrun --nproc-per-node=4 experiments/core_exp/combined_profiling.py \
    --model_path checkpoints \
    --output_dir ./results/core_exp \
    --dataset_name coco_2014_vqa \
    --sampling_strategy balanced \
    --num_samples 1000 \
    --num_runs_per_sample 3 \
    --vision_tokens_list 432 720 1008 1296 1584 \
    --top_k_list 4 8 12 \
    --num_active_blocks_list 12 13 14 15 16
```

### Multi-Dataset (Recommended)

```bash
# Run on all configured datasets
./run_multi_datasets.sh

# Run on specific dataset only
./run_multi_datasets.sh coco_2014_vqa
```

### With PyTorch Profiler

```bash
# Enable profiler for detailed operator-level analysis
USE_PROFILER=true NUM_SAMPLES=200 ./run_combined_profiling.sh

# Or via command line
python experiments/core_exp/combined_profiling.py \
    --use_profiler \
    --num_samples 200 \
    --vision_tokens_list 288 432 576 720
```

### Custom Configuration

```bash
# Custom vision tokens, top_k, and blocks
python experiments/core_exp/combined_profiling.py \
    --vision_tokens_list 288 432 576 \
    --top_k_list 4 8 16 32 \
    --num_active_blocks_list 8 12 16 20 24 \
    --sampling_strategy stratified \
    --num_samples 2000
```

## Output Files

### Combined Profiling

- `combined_profiling_results.json`: Final aggregated results
- `combined_profiling_results_{config_idx}.json`: Intermediate results per configuration
- `profiler_results_config_{config_idx}.txt`: Profiler output (if `--use_profiler` enabled)

**Structure**:
```json
{
  "summary": [
    {
      "target_vision_tokens": 1008,
      "actual_vision_tokens_mean": 1008.5,
      "vision_tokens": 1008,  # Primary knob value
      "num_crops": 6,  # Actual number of crops used
      "max_crops": 16,  # Fixed upper limit for compatibility
      "top_k": 12,
      "num_active_blocks": 16,
      "accuracy": 0.85,
      "accuracy_std": 0.02,
      "num_samples": 1000,
      "aggregate_stats": {
        "T_vision_encoder_mean": 45.2,
        "T_projector_mean": 2.1,
        "T_LLM_prefill_mean": 120.5,
        "T_LLM_decode_mean": 15.3,
        "T_total_mean": 183.1,
        "T_total_p95": 195.2,
        "T_total_p99": 210.5,
        "vision_tokens_mean": 1872.5,
        ...
      },
      "per_sample_results": [
        {
          "sample_id": 0,
          "target_vision_tokens": 1008,
          "actual_vision_tokens": 1008,
          "vision_tokens": 1008,
          "num_crops": 6,  # Actual number of crops used
          "top_k": 12,
          "num_active_blocks": 16,
          "language_prompt": "What is in the image?",
          "accuracy": 0.9,
          "T_vision_encoder": 45.1,
          "T_projector": 2.0,
          "T_LLM_prefill": 120.3,
          "T_LLM_decode": 15.2,
          "T_total": 182.6,
          ...
        }
      ]
    }
  ],
  "config": {
    "dataset_name": "coco_2014_vqa",
    "num_samples": 1000,
    "use_profiler": false,
    "vision_tokens_list": [288, 432, 576, 720, 1008, 1440, 1872],
    ...
  }
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

Uses `combined_profiling_results.json`:
- **Stage breakdown**: `T_vision_encoder`, `T_projector`, `T_LLM_prefill`, `T_LLM_decode` (per sample)
- **Vision tokens**: `actual_vision_tokens` per sample
- **Scaling curves**: Prefill vs input tokens, Decode vs output tokens
- **Profiler results**: If `--use_profiler` enabled, use `profiler_results_config_*.txt` for operator-level breakdown

### E2: Knob Coupling + Pareto-Front Structure

Uses `combined_profiling_results.json`:
- **Quality**: `accuracy` from per-sample results
- **Latency**: `T_total_p95` or `T_total_p99` from aggregate_stats
- **Vision tokens**: `target_vision_tokens` and `actual_vision_tokens_mean`
- **Pareto frontiers**: Combine quality and latency
- **Coupling analysis**: Compare frontiers with fixed knobs

### E3: Latency Estimator

Uses `combined_profiling_results.json`:
- **Training data**: Stage latencies for different configurations
- **Features**: `target_vision_tokens`, `top_k`, `num_active_blocks`
- **Targets**: Stage latencies (`T_vision_encoder`, `T_projector`, `T_LLM_prefill`, `T_LLM_decode`)
- **Per-sample data**: Use `per_sample_results` for training

## Key Changes from Previous Experiments

### 1. Vision Tokens Control

**Before**: Used `max_crops` (only sets upper limit, imprecise)
**Now**: Uses `target_vision_tokens` (precise control)

```python
# Old approach
max_crops = 12  # May result in different vision tokens depending on image

# New approach
target_vision_tokens = 1872  # Precise target
max_crops = (target_vision_tokens // 144) - 1  # = 12
```

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

