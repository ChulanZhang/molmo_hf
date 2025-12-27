# Profiling Experiments

This document details the "Profiling" experiments designed to investigate specific model behaviors and performance characteristics, such as context scaling and MoE expert selection.

## Overview

These experiments focus on specific architectural or input parameters to understand their impact on performance.

**Note**: Some Profiling experiments correspond to Motivation experiments:
- **Profiling Exp 1** (Context Scaling - Vision Tokens) ↔ **Motivation Exp 3** (Vision Tokens vs Latency)
- **Profiling Exp 4** (Output Tokens Scaling) ↔ **Motivation Exp 4** (Language Tokens vs Latency)

The Profiling versions are simplified and focused on specific control knobs, while Motivation experiments use real datasets and provide more comprehensive analysis.

### Data Collection Methodology
Data is collected using specific scripts for each experiment, but generally follows the same principles as the Motivation experiments.

*   **Latencies**: Measured using `time.perf_counter()` around specific model components.
    ```python
    # experiments/motivate/base_experiment.py
    start = time.perf_counter()
    # ... model execution ...
    latency = (time.perf_counter() - start) * 1000
    ```
*   **Tokens**: Captured from actual tensor shapes during inference.
    ```python
    # experiments/motivate/base_experiment.py
    results["num_input_text_tokens"] = int(input_ids.shape[1])
    ```
*   **Per-Sample Results**: All experiments save complete per-sample results (similar to Motivation experiments), including:
    - All latency metrics from `measure_inference_latency`
    - Configuration parameters (e.g., `top_k`, `num_active_blocks`)
    - Token counts
    - Statistical summaries
*   **Code Reference**: See `experiments/profiling/` scripts and `experiments/base_experiment.py`.

### Output Format

All Profiling experiments save results in a unified format:

```json
{
  "summary": [
    {
      // Statistical summary for each configuration
      // (mean, std, percentiles, etc.)
    },
    ...
  ],
  "all_samples": [
    {
      "sample_id": 0,
      // Configuration parameters
      // All metrics from measure_inference_latency
      // Token counts, etc.
    },
    ...
  ]
}
```

- **`summary`**: Aggregated statistics for each tested configuration
- **`all_samples`**: Complete per-sample results for detailed analysis

## Experiments

### 1. Context Scaling (Exp 1) - Vision Tokens
**Goal**: Research the impact of input vision tokens on Prefill latency.

*   **Corresponds to**: Motivation Experiment 3 (Vision Tokens vs Latency)
*   **Script**: `experiments/profiling/knob1_tokens/exp_context_scaling.py`
*   **Bash Script**: `experiments/profiling/run_exp1_context_scaling.sh`
*   **Method**:
    *   Resize dummy images to different resolutions to trigger different crop counts.
    *   Test various tiling configurations (1×1, 1×2, ..., 1×12) to cover different vision token counts.
    *   Measure `T_LLM_prefill` and component latencies.
*   **Usage (Python)**:
    ```bash
    python experiments/profiling/knob1_tokens/exp_context_scaling.py \
        --output_dir ./results/context_scaling \
        --num_samples 50
    ```
*   **Usage (Bash - Recommended)**:
    ```bash
    # Basic usage
    bash experiments/profiling/run_exp1_context_scaling.sh 0
    
    # Custom parameters
    bash experiments/profiling/run_exp1_context_scaling.sh 0 \
        --num_samples 12 \
        --max_grid_size 12 \
        --num_runs 10 \
        --model_path checkpoints \
        --output_dir ./results/custom_context_scaling
    ```
*   **Visualization**: `experiments/profiling/plot_exp1_context_scaling.py`
    ```bash
    python experiments/profiling/plot_exp1_context_scaling.py \
        --json_file ./results/profiling/context_scaling/exp1_context_scaling_results.json
    ```

### 2. MoE Top-K Analysis (Exp 2)
**Goal**: Research the impact of MoE Top-K parameter on Prefill and Decode latency.

*   **Script**: `experiments/profiling/knob2_topk/exp_moe_topk.py`
*   **Bash Script**: `experiments/profiling/run_exp2_moe_topk.sh`
*   **Method**:
    *   Test `top_k = [1, 2, 4, 8, 16, 32]` (default, can be customized via `--top_k_values`).
    *   Dynamically modify `block.ffn.args.top_k` for all MoE blocks.
    *   Measure `T_LLM_prefill` and `T_LLM_decode` for each top_k value.
    *   Model has 64 experts, so testing up to k=32 covers half the expert range.
*   **See Also**: Detailed analysis document: `docs/EXP_MOE_TOPK_ANALYSIS.md`
*   **Usage (Python)**:
    ```bash
    # Use default top_k values [1, 2, 4, 8, 16, 32]
    python experiments/profiling/knob2_topk/exp_moe_topk.py \
        --output_dir ./results/moe_topk \
        --num_samples 50
    
    # Custom top_k values
    python experiments/profiling/knob2_topk/exp_moe_topk.py \
        --output_dir ./results/moe_topk \
        --num_samples 50 \
        --top_k_values 1 2 4 8
    ```
*   **Usage (Bash - Recommended)**:
    ```bash
    # Basic usage (default top_k values)
    bash experiments/profiling/run_exp2_moe_topk.sh 0
    
    # Custom top_k values
    bash experiments/profiling/run_exp2_moe_topk.sh 0 \
        --num_samples 100 \
        --top_k_values 1 2 4 8 16 \
        --model_path checkpoints \
        --output_dir ./results/custom_moe_topk
    ```
*   **Visualization**: `experiments/profiling/plot_exp2_moe_topk.py`
    ```bash
    python experiments/profiling/plot_exp2_moe_topk.py \
        --json_file ./results/profiling/moe_topk/exp2_moe_topk_results.json
    ```

### 3. Transformer Blocks Mask (Exp 3)
**Goal**: Research the impact of number of active transformer blocks (model depth) on Prefill and Decode latency using a mask mechanism (not early exit).

*   **Script**: `experiments/profiling/knob3_layers/exp_transformer_blocks_mask.py`
*   **Bash Script**: `experiments/profiling/run_exp3_transformer_blocks_mask.sh`
*   **Method**:
    *   Test various numbers of active blocks (default: all counts from 1 to total_blocks, e.g., 1, 2, 3, ..., 16).
    *   Uses a mask mechanism to skip certain blocks during forward pass without removing them from the model.
    *   Activates blocks from early to late (first N blocks).
    *   Measure `T_LLM_prefill` and `T_LLM_decode` for each configuration.
*   **See Also**: Detailed documentation: `docs/knobs/transformer_blocks_knob.md` (includes importance score-based selection)
*   **Usage (Python)**:
    ```bash
    # Use default: test all block counts from 1 to total_blocks (e.g., 1, 2, 3, ..., 16)
    python experiments/profiling/knob3_layers/exp_transformer_blocks_mask.py \
        --output_dir ./results/transformer_blocks_mask \
        --num_samples 50
    
    # Custom active block counts (if you want to test specific counts only)
    python experiments/profiling/knob3_layers/exp_transformer_blocks_mask.py \
        --output_dir ./results/transformer_blocks_mask \
        --num_samples 50 \
        --num_active_blocks 4 8 12 16
    ```
*   **Usage (Bash - Recommended)**:
    ```bash
    # Basic usage (default: test all block counts from 1 to total_blocks)
    bash experiments/profiling/run_exp3_transformer_blocks_mask.sh 0
    
    # Custom active block counts (if you want to test specific counts only)
    bash experiments/profiling/run_exp3_transformer_blocks_mask.sh 0 \
        --num_samples 100 \
        --num_active_blocks 4 8 12 16 \
        --model_path checkpoints \
        --output_dir ./results/custom_blocks_mask
    ```
*   **Visualization**: `experiments/profiling/plot_exp3_transformer_blocks_mask.py`
    ```bash
    python experiments/profiling/plot_exp3_transformer_blocks_mask.py \
        --json_file ./results/profiling/transformer_blocks_mask/exp3_transformer_blocks_mask_results.json
    ```

### 4. Output Tokens Scaling (Exp 4)
**Goal**: Research the impact of number of output tokens on Decode latency.

*   **Corresponds to**: Motivation Experiment 4 (Language Tokens vs Latency)
*   **Script**: `experiments/profiling/knob4_output_tokens/exp_output_tokens_scaling.py`
*   **Bash Script**: `experiments/profiling/run_exp4_output_tokens.sh`
*   **Method**:
    *   Force generation of different numbers of output tokens with fixed image.
    *   Test `max_new_tokens = [1, 5, 10, 20, 50, 100, 200]` (default, can be customized).
    *   Measure `T_LLM_decode` and `T_total` for each max_new_tokens value.
*   **Usage (Bash - Recommended)**:
    ```bash
    # Basic usage (default max_new_tokens values)
    bash experiments/profiling/run_exp4_output_tokens.sh 0
    
    # Custom max_new_tokens values
    bash experiments/profiling/run_exp4_output_tokens.sh 0 \
        --num_samples 100 \
        --max_new_tokens 1 10 20 50 100 \
        --model_path checkpoints \
        --output_dir ./results/custom_output_tokens
    ```
*   **Visualization**: `experiments/profiling/plot_exp4_output_tokens.py`
    ```bash
    python experiments/profiling/plot_exp4_output_tokens.py \
        --json_file ./results/profiling/output_tokens/exp4_output_tokens_scaling_results.json
    ```

### 5. FLOPs Scaling Analysis
**Goal**: Measure theoretical vs. actual latency scaling with Top-K.

*   **Script**: `experiments/profiling/utils/measure_flops_scaling.py`
*   **Method**: Compare `top_k=1` vs `top_k=8`.

## Visualization

All Profiling experiments have corresponding plotting scripts that generate stacked bar charts showing latency breakdowns:

*   **`plot_exp1_context_scaling.py`**: Vision tokens vs latency breakdown (stacked: Vision Encoder + Projector + LLM Prefill)
*   **`plot_exp2_moe_topk.py`**: MoE Top-K vs latency (two subplots: Prefill and Decode breakdowns)
*   **`plot_exp3_transformer_blocks_mask.py`**: Transformer blocks vs latency (two subplots: Prefill and Decode breakdowns)
*   **`plot_exp4_output_tokens.py`**: Output tokens vs latency (stacked: Vision + Prefill + Decode, log scale)
*   **`plot_all_experiments.sh`**: Plot all experiments at once

### Usage

```bash
# Plot individual experiment
python experiments/profiling/plot_exp1_context_scaling.py \
    --json_file ./results/profiling/context_scaling/exp1_context_scaling_results.json

# Plot all experiments
bash experiments/profiling/plot_all_experiments.sh

# Custom results directory
bash experiments/profiling/plot_all_experiments.sh --results_dir ./custom_results/profiling
```

All plots are saved to `{output_dir}/figures/` directory with high resolution (300 DPI).

## Debugging & Inspection Tools

The `experiments/profiling/` directory also contains several tools for inspecting the model:

*   **`analyze_tokens.py`**: Analyze input tokenization, padding, and special tokens.
*   **`inspect_moe_layer.py`**: Check MoE layer structure and configuration.
*   **`inspect_molmo_flow.py`**: Trace the forward pass and activation shapes.
*   **`inspect_pooling_params.py`**: Check vision pooling parameters.
*   **`quick_inspect_structure.py`**: Overview of model hierarchy and parameter counts.
*   **`verify_moe_topk.py`**: Verify that Top-K changes are actually taking effect during inference.
*   **`test_hf_model.py`**: Basic sanity check for model loading and inference.

## Running All Experiments

You can run all profiling experiments sequentially using:

```bash
# Run all experiments with default settings
bash experiments/profiling/run_all_experiments.sh 0

# Run all experiments with custom sample count
bash experiments/profiling/run_all_experiments.sh 0 --num_samples 100
```

## Bash Scripts

Each experiment has a corresponding bash script for easy execution:

*   **`run_exp1_context_scaling.sh`**: Context scaling experiment (vision tokens)
*   **`run_exp2_moe_topk.sh`**: MoE Top-K analysis
*   **`run_exp3_transformer_blocks_mask.sh`**: Transformer blocks mask
*   **`run_exp4_output_tokens.sh`**: Output tokens scaling
*   **`run_all_experiments.sh`**: Run all experiments sequentially

All bash scripts follow the same pattern:
1. First argument is GPU_ID (default: 0)
2. Support common arguments: `--model_path`, `--output_dir`, `--num_samples`
3. Support experiment-specific arguments
4. Provide clear output and error handling

Example:
```bash
# Run on GPU 1 with custom parameters
bash experiments/profiling/run_exp2_moe_topk.sh 1 \
    --num_samples 100 \
    --top_k_values 1 2 4 8 \
    --output_dir ./results/custom
```

## Directory Structure
```text
experiments/profiling/
    ├── run_exp1_context_scaling.sh
    ├── run_exp2_moe_topk.sh
    ├── run_exp3_transformer_blocks_mask.sh
    ├── run_all_experiments.sh
    ├── knob1_tokens/
    │   └── exp_context_scaling.py
    ├── knob2_topk/
    │   └── exp_moe_topk.py
    ├── knob3_layers/
    │   ├── exp_transformer_blocks_mask.py
    │   └── exp_layer_skipping.py
    ├── knob4_output_tokens/
    │   └── exp_output_tokens_scaling.py
    ├── utils/
    │   ├── measure_flops_scaling.py
    │   ├── plot_context_scaling.py
    │   ├── analyze_tokens.py
    │   ├── inspect_moe_layer.py
    │   └── ...
    └── ...
```
