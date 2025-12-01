# Profiling Experiments

This document details the "Profiling" experiments designed to investigate specific model behaviors and performance characteristics, such as context scaling and MoE expert selection.

## Overview

These experiments focus on specific architectural or input parameters to understand their impact on performance.

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
*   **Code Reference**: See `experiments/profiling/` scripts and `experiments/motivate/base_experiment.py`.

## Experiments

### 1. Context Scaling (Exp 1)
**Goal**: Research the impact of input text length on Prefill latency.

*   **Script**: `experiments/profiling/exp1_context_scaling.py`
*   **Method**:
    *   Fixed 336x336 image.
    *   Vary text length: 50, 150, ..., 1500 tokens.
    *   Measure `T_LLM_prefill`.
*   **Usage**:
    ```bash
    python experiments/profiling/exp1_context_scaling.py \
        --output_dir ./results/context_scaling \
        --num_samples 50
    ```
*   **Visualization**: `experiments/profiling/plot_context_scaling.py`

### 2. MoE Top-K Analysis (Exp 2)
**Goal**: Research the impact of MoE Top-K parameter on Prefill and Decode latency.

*   **Script**: `experiments/profiling/exp2_moe_topk.py`
*   **Method**:
    *   Test `top_k = [1, 2, 4, 8]`.
    *   Dynamically modify `block.ffn.args.top_k`.
    *   Measure `T_LLM_prefill` and `T_LLM_decode`.
*   **Usage**:
    ```bash
    python experiments/profiling/exp2_moe_topk.py \
        --output_dir ./results/moe_topk \
        --num_samples 50
    ```

### 3. FLOPs Scaling Analysis
**Goal**: Measure theoretical vs. actual latency scaling with Top-K.

*   **Script**: `experiments/profiling/measure_flops_scaling.py`
*   **Method**: Compare `top_k=1` vs `top_k=8`.

## Debugging & Inspection Tools

The `experiments/profiling/` directory also contains several tools for inspecting the model:

*   **`analyze_tokens.py`**: Analyze input tokenization, padding, and special tokens.
*   **`inspect_moe_layer.py`**: Check MoE layer structure and configuration.
*   **`inspect_molmo_flow.py`**: Trace the forward pass and activation shapes.
*   **`inspect_pooling_params.py`**: Check vision pooling parameters.
*   **`quick_inspect_structure.py`**: Overview of model hierarchy and parameter counts.
*   **`verify_moe_topk.py`**: Verify that Top-K changes are actually taking effect during inference.
*   **`test_hf_model.py`**: Basic sanity check for model loading and inference.

## Directory Structure
```text
experiments/profiling/
    ├── exp1_context_scaling.py
    ├── exp2_moe_topk.py
    ├── measure_flops_scaling.py
    ├── plot_context_scaling.py
    ├── analyze_tokens.py
    ├── inspect_moe_layer.py
    └── ...
```
