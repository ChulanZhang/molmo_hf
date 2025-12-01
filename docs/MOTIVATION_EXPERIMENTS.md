# Motivation Experiments

This document details the "Motivation" experiments designed to systematically analyze the latency characteristics of Vision-Language Models (VLMs) and motivate the need for content- and resource-aware adaptive inference.

## Overview

The experiments are organized into five independent experiments, each focusing on a specific aspect of VLM latency:

1. **Exp 1: Latency Distribution** - Measure latency distribution on real-world datasets
2. **Exp 2: Component Profiling** - Analyze time/parameter cost of each component
3. **Exp 3: Vision Tokens vs Latency** - Quantify the cost of vision tokens (prefill)
4. **Exp 4: Language Tokens vs Latency** - Quantify the cost of language tokens (decode)
5. **Exp 5: FLOPs vs Latency** - Analyze FLOPs-latency correlation

## Experiment 1: Latency Distribution

**Goal**: Measure latency distribution on real-world datasets to identify tail latency issues.

**Method**:
- Dataset: VQA v2 Validation Set
- Sample Size: 5000 samples (recommended for statistical significance)
- Measurement: Total latency only (fast, single forward pass)
- Metrics: `T_total` (end-to-end latency)

**Output**:
- Histogram of latency distribution
- CDF (Cumulative Distribution Function) curve
- Statistics: P50, P95, P99, mean, std, min, max
- Visualization: `results/motivation/exp1/figures/exp1_latency_distribution.png`
- Data: `results/motivation/exp1/exp1_latency_distribution.json`

**Key Insights**:
- Demonstrates tail latency problem (P99 >> P50)
- Shows variability in real-world inference

**Script**: `experiments/motivate/exp1_latency_distribution.py`

**Usage**:
```bash
python experiments/motivate/exp1_latency_distribution.py \
    --model_path checkpoints \
    --dataset coco_2014_vqa \
    --split validation \
    --num_samples 5000 \
    --output_dir ./results/motivation/exp1
```

## Experiment 2: Component Profiling

**Goal**: Analyze the time/parameter cost of each component (Vision Encoder, Projector, LLM).

**Method**:
- Dataset: VQA v2 Validation Set
- Sample Size: 1000 samples (detailed measurement is slower)
- Measurement: Detailed component latencies using subtraction method
- Metrics: `T_vision_encoder`, `T_projector`, `T_LLM_prefill`, `T_LLM_decode`
- Parameters: Count parameters for each component

**Output**:
- Pie chart: Parameter distribution (Vision Encoder, Projector, LLM)
- Pie chart: Latency distribution (Vision Encoder, Projector, LLM Prefill)
- Statistics: Average latencies and parameter counts per component
- Visualization: `results/motivation/exp2/figures/exp2_parameters.png`, `results/motivation/exp2/figures/exp2_latency.png`
- Data: `results/motivation/exp2/exp2_component_profiling.json`

**Key Insights**:
- Shows bottleneck is in LLM (both parameters and latency)
- Quantifies relative cost of vision vs language processing

**Script**: `experiments/motivate/exp2_component_profiling.py`

**Usage**:
```bash
python experiments/motivate/exp2_component_profiling.py \
    --model_path checkpoints \
    --dataset coco_2014_vqa \
    --split validation \
    --num_samples 1000 \
    --output_dir ./results/motivation/exp2
```

## Experiment 3: Vision Tokens vs Latency

**Goal**: Quantify the cost of adding vision tokens (mainly parallelizable prefill).

**Method**:
- Variable: Image resolution (Vision Tokens)
- Fixed: Text prompt, output length (max_new_tokens=0 for prefill only)
- Technique: Resize dummy images to different resolutions to trigger different grid sizes
  - Grid sizes: 1x1, 1x2, ..., 1x12 (rectangular images: 336×336k)
  - Each grid size produces different number of vision tokens
- Measurement: Component latencies (T_vision, T_projector, T_LLM_prefill)
- **Repetition**: **10 runs per resolution** (default, configurable via `--num_runs`) for stability
  - Each resolution is measured 10 times and results are averaged
  - This reduces variance and provides more reliable latency measurements

**Output**:
- Plot: Vision Tokens vs Total Latency
- Plot: Vision Tokens vs Prefill Latency (T_vision + T_projector + T_LLM_prefill)
- Plot: Vision Tokens vs Individual Components (T_vision, T_projector, T_LLM_prefill)
- Statistics: Cost per vision token (slope)
- Visualization: `results/motivation/exp3/figures/exp3_vision_tokens_vs_latency.png`
- Data: `results/motivation/exp3/exp3_vision_tokens_vs_latency.json`

**Key Insights**:
- Vision tokens scale sub-linearly (parallelizable)
- Prefill cost increases with vision tokens
- Quantifies vision token cost per token
- **Note**: By default, `T_LLM_prefill` uses subtraction method. Use `--use_hook_for_llm_prefill` flag for more accurate direct measurement using forward hooks.

**Script**: `experiments/motivate/exp3_vision_tokens_vs_latency.py`

**Usage**:
```bash
# Default (subtraction method for T_LLM_prefill)
python experiments/motivate/exp3_vision_tokens_vs_latency.py \
    --model_path checkpoints \
    --output_dir ./results/motivation/exp3 \
    --max_grid_size 12 \
    --num_runs 10

# Use hook method for more accurate LLM prefill measurement
python experiments/motivate/exp3_vision_tokens_vs_latency.py \
    --model_path checkpoints \
    --output_dir ./results/motivation/exp3 \
    --max_grid_size 12 \
    --num_runs 10 \
    --use_hook_for_llm_prefill
```

Or using the bash script:
```bash
# Default (subtraction method)
bash experiments/motivate/run_exp3.sh [GPU_ID] [--max_grid_size N] [--num_runs N]

# Use hook method for more accurate LLM prefill measurement
bash experiments/motivate/run_exp3.sh [GPU_ID] [--max_grid_size N] [--num_runs N] --use_hook_for_llm_prefill
```

## Experiment 4: Language Tokens vs Latency

**Goal**: Quantify the cost of adding language tokens (sequential decode).

**Method**:
- Variable: Output token length (max_new_tokens)
- Fixed: Image (fixed resolution), text prompt
- Technique: Force generation of different numbers of output tokens
  - max_new_tokens: [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] (default)
  - Covers a wide range from short to very long outputs
- Measurement: T_LLM_decode (decode latency)
- Dataset: Use fixed set of 10 images from VQA v2 validation set

**Output**:
- Plot: Output Tokens vs Decode Latency
- Plot: Output Tokens vs Total Latency (with prefill baseline)
- Statistics: Cost per output token (slope)
- Visualization: `results/motivation/exp4/figures/exp4_language_tokens_vs_latency.png`
- Data: `results/motivation/exp4/exp4_language_tokens_vs_latency.json`

**Key Insights**:
- Language tokens scale linearly (sequential decode)
- Decode cost is additive per token
- Quantifies language token cost per token

**Script**: `experiments/motivate/exp4_language_tokens_vs_latency.py`

**Usage**:
```bash
python experiments/motivate/exp4_language_tokens_vs_latency.py \
    --model_path checkpoints \
    --dataset coco_2014_vqa \
    --split validation \
    --num_samples 10 \
    --output_dir ./results/motivation/exp4 \
    --max_new_tokens_list 8 16 32 64 128 256 512 1024 2048 4096
```

Or using the bash script:
```bash
bash experiments/motivate/run_exp4.sh [GPU_ID] [--num_samples N] [--max_new_tokens_list ...]
```

## Experiment 5: FLOPs vs Latency

**Goal**: Analyze FLOPs-latency correlation to show FLOPs cannot accurately predict single-request latency.

**Method**:
- Data Source: Uses results from Exp 3 and Exp 4
- Analysis: Scatter plot of FLOPs vs Latency
- Metrics: Correlation coefficient, R²
- Components: Separate analysis for Vision (Exp 3) and Language (Exp 4)

**Output**:
- Plot: FLOPs vs Latency (Vision scaling from Exp 3)
- Plot: FLOPs vs Latency (Language scaling from Exp 4)
- Plot: Combined FLOPs vs Latency (all data)
- Statistics: Correlation coefficients, R² values
- Visualization: `results/motivation/exp5/figures/exp5_flops_vs_latency.png`
- Data: `results/motivation/exp5/exp5_flops_vs_latency.json`

**Key Insights**:
- FLOPs are not a perfect predictor of latency
- Memory access, parallelism, and hardware characteristics matter
- Justifies need for fine-grained control beyond FLOPs

**Script**: `experiments/motivate/exp5_flops_vs_latency.py`

**Usage**:
```bash
python experiments/motivate/exp5_flops_vs_latency.py \
    --exp3_results ./results/motivation/exp3/exp3_vision_tokens_vs_latency.json \
    --exp4_results ./results/motivation/exp4/exp4_language_tokens_vs_latency.json \
    --output_dir ./results/motivation/exp5
```

## Data Collection Methodology

### Image Cropping and Tiling

The model uses `max_crops=12` as the **maximum** number of crops allowed, but the actual number of crops is determined adaptively by the `select_tiling` function based on image size.

**How it works:**
- `select_tiling` considers all possible tiling configurations (i×j) where i×j ≤ max_crops
- It selects the tiling that requires the **least upscaling** to fit the image
- For typical VQA v2 images (e.g., 640×512), this often results in 7 crops (1×7 or 7×1 tiling)
- For larger images (e.g., 612×612), this may result in 10 crops (2×5 or 5×2 tiling)

**Why this matters:**
- This adaptive behavior avoids unnecessary upscaling, which would waste computation
- The actual `num_crops` in results reflects this adaptive selection, not the `max_crops` limit
- All experiments (Exp 1, 2, 4) use the same `max_crops=12` configuration, ensuring consistent behavior

### Latency Measurement (`base_experiment.py`)

We use a "Subtraction Method" to isolate component latencies because the model's forward pass implicitly re-computes vision features.

*   **`T_vision_encoder` (ViT Only)**: Measured directly by calling `vision_backbone.encode_image`.
    ```python
    # experiments/motivate/base_experiment.py
    start = time.perf_counter()
    _ = vision_backbone.encode_image(batch["images"])
    T_vision_encoder = (time.perf_counter() - start) * 1000
    ```

*   **`T_vision_total` (ViT + Projector)**: Measured by calling the full `vision_backbone`.
    ```python
    # experiments/motivate/base_experiment.py
    start = time.perf_counter()
    _ = vision_backbone(batch["images"], batch.get("image_masks"))
    T_vision_total = (time.perf_counter() - start) * 1000
    ```

*   **`T_projector`**: Derived as the difference between total vision time and encoder time.
    ```python
    T_projector = max(0.0, T_vision_total - T_vision_encoder)
    ```

*   **`T_LLM_prefill`**: Measured using one of two methods (configurable):
    
    **Default (Subtraction Method)**:
    ```python
    # experiments/motivate/base_experiment.py
    start = time.perf_counter()
    _ = self.model(input_ids=..., images=...) # Implicitly runs vision
    T_prefill_step = (time.perf_counter() - start) * 1000
    
    # Isolate LLM cost
    T_LLM_prefill = max(0.0, T_prefill_step - T_vision_total)
    ```
    
    **Alternative (Direct Hook Method)**:
    ```python
    # Register hooks on first and last transformer blocks
    # Start timer in first block's forward hook
    # Stop timer in last block's forward hook
    # This directly measures LLM prefill time without subtraction
    ```
    
    **Note**: The subtraction method can have measurement errors that cause `T_LLM_prefill` to decrease when vision tokens increase (which is physically impossible). The hook method provides more accurate direct measurement. Use `--use_hook_for_llm_prefill` flag to enable hook method.

*   **`T_LLM_decode`**: Derived from the total generation time minus the prefill step.
    ```python
    # experiments/motivate/base_experiment.py
    start = time.perf_counter()
    _ = self.model.generate(...)
    T_total = (time.perf_counter() - start) * 1000
    
    T_LLM_decode = max(0.0, T_total - T_vision_total - T_LLM_prefill)
    ```

### Parameter Counting (`base_experiment.py`)

Parameters are counted by accessing model components directly:

*   **Vision Encoder Parameters**: Count parameters in `model.model.vision_backbone.image_vit`
*   **Projector Parameters**: Count parameters in `model.model.vision_backbone.image_projector`
*   **LLM Parameters**: Count parameters in `model.model.transformer`

Uses `MolmoModel.get_vit_parameters()`, `MolmoModel.get_connector_parameters()`, and `MolmoModel.get_llm_parameters()` static methods for component identification.

### Token Counting

*   **`num_vision_tokens`**: Calculated from `batch["images"].shape[1]` (number of crops) × 576 (tokens per crop), or from `image_input_idx` valid entries.
    ```python
    # Count valid vision tokens from image_input_idx
    num_vision_tokens = (batch["image_input_idx"] >= 0).sum().item()
    ```

*   **`num_language_tokens`**: Calculated as `num_total_tokens - num_vision_tokens`.
    ```python
    num_language_tokens = num_total_tokens - num_vision_tokens
    ```

*   **`num_total_tokens`**: Extracted directly from `batch["input_ids"].shape[1]` (actual sequence length).
    ```python
    num_total_tokens = batch["input_ids"].shape[1]
    ```

*   **`num_output_tokens`**: Extracted from generation output length minus input length.
    ```python
    num_output_tokens = output.shape[1] - input_ids.shape[1]
    ```

### FLOPs Estimation (`base_experiment.py`)

FLOPs are estimated based on parameter counts and sequence lengths:

*   **Vision Encoder FLOPs**: `vision_params × num_patches × 2`
*   **Projector FLOPs**: `projector_params × 2`
*   **LLM Prefill FLOPs**: `llm_params × input_length × 2`
*   **LLM Decode FLOPs**: `llm_params × output_length × 2`

Note: These are rough estimates. Actual FLOPs depend on implementation details.

## Usage

### Running Individual Experiments

Each experiment has its own bash script for convenience:

```bash
# Exp 1: Latency Distribution
bash experiments/motivate/run_exp1.sh [GPU_ID] [--num_samples N]

# Exp 2: Component Profiling
bash experiments/motivate/run_exp2.sh [GPU_ID] [--num_samples N]

# Exp 3: Vision Tokens vs Latency
bash experiments/motivate/run_exp3.sh [GPU_ID] [--max_grid_size N]

# Exp 4: Language Tokens vs Latency
bash experiments/motivate/run_exp4.sh [GPU_ID] [--num_samples N] [--max_new_tokens_list ...]

# Exp 5: FLOPs vs Latency (requires Exp 3 and 4 results)
bash experiments/motivate/run_exp5.sh [--exp3_results PATH] [--exp4_results PATH]
```

You can also run the Python scripts directly:

```bash
# Exp 1: Latency Distribution
python experiments/motivate/exp1_latency_distribution.py \
    --model_path checkpoints \
    --num_samples 5000 \
    --output_dir ./results/motivation/exp1

# Exp 2: Component Profiling
python experiments/motivate/exp2_component_profiling.py \
    --model_path checkpoints \
    --num_samples 1000 \
    --output_dir ./results/motivation/exp2

# Exp 3: Vision Tokens vs Latency
python experiments/motivate/exp3_vision_tokens_vs_latency.py \
    --model_path checkpoints \
    --output_dir ./results/motivation/exp3

# Exp 4: Language Tokens vs Latency
python experiments/motivate/exp4_language_tokens_vs_latency.py \
    --model_path checkpoints \
    --output_dir ./results/motivation/exp4

# Exp 5: FLOPs vs Latency (requires Exp 3 and 4 results)
python experiments/motivate/exp5_flops_vs_latency.py \
    --exp3_results ./results/motivation/exp3/exp3_vision_tokens_vs_latency.json \
    --exp4_results ./results/motivation/exp4/exp4_language_tokens_vs_latency.json \
    --output_dir ./results/motivation/exp5
```

### Running All Experiments

Use the convenience script to run all experiments in sequence:

```bash
bash experiments/motivate/run_all_experiments.sh [GPU_ID]
```

This script will:
1. Run Exp 1 (5000 samples)
2. Run Exp 2 (1000 samples)
3. Run Exp 3 (controlled scaling)
4. Run Exp 4 (language token scaling)
5. Run Exp 5 (FLOPs analysis, using Exp 3 and 4 results)

### Running Phase Scripts

For backward compatibility, phase scripts are also available:

```bash
# Phase 1: Exp 1 & 2 (Dataset Profiling)
bash experiments/motivate/run_phase1.sh [GPU_ID]

# Phase 2: Exp 3, 4, 5 (Controlled Scaling)
bash experiments/motivate/run_phase2.sh [GPU_ID]
```

Note: Phase scripts now call the individual experiment scripts internally.

### Data Format

Results are saved as JSON files in `results/motivation/`. Each entry contains:

```json
{
  "num_crops": 7,  // Actual number of crops (adaptively selected, ≤ max_crops=12)
  "num_vision_tokens": 6912,
  "num_language_tokens": 34,
  "num_total_tokens": 6946,
  "num_output_tokens": 128,
  "T_total": 123.4,
  "T_vision_encoder": 10.1,
  "T_vision_total": 12.3,
  "T_projector": 2.2,
  "T_LLM_prefill": 45.6,
  "T_LLM_decode": 65.5,
  "flops_vision": 1.2e9,
  "flops_projector": 1.5e8,
  "flops_llm_prefill": 2.3e9,
  "flops_llm_decode": 1.8e9,
  "flops_total": 5.6e9
}
```

## Implementation Details

### Experiment 1: Latency Distribution

**File**: `experiments/motivate/exp1_latency_distribution.py`

**Key Implementation**:
- Uses `measure_components=False` for fast measurement (only T_total)
- Collects 5000 samples for statistical significance
- Generates histogram and CDF plots
- Computes percentile statistics (P50, P95, P99)

**Visualization**:
- Histogram: Distribution of latencies
- CDF: Cumulative distribution function
- Statistics table: P50, P95, P99, mean, std

### Experiment 2: Component Profiling

**File**: `experiments/motivate/exp2_component_profiling.py`

**Key Implementation**:
- Uses `measure_components=True` for detailed measurement
- Counts parameters for each component using `count_parameters()` method
- Generates two pie charts: parameters and latency
- Computes average latencies per component

**Visualization**:
- Pie chart: Parameter distribution
- Pie chart: Latency distribution
- Statistics table: Parameters and latencies per component

### Experiment 3: Vision Tokens vs Latency

**File**: `experiments/motivate/exp3_vision_tokens_vs_latency.py`

**Key Implementation**:
- Creates rectangular dummy images (336×336k) to trigger different grid sizes
- Grid sizes: 1x1, 1x2, ..., 1x12 (k=1 to 12)
- Each grid produces: k crops + 1 global crop = k+1 crops total
- Vision tokens = (k+1) × 576 (tokens per crop)
- **Repetition**: Runs 10 times per resolution (default, configurable via `--num_runs`) and averages results for stability
- Measures T_vision, T_projector, T_LLM_prefill
- Computes FLOPs for each configuration

**Visualization**:
- Scatter plot: Vision tokens vs Total latency
- Line plot: Vision tokens vs Prefill latency
- Component breakdown: T_vision, T_projector, T_LLM_prefill vs Vision tokens
- Cost per token: Slope of linear fit

### Experiment 4: Language Tokens vs Latency

**File**: `experiments/motivate/exp4_language_tokens_vs_latency.py`

**Key Implementation**:
- Uses fixed set of 10 images from VQA v2 validation set
- Varies `max_new_tokens`: [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] (default)
  - Covers a wide range from short (8 tokens) to very long (4096 tokens) outputs
- Measures T_LLM_decode for each output length
- Computes FLOPs for each configuration
- Groups results by max_new_tokens for analysis

**Visualization**:
- Line plot: Output tokens vs Decode latency
- Stacked bar chart: Prefill + Decode latency breakdown
- Cost per token: Slope of linear fit

### Experiment 5: FLOPs vs Latency

**File**: `experiments/motivate/exp5_flops_vs_latency.py`

**Key Implementation**:
- Loads results from Exp 3 and Exp 4 JSON files
- Extracts FLOPs and latency data
- Computes correlation coefficients and R²
- Generates scatter plots with linear fits

**Visualization**:
- Scatter plot: FLOPs vs Latency (Vision scaling)
- Scatter plot: FLOPs vs Latency (Language scaling)
- Combined plot: All FLOPs vs Latency data
- Statistics: Correlation coefficients, R² values

## Notes

- All experiments use `num_runs=1` for Exp 1 and 2 (real dataset), and `num_runs=10` for Exp 3 (controlled scaling) for stability.
- Warmup is performed before measurements to avoid initialization costs.
- Results are saved in JSON format for further analysis.
- Visualizations are saved as PNG files with 300 DPI for publication quality.
- **Important**: `max_crops=12` is the maximum allowed, but actual `num_crops` is adaptively selected based on image size. Typical values are 7-10 crops for VQA v2 images. This is expected behavior to avoid unnecessary upscaling.
