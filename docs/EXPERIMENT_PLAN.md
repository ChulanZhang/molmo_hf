# Motivational Study Experiment Plan

## 1. Overview
This study aims to systematically analyze the latency characteristics of Vision-Language Models (VLMs) to motivate the need for content- and resource-aware adaptive inference.

## 2. Methodology Structure

The experiments are organized into three logical phases:

### Phase 1: Real-World Profiling (The "What")
**Goal**: Establish a baseline by measuring latency on a real-world dataset.
- **Dataset**: VQA v2 Validation Set (Diverse images, short answers).
- **Activities**:
    - Run inference on a large subset (e.g., 5k samples).
    - Measure `T_total`, `T_vision`, `T_projector`, `T_LLM_prefill`, `T_LLM_decode`.
- **Analysis**:
    - **Exp 1 (Distribution)**: Histograms and CDFs of latency. Identify P50, P95, P99 (Tail Latency).
    - **Exp 2 (Correlation)**: FLOPs vs. Latency scatter plot. Show that FLOPs are not a perfect predictor.
    - **Exp 3 (Breakdown)**: Pie charts showing the time/parameter cost of Vision vs. Language components.

### Phase 2: Mechanism Study (The "Why")
**Goal**: Isolate variables to understand the cost mechanism of each component. These are controlled experiments.

#### 2A. Vision Scaling (Prefill Cost)
- **Variable**: Image Resolution (Vision Tokens).
- **Fixed**: Text Prompt, Output Length.
- **Method**: Resize a dummy image to `[336, 504, ..., 2352]` px.
- **Metric**: `T_vision`, `T_projector`, `T_LLM_prefill`.
- **Insight**: Quantify the cost of adding vision tokens (mostly parallelizable prefill).

#### 2B. Decode Scaling (Generation Cost)
- **Variable**: Output Token Length.
- **Fixed**: Image (Fixed resolution), Prompt.
- **Method**: Force generation of `[12, 32, 64, 128, 256, 512]` tokens.
- **Metric**: `T_LLM_decode`.
- **Insight**: Quantify the cost of generating language tokens (sequential decode).

### Phase 3: Comparative Synthesis (Exp 5)
**Goal**: Compare the "Cost per Token" across different modalities.
- **Method**: Combine data from Phase 2A and 2B.
- **Analysis**:
    - Plot "Latency Increase" vs. "Number of Tokens Added".
    - Compare the slope of **Vision Tokens** (Prefill) vs. **Output Tokens** (Decode).
- **Hypothesis**: Decoding tokens is significantly more expensive per unit than vision tokens due to memory bandwidth and sequential execution.

## 3. Execution Plan

### Step 1: Run Unified Experiments
Use the provided shell scripts to collect data for all phases efficiently.

```bash
# Phase 1: Dataset Profiling (Exp 1 & 3)
bash experiments/motivate/run_phase1.sh

# Phase 2: Controlled Scaling (Exp 2, 4a, 5)
bash experiments/motivate/run_phase2.sh
```

### Step 2: Analyze Results
1.  **Exp 1-3**: Analyze `results/phase1-5k/phase1_dataset_profiling.json`.
2.  **Exp 4A**: Analyze `results/phase2/phase2_scaling.json`.
3.  **Exp 4B**: Analyze `results/exp4/exp4b_language_tokens.json` (if running Exp 4 separately).
4.  **Exp 5**: Run the synthesis script to generate the comparison plot:
    ```bash
    python experiments/motivate/exp5_token_comparison.py \
        --phase2_results results/phase2/phase2_scaling.json \
        --phase3_results results/exp4/exp4b_language_tokens.json
    ```

## 4. Data Format
All results are saved as JSON with the following schema:
```json
{
  "experiment_name": "...",
  "results": [
    {
      "T_total": 123.4,
      "T_vision": 10.1,
      "T_projector": 5.2,
      "T_LLM_prefill": 20.3,
      "T_LLM_decode": 80.5,
      "num_vision_tokens": 256,
      "num_output_tokens": 16
    }
  ]
}
```
