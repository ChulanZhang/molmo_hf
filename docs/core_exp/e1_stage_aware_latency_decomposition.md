# E1: Stage-Aware Latency Decomposition + "FLOPs ≠ Wall-Clock"

## Goal

Quantify which stages dominate end-to-end latency and demonstrate why theoretical compute (FLOPs/params) is not predictive of measured latency.

## Key Questions

1. Which stages (vision encoder, projector, LLM prefill, LLM decode) dominate end-to-end latency?
2. Within LLM stages, which sub-components (Attention vs FFN, MoE routing/experts) dominate?
3. How does stage dominance change across different input regimes?
4. Why do FLOPs not accurately predict wall-clock latency?

## What to Run

### End-to-End Latency by Stage

Measure latency for each stage:
- **Vision encoder**: ViT processing time
- **Projector**: Image feature projection time
- **LLM prefill**: Transformer blocks processing all input tokens
- **LLM decode**: Per-token generation time

### Within LLM Stages Decomposition

Using PyTorch profiler, further decompose LLM stages:
- **Attention**: Self-attention computation time
- **FFN/MLP**: Feed-forward network time
- **MoE routing** (if applicable): Expert selection and routing time
- **MoE experts** (if applicable): Expert computation time

### Input Regime Sweep

Test representative input regimes:

1. **Image resolution settings** (drives vision tokens → prefill)
   - Low: 336×336 (1 crop, ~288 vision tokens)
   - Medium: 560×560 (4 crops, ~720 vision tokens)
   - High: 1008×784 (12 crops, ~1872 vision tokens)

2. **Prompt type/length** (affects prefill)
   - Short QA prompts: ~10-20 tokens
   - Medium prompts: ~30-50 tokens
   - Long captioning prompts: ~100+ tokens

3. **Decode length** (drives decode)
   - Short: 8-16 tokens (VQA answers)
   - Medium: 32-64 tokens (short captions)
   - Long: 128-256 tokens (long-form generation)

### Measurement Protocol

- **Batch size**: 1 (single request)
- **Warmup runs**: 3-5 (not recorded)
- **Timed runs**: 10-20 per configuration
- **Metrics**: Mean, median, P95, P99, std

## Key Outputs/Plots

### 1. Stack Plots: Stage Time Shares

**Plot**: Stacked bar chart showing time share of each stage
- X-axis: Different input regimes (e.g., low/medium/high vision tokens)
- Y-axis: Latency (ms)
- Stacked segments: Vision encoder, Projector, LLM prefill, LLM decode
- **Insight**: Shows which stage dominates in different regimes

### 2. Scaling Curves

**Plot 1**: Prefill latency vs (vision tokens + prompt tokens)
- X-axis: Total input tokens (vision + text)
- Y-axis: Prefill latency (ms)
- Multiple lines: Different decode lengths (to show decode doesn't affect prefill)
- **Insight**: Linear/sub-linear scaling relationship

**Plot 2**: Decode latency vs generated tokens
- X-axis: Number of generated tokens
- Y-axis: Decode latency (ms)
- Multiple lines: Different input token counts (to show prefill doesn't affect decode)
- **Insight**: Linear scaling per token

### 3. Attention vs FFN Breakdown

**Plot**: Component breakdown within LLM stages
- X-axis: Different configurations (vision tokens, prompt length, etc.)
- Y-axis: Latency (ms) or percentage
- Stacked/grouped bars: Attention, FFN, MoE routing, MoE experts
- **Insight**: Shows which sub-kernels dominate and how dominance changes

### 4. FLOPs vs Latency Scatter

**Plot**: FLOPs vs measured latency
- X-axis: Theoretical FLOPs (computed from model specs)
- Y-axis: Measured wall-clock latency (ms)
- Color: Different stages or configurations
- **Insight**: Demonstrates FLOPs ≠ wall-clock latency (scatter, not linear)

## Implementation Details

### Stage Measurement

Use `experiments/base_experiment.py::measure_inference_latency()`:
- `T_vision_encoder`: Direct measurement via `vision_backbone.encode_image()`
- `T_projector`: Derived from `T_vision_total - T_vision_encoder`
- `T_LLM_prefill`: Measured using forward hooks or subtraction method
- `T_LLM_decode`: Derived from `T_total - T_vision_total - T_LLM_prefill`

### PyTorch Profiler for Sub-Component Breakdown

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    with record_function("model_inference"):
        outputs = model.generate(...)

# Analyze profiler results
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### FLOPs Calculation

Estimate FLOPs for each stage:
- **Vision encoder**: `vision_params × num_patches × 2`
- **Projector**: `projector_params × num_tokens × 2`
- **LLM prefill**: `llm_params × input_length² × 2` (attention) + `llm_params × input_length × 2` (FFN)
- **LLM decode**: `llm_params × seq_length × 2` (per token)

## Expected Findings

1. **Prefill dominates** for high vision token counts
2. **Decode dominates** for long generation tasks
3. **Attention dominates** in prefill for long sequences (O(n²))
4. **FFN/MoE dominates** in decode (per-token computation)
5. **FLOPs correlation is weak** due to:
   - Memory access patterns
   - Hardware utilization
   - Parallelism opportunities
   - Cache effects

## Code References

- **Latency measurement**: `experiments/base_experiment.py::measure_inference_latency()`
- **Vision encoding**: `molmo/models/modeling_molmoe.py::encode_image()`
- **Model forward**: `molmo/models/modeling_molmoe.py::forward()`
- **Profiling**: Use PyTorch profiler for detailed breakdown

## Related Experiments

- **E2**: Uses stage decomposition to understand knob effects
- **E3**: Builds latency estimator based on stage-aware measurements
- **M1**: Contrasts prefill-heavy vs decode-heavy workloads

## Output Files

- `e1_stage_latency_breakdown.json`: Per-configuration stage latencies
- `e1_flops_vs_latency.json`: FLOPs and latency measurements
- `e1_attention_ffn_breakdown.json`: Sub-component breakdowns
- `figures/e1_stage_stack_plot.png`: Stacked bar chart
- `figures/e1_scaling_curves.png`: Prefill and decode scaling
- `figures/e1_flops_vs_latency.png`: FLOPs correlation scatter

