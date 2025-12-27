# M1: Prefill-Heavy vs Decode-Heavy Workload Contrast

## Goal

Show that QA-like workloads are often prefill-sensitive, while captioning/long-form are decode-sensitive, producing different Pareto structures.

## Key Questions

1. How do latency characteristics differ between QA and captioning tasks?
2. Which stage dominates in each workload type?
3. Do optimal configurations differ between workload types?
4. How does Pareto frontier structure differ?

## What to Run

### Workload Selection

1. **Prefill-heavy workload**: QA tasks
   - Dataset: VQA v2 validation
   - Characteristics:
     - Short answers (1-10 tokens)
     - Prefill-dominated latency
     - High vision token sensitivity

2. **Decode-heavy workload**: Captioning tasks
   - Dataset: COCO Captions validation
   - Characteristics:
     - Long outputs (20-100+ tokens)
     - Decode-dominated latency
     - Lower vision token sensitivity

### Measurements

For each workload, measure:
- **Stage breakdown**: T_vision, T_projector, T_prefill, T_decode (from E1)
- **Latency distribution**: Mean, P50, P95, P99
- **Optimal configurations**: From E2 Pareto frontiers
- **Sensitivity analysis**: How each knob affects latency

## Key Outputs/Plots

### 1. Stage Breakdown Comparison

**Plot**: Stacked bar chart comparing workloads
- X-axis: Workload type (QA, Captioning)
- Y-axis: Latency (ms)
- Stacked segments: Vision, Projector, Prefill, Decode
- **Insight**: Prefill dominates QA, Decode dominates Captioning

### 2. Latency Distribution Comparison

**Plot**: CDF or histogram
- X-axis: Latency (ms)
- Y-axis: Cumulative probability or frequency
- Multiple lines: QA vs Captioning
- **Insight**: Different latency distributions

### 3. Pareto Frontier Comparison

**Plot**: Side-by-side Pareto frontiers
- Left: QA workload frontier
- Right: Captioning workload frontier
- X-axis: Latency (ms)
- Y-axis: Quality (accuracy/score)
- **Insight**: Different optimal configurations

### 4. Knob Sensitivity Comparison

**Plot**: Sensitivity heatmap or bar chart
- X-axis: Knob (vision_tokens, top_k, active_blocks)
- Y-axis: Latency change (%)
- Grouped bars: QA vs Captioning
- **Insight**: Different sensitivity patterns

## Implementation Details

Use results from E1 and E2:
- E1 provides stage breakdowns
- E2 provides Pareto frontiers per workload

## Expected Findings

1. **QA is prefill-sensitive**: Prefill latency dominates
2. **Captioning is decode-sensitive**: Decode latency dominates
3. **Different optimal configs**: QA prefers lower vision tokens, Captioning prefers fewer blocks
4. **Different Pareto shapes**: QA frontier steeper, Captioning frontier flatter

## Code References

- **E1 results**: Stage decomposition data
- **E2 results**: Pareto frontiers per workload

## Output Files

- `m1_workload_contrast.json`: Comparison results
- `figures/m1_stage_breakdown.png`: Stage comparison
- `figures/m1_pareto_comparison.png`: Frontier comparison

