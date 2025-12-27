# M2: Latency Repeatability (Variance) at Fixed Configuration

## Goal

Run repeated trials at identical config (batch=1) and report std/CoV; justify how many repeats are needed for profiling points.

## Key Questions

1. How much variance is there in latency measurements?
2. How many repeats are needed for stable measurements?
3. Does variance differ across configurations?
4. What is the coefficient of variation (CoV)?

## What to Run

### Measurement Protocol

1. **Select representative configurations**:
   - Low latency: Small vision tokens, low top_k, few blocks
   - Medium latency: Medium settings
   - High latency: Large vision tokens, high top_k, many blocks

2. **Repeated measurements**:
   - Fixed configuration
   - Batch size = 1
   - Warmup: 5 runs (discard)
   - Timed: 50-100 runs per configuration
   - Measure: T_total, T_vision, T_prefill, T_decode

3. **Statistics**:
   - Mean, std, CoV (std/mean)
   - P50, P95, P99
   - Distribution shape

## Key Outputs/Plots

### 1. Latency Distribution

**Plot**: Histogram of latency measurements
- X-axis: Latency (ms)
- Y-axis: Frequency
- Multiple subplots: One per configuration
- **Insight**: Distribution shape and variance

### 2. CoV vs Configuration

**Plot**: Coefficient of variation across configurations
- X-axis: Configuration (or mean latency)
- Y-axis: CoV (%)
- **Insight**: Which configurations have higher variance

### 3. Convergence Analysis

**Plot**: Mean and std vs number of samples
- X-axis: Number of samples
- Y-axis: Mean latency or std
- **Insight**: How many samples needed for stable measurement

## Expected Findings

1. **Low variance**: CoV typically < 5-10%
2. **10-20 samples sufficient**: For stable mean
3. **More samples for tail**: P95/P99 need more samples
4. **Variance differs**: Some configs more variable than others

## Code References

- **Latency measurement**: `experiments/base_experiment.py::measure_inference_latency()`

## Output Files

- `m2_repeatability_results.json`: Variance measurements
- `figures/m2_latency_distribution.png`: Histograms
- `figures/m2_cov_analysis.png`: CoV across configs

