# M3: Controller Runtime Overhead

## Goal

Measure end-to-end added overhead per request (controller compute + any reconfiguration overhead), and show that it is negligible vs savings.

## Key Questions

1. What is the controller decision time?
2. What is the reconfiguration overhead?
3. Is overhead negligible compared to latency savings?
4. How does overhead scale with number of candidates?

## What to Run

### Overhead Measurement

1. **Controller decision time**:
   - Measure CPU/GPU time for `controller.select_config()`
   - Test with different numbers of candidate configurations
   - Test with/without latency estimator

2. **Reconfiguration overhead**:
   - Time to apply configuration (set vision_tokens, top_k, active_blocks)
   - Time to apply BlockMaskWrapper
   - Total reconfiguration time

3. **Total overhead**:
   - Controller time + reconfiguration time
   - Compare to: Latency savings from adaptive selection

### Measurement Protocol

```python
# Controller overhead
start = time.perf_counter()
config = controller.select_config(request_features)
controller_time = (time.perf_counter() - start) * 1000

# Reconfiguration overhead
start = time.perf_counter()
apply_config(model, config)
reconfig_time = (time.perf_counter() - start) * 1000

total_overhead = controller_time + reconfig_time
```

## Key Outputs/Plots

### 1. Overhead Breakdown

**Plot**: Stacked bar chart
- X-axis: Different controller methods or candidate counts
- Y-axis: Time (ms)
- Stacked: Controller decision, Reconfiguration
- **Insight**: Overhead components

### 2. Overhead vs Savings

**Plot**: Comparison
- X-axis: Different configurations or budgets
- Y-axis: Time (ms)
- Multiple bars: Overhead, Latency savings
- **Insight**: Overhead << savings

### 3. Overhead Scaling

**Plot**: Overhead vs number of candidates
- X-axis: Number of candidate configurations
- Y-axis: Controller decision time (ms)
- **Insight**: How overhead scales

## Expected Findings

1. **Low overhead**: < 1-5 ms typically
2. **Negligible vs savings**: Overhead << 10% of savings
3. **Scales with candidates**: More candidates = more time
4. **Estimator helps**: Reduces candidates, reduces overhead

## Code References

- **Controller**: E5 implementations
- **Reconfiguration**: `docs/knobs/` documents

## Output Files

- `m3_overhead_measurements.json`: Overhead data
- `figures/m3_overhead_breakdown.png`: Overhead components
- `figures/m3_overhead_vs_savings.png`: Comparison

