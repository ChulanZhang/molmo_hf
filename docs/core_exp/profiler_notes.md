# PyTorch Profiler Notes

## Does PyTorch Profiler Affect Inference Speed?

**Short answer**: Yes, but the overhead can be minimized.

### Overhead Analysis

1. **Default profiler (with_stack=True, record_shapes=True)**:
   - Overhead: ~10-30% slower
   - Records detailed call stacks and tensor shapes
   - Useful for debugging but too slow for production profiling

2. **Minimal profiler (with_stack=False, record_shapes=False)**:
   - Overhead: ~2-5% slower
   - Records only function names and timing
   - Acceptable for production profiling

3. **No profiler**:
   - Use manual timing with `time.perf_counter()` or `torch.cuda.Event()`
   - Overhead: < 0.1%
   - Recommended for latency profiling experiments

### Recommendations

**For E1 (Stage Decomposition)**:
- Use **manual timing** for production runs (measure T_vision, T_prefill, T_decode separately)
- Use **minimal profiler** only for initial exploration to understand component breakdown
- Use profiler results to identify measurement points, then switch to manual timing

**For E2/E3 (Accuracy/Latency Profiling)**:
- Use **manual timing only** (no profiler)
- Record stage latencies using forward hooks or subtraction method
- Profiler overhead would skew latency measurements

**For Debugging**:
- Use full profiler with `with_stack=True` to understand call hierarchy
- Run on small subset of data only

### Implementation Strategy

```python
# Option 1: Manual timing (recommended for experiments)
def measure_stage_latency(model, inputs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # Vision encoding
    vision_start = time.perf_counter()
    vision_features = model.encode_image(inputs['images'])
    torch.cuda.synchronize()
    vision_time = (time.perf_counter() - vision_start) * 1000
    
    # Prefill
    prefill_start = time.perf_counter()
    outputs = model(**inputs, use_cache=True)
    torch.cuda.synchronize()
    prefill_time = (time.perf_counter() - prefill_start) * 1000
    
    # Decode (per token)
    decode_times = []
    for _ in range(num_tokens):
        decode_start = time.perf_counter()
        outputs = model.generate_step(...)
        torch.cuda.synchronize()
        decode_times.append((time.perf_counter() - decode_start) * 1000)
    
    return {
        'vision_time': vision_time,
        'prefill_time': prefill_time,
        'decode_times': decode_times
    }

# Option 2: Minimal profiler (for exploration only)
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CUDA],
    record_shapes=False,  # Disable shape recording
    with_stack=False,     # Disable stack recording
    profile_memory=False  # Disable memory profiling
) as prof:
    outputs = model.generate(...)

# Analyze results
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Best Practice

1. **Development phase**: Use minimal profiler to understand component breakdown
2. **Production profiling**: Use manual timing with `torch.cuda.synchronize()` for accurate measurements
3. **Never use full profiler** in production runs (too slow)

## Dataset Sampling

### Can We Sample the Dataset?

**Yes, with proper sampling strategy.**

### Sampling Strategy

1. **Random sampling** (recommended):
   - Randomly sample N samples from validation set
   - Preserves distribution of image sizes, question types, etc.
   - **Does not change Pareto frontier shape** if sample size is sufficient

2. **Stratified sampling**:
   - Sample proportionally across different categories
   - Ensures representation of all types
   - More complex but better for heterogeneous datasets

3. **Minimum sample size**:
   - For accuracy: ~1000-2000 samples sufficient for stable estimates
   - For latency: ~500-1000 samples sufficient (less variance)
   - For Pareto frontier: ~500-1000 samples per configuration

### Impact on Pareto Frontier

**Sampling does NOT change Pareto frontier trends** if:
- Sample size is sufficient (≥500 samples per config)
- Sampling is random (not biased)
- Same samples used across all configurations (for fair comparison)

**Why it's safe**:
- Pareto frontier is about **relative** performance between configurations
- Random sampling preserves relative ordering
- Statistical noise is averaged out across configurations

### Recommended Sampling

```python
# For accuracy profiling (exp5)
NUM_SAMPLES = 2000  # Sufficient for stable accuracy estimates

# For latency profiling (exp6)
NUM_SAMPLES = 1000  # Sufficient for stable latency estimates

# For Pareto frontier analysis (E2)
NUM_SAMPLES_PER_CONFIG = 500  # Minimum for stable comparison
```

### Implementation

```python
def create_sampled_dataset(dataset, num_samples: int, seed: int = 42):
    """Create randomly sampled dataset."""
    if num_samples is None or num_samples >= len(dataset):
        return dataset
    
    # Use deterministic random sampling
    indices = np.random.RandomState(seed=seed).choice(
        len(dataset),
        size=num_samples,
        replace=False
    )
    
    return torch.utils.data.Subset(dataset, indices)
```

