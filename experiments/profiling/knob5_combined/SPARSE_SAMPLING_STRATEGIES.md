# Exp5 Sparse Sampling Strategies

## Overview

Exp5 tests combinations of three control knobs:
- **max_crops**: Vision tokens (range: 2-12, step 2) → [2, 4, 6, 8, 10, 12] (6 values)
- **top_k**: MoE expert selection (range: 4-32, step 4) → [4, 8, 12, 16, 20, 24, 28, 32] (8 values)
- **num_active_blocks**: Transformer depth (range: 8-16, step 2) → [8, 10, 12, 14, 16] (5 values)

**Full grid search**: 6 × 8 × 5 = **240 combinations** (too many!)

## Sparse Sampling Strategies

### 1. **Stratified Sampling** (27 combinations)

Selects min, middle, and max values from each dimension:
- max_crops: [2, 6, 12]
- top_k: [4, 16, 32]
- num_active_blocks: [8, 12, 16]

**Total**: 3 × 3 × 3 = **27 combinations**

**Pros**: Very sparse, covers boundaries well
**Cons**: May miss important intermediate values

### 2. **Boundary Sampling** (45 combinations)

More comprehensive boundary coverage:
- max_crops: [2, 6, 12] (min, middle, max)
- top_k: [4, 8, 16, 24, 32] (min, 1/4, middle, 3/4, max)
- num_active_blocks: [8, 12, 16] (min, middle, max)

**Total**: 3 × 5 × 3 = **45 combinations**

**Pros**: Better coverage of top_k dimension
**Cons**: Still relatively sparse

### 3. **Balanced Sampling** (36 combinations) ⭐ RECOMMENDED

Balanced coverage ensuring each dimension is well-represented:
- max_crops: [2, 6, 12] (3 values)
- top_k: [4, 12, 20, 32] (4 values, includes values close to default 8)
- num_active_blocks: [8, 12, 16] (3 values)

**Total**: 3 × 4 × 3 = **36 combinations**

**Pros**: 
- Good balance between sparsity and coverage
- Includes important values (e.g., top_k=12 is close to default 8)
- Each dimension is well-represented

**Cons**: None significant

### 4. **Custom Sparse** (36 combinations)

Takes every 2nd value from each dimension:
- max_crops: [2, 6, 12] (every 2nd: 2, 6, 12)
- top_k: [4, 12, 20, 32] (every 2nd: 4, 12, 20, 32)
- num_active_blocks: [8, 12, 16] (every 2nd: 8, 12, 16)

**Total**: 3 × 4 × 3 = **36 combinations**

**Pros**: Simple, uniform spacing
**Cons**: May miss important boundary values

### 5. **Latin Hypercube Sampling (LHS)** (customizable)

Randomly samples combinations ensuring each dimension is well-covered:
- Uses random sampling with seed=42 for reproducibility
- Default: ~60 combinations (1/4 of full grid)

**Pros**: 
- Ensures uniform coverage of each dimension
- Can specify exact number of combinations
- Good for exploring unknown parameter spaces

**Cons**: 
- Less interpretable (random combinations)
- May not include all important boundary values

### 6. **Full Grid Search** (240 combinations)

Tests all combinations:
- max_crops: [2, 4, 6, 8, 10, 12] (all 6 values)
- top_k: [4, 8, 12, 16, 20, 24, 28, 32] (all 8 values)
- num_active_blocks: [8, 10, 12, 14, 16] (all 5 values)

**Total**: 6 × 8 × 5 = **240 combinations**

**Pros**: Complete coverage
**Cons**: Very time-consuming (may take days/weeks)

## Recommendation

For most use cases, **"balanced"** sampling strategy is recommended:
- **36 combinations** (15% of full grid)
- Good balance between coverage and efficiency
- Includes important values (boundaries, middle, near-default)
- Reasonable runtime (can complete in 1-2 days with multi-GPU)

## Usage Examples

### Balanced Sampling (Recommended)
```bash
python experiments/profiling/knob5_combined/exp5_accuracy.py \
    --model_path checkpoints \
    --output_dir ./results/profiling/exp5_accuracy \
    --sampling_strategy balanced \
    --batch_size 8
```

### Stratified Sampling (Fastest)
```bash
python experiments/profiling/knob5_combined/exp5_accuracy.py \
    --sampling_strategy stratified \
    --batch_size 8
```

### Boundary Sampling (More Comprehensive)
```bash
python experiments/profiling/knob5_combined/exp5_accuracy.py \
    --sampling_strategy boundary \
    --batch_size 8
```

### Custom Sparse
```bash
python experiments/profiling/knob5_combined/exp5_accuracy.py \
    --sampling_strategy custom_sparse \
    --batch_size 8
```

### Latin Hypercube Sampling
```bash
python experiments/profiling/knob5_combined/exp5_accuracy.py \
    --sampling_strategy lhs \
    --batch_size 8
```

### Full Grid Search (Complete Coverage)
```bash
python experiments/profiling/knob5_combined/exp5_accuracy.py \
    --sampling_strategy full \
    --batch_size 8
```

## Multi-GPU Usage

```bash
torchrun --nproc-per-node=4 experiments/profiling/knob5_combined/exp5_accuracy.py \
    --model_path checkpoints \
    --output_dir ./results/profiling/exp5_accuracy \
    --sampling_strategy balanced \
    --batch_size 8
```

## Comparison Table

| Strategy | Combinations | Coverage | Runtime | Use Case |
|----------|-------------|----------|---------|----------|
| **Stratified** | 27 | Low | Fastest | Quick exploration |
| **Boundary** | 45 | Medium | Fast | Better coverage |
| **Balanced** ⭐ | 36 | High | Medium | **Recommended** |
| **Custom Sparse** | 36 | Medium | Medium | Uniform spacing |
| **LHS** | ~60 | High | Medium | Random exploration |
| **Full** | 240 | Complete | Very Slow | Complete analysis |

## Further Sparsification

If you need even fewer combinations, you can:

1. **Reduce knob ranges**:
   - max_crops: [2, 6, 12] (3 values instead of 6)
   - top_k: [4, 16, 32] (3 values instead of 8)
   - num_active_blocks: [8, 12, 16] (3 values instead of 5)
   - Total: 3 × 3 × 3 = **9 combinations** (stratified)

2. **Use custom knob lists**:
   ```bash
   python experiments/profiling/knob5_combined/exp5_accuracy.py \
       --max_crops 2 6 12 \
       --top_k 4 16 32 \
       --num_active_blocks 8 12 16 \
       --sampling_strategy full
   ```
   This gives exactly 3 × 3 × 3 = **9 combinations** with full grid search.

3. **Focus on specific ranges**:
   - Only test high-performance configs: max_crops=[2,4], top_k=[4,8], blocks=[12,14,16]
   - Only test low-latency configs: max_crops=[2,4,6], top_k=[4,8,12], blocks=[8,10,12]


