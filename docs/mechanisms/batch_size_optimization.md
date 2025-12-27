# Batch Size Optimization

## Overview

This document explains the automatic batch size adjustment mechanism used in accuracy profiling experiments. The system automatically finds the maximum usable batch size for each configuration to avoid OOM (Out of Memory) errors while maximizing throughput.

## Key Concept

**Important**: Automatic batch size adjustment does **not** start from `--batch_size` and test downward. Instead:
1. **Estimates** an initial batch size based on the configuration (e.g., `max_crops`, `top_k`, `num_active_blocks`)
2. Uses **binary search** starting from the estimate to find the maximum usable batch size
3. **Each configuration is tested independently** - batch sizes for different configurations don't affect each other

## Detailed Algorithm

### Step 1: Initial Estimation

For each configuration value, estimate initial batch size using empirical formulas:

#### For Vision Tokens (max_crops)

```python
def estimate_batch_size_for_max_crops(max_crops: int, base_batch_size: int) -> int:
    """Estimate batch size based on max_crops."""
    if max_crops <= 2:
        scale_factor = 1.0      # Use 100% of base batch size
    elif max_crops <= 4:
        scale_factor = 0.8      # Use 80%
    elif max_crops <= 6:
        scale_factor = 0.6      # Use 60%
    elif max_crops <= 8:
        scale_factor = 0.5      # Use 50%
    elif max_crops <= 10:
        scale_factor = 0.4      # Use 40%
    elif max_crops <= 12:
        scale_factor = 0.3      # Use 30%
    else:
        scale_factor = 0.25     # Use 25%
    
    estimated = int(base_batch_size * scale_factor)
    return min(estimated, base_batch_size)  # Don't exceed base
```

**Scale factor table**:
| max_crops | scale_factor | Example (base=64) |
|-----------|--------------|-------------------|
| ≤ 2 | 1.0 | 64 |
| ≤ 4 | 0.8 | 51 |
| ≤ 6 | 0.6 | 38 |
| ≤ 8 | 0.5 | 32 |
| ≤ 10 | 0.4 | 25 |
| ≤ 12 | 0.3 | 19 |
| > 12 | 0.25 | 16 |

#### For MoE Top-K

```python
def estimate_batch_size_for_top_k(top_k: int, base_batch_size: int) -> int:
    """Estimate batch size based on MoE top_k."""
    # Top-K has less impact than max_crops
    if top_k <= 2:
        scale_factor = 1.0
    elif top_k <= 4:
        scale_factor = 0.9
    elif top_k <= 8:
        scale_factor = 0.8
    else:
        scale_factor = 0.7
    
    estimated = int(base_batch_size * scale_factor)
    return min(estimated, base_batch_size)
```

#### For Transformer Blocks

```python
def estimate_batch_size_for_blocks(num_active: int, total_blocks: int, base_batch_size: int) -> int:
    """Estimate batch size based on number of active blocks."""
    depth_ratio = num_active / total_blocks
    # Linear scaling with depth
    scale_factor = depth_ratio
    estimated = int(base_batch_size * scale_factor)
    return min(estimated, base_batch_size)
```

### Step 2: Binary Search

Starting from the estimate, use binary search to find the maximum usable batch size:

#### Scenario A: Estimate is too small (can increase)

```
Try batch_size=19 → ✅ Success
  → Try increasing to 19 * 1.5 = 28
Try batch_size=28 → ✅ Success
  → Try increasing to 28 * 1.5 = 42
Try batch_size=42 → ✅ Success
  → Try increasing to 42 * 1.5 = 63 (not exceeding 64)
Try batch_size=63 → ✅ Success
  → Return 63 (maximum usable value)
```

#### Scenario B: Estimate is too large (need to decrease)

```
Try batch_size=19 → ❌ OOM
  → Decrease to 19 // 2 = 9
Try batch_size=9 → ✅ Success
  → Try increasing to (9 + 19) // 2 = 14
Try batch_size=14 → ✅ Success
  → Try increasing to (14 + 19) // 2 = 16
Try batch_size=16 → ✅ Success
  → Return 16 (maximum usable value)
```

#### Scenario C: Estimate is just right

```
Try batch_size=19 → ✅ Success
  → Try increasing to 19 * 1.5 = 28
Try batch_size=28 → ❌ OOM
  → Try decreasing to (19 + 28) // 2 = 23
Try batch_size=23 → ✅ Success
  → Return 23 (maximum usable value)
```

### Step 3: Testing Method

For each candidate batch size:
1. Create dataloader with the batch size
2. Load one batch
3. Execute forward pass **with generation** (critical for accurate memory estimation)
4. Check for OOM

**Important**: The test includes generation because generation requires more memory than forward pass alone.

## Usage

### Basic Usage

```bash
# Enable automatic batch size adjustment
python experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --batch_size 64 \
    --auto_adjust_batch_size
```

### Start from Specific Configuration

```bash
# Start from max_crops=12, automatically adjust batch size
python experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --batch_size 64 \
    --start_from_max_crops 12 \
    --auto_adjust_batch_size
```

### Specify Configuration List

```bash
# Test specific max_crops values with auto adjustment
python experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --batch_size 64 \
    --max_crops_list 12 13 14 \
    --auto_adjust_batch_size
```

### Disable Auto Adjustment

```bash
# Use fixed batch size, manual control
python experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --batch_size 32 \
    --no_auto_adjust_batch_size
```

## Performance Impact

### Overhead

- **Time cost**: Each configuration requires 1-5 tests (each ~1-2 seconds)
- **Total overhead**: ~10-30 seconds per configuration (depending on number of attempts)
- **Maximum attempts**: 5 (to avoid infinite loops)

### Benefits

- **Avoid OOM**: Prevents crashes from batch size being too large
- **Maximize performance**: Automatically finds maximum usable batch size for each configuration
- **Automation**: No manual adjustment needed
- **Independent per configuration**: Each configuration finds its own optimal batch size

## Example Execution Flow

### Command

```bash
torchrun --nproc-per-node=4 experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --batch_size 64 \
    --start_from_max_crops 12 \
    --auto_adjust_batch_size
```

### For max_crops=12:

1. **Initial estimation**:
   ```
   estimated = 64 * 0.3 = 19
   starting = min(19, 64) = 19
   ```

2. **Testing process** (example):
   ```
   ✓ Batch size 19 works
     → Trying larger batch size: 28...
   ✓ Batch size 28 works
     → Trying larger batch size: 42...
   ✓ Batch size 42 works
     → Trying larger batch size: 63...
   ✓ Batch size 63 works
   Using batch size: 63 for max_crops=12
   ```

3. **Actual run**: Use batch_size=63 for full accuracy test

### For max_crops=13:

1. **Initial estimation**:
   ```
   estimated = 64 * 0.25 = 16
   starting = min(16, 64) = 16
   ```

2. **Testing process** (example):
   ```
   ✓ Batch size 16 works
     → Trying larger batch size: 24...
   ✓ Batch size 24 works
     → Trying larger batch size: 36...
   ✗ Batch size 36 caused OOM
     → Trying 24...
   Using batch size: 24 for max_crops=13
   ```

3. **Actual run**: Use batch_size=24 for full accuracy test

## Key Points

### 1. Not Downward Testing

❌ **Wrong understanding**: Start from batch_size=64, if OOM reduce to 32, if OOM reduce to 16...

✅ **Correct understanding**:
- For max_crops=12: Start from estimate 19, search **upward** to maximum usable value (may be 63)
- For max_crops=13: Start from estimate 16, search **upward** to maximum usable value (may be 24)

### 2. Independent per Configuration

- `max_crops=12` and `max_crops=13` batch sizes are **completely independent**
- max_crops=12 using 63 doesn't mean max_crops=13 must be smaller
- Each finds its own maximum usable value

### 3. Binary Search Strategy

- **If successful**: Try increasing (×1.5) until reaching base_batch_size or OOM
- **If failed**: Use binary search between known "working value" and "failing value"
- **Maximum attempts**: 5 to avoid infinite loops

### 4. Test Includes Generation

Each test:
1. Creates dataloader
2. Loads one batch
3. Executes forward pass
4. **Executes generation** (critical - generation needs more memory)

This ensures the found batch size won't OOM during actual runs.

## Why This Design?

### Advantages

1. **Efficient**: Don't need to start from large value and test downward, saves time
2. **Smart**: Automatically estimates based on configuration, usually very close to optimal
3. **Safe**: Tests include generation, ensuring actual runs won't OOM
4. **Maximize performance**: Finds maximum usable batch size for each configuration

### Comparison

#### Method A: Downward from base (inefficient)

```
max_crops=12:
  Try 64 → OOM
  Try 32 → OOM
  Try 16 → ✅ Success
  Use 16 (but 20-30 might also work!)
```

#### Method B: Upward from estimate (current method, efficient)

```
max_crops=12:
  Estimate 19 → ✅ Success
  Try 28 → ✅ Success
  Try 42 → ✅ Success
  Try 63 → ✅ Success
  Use 63 (found maximum usable value!)
```

## Log Interpretation

You'll see logs like:

```
INFO:__main__:Testing max_crops=12
INFO:__main__:Finding optimal batch size for max_crops=12 (estimated: 19, starting with: 19)...
INFO:__main__:✓ Batch size 19 works for max_crops=12
INFO:__main__:  Trying larger batch size: 28...
INFO:__main__:✓ Batch size 28 works for max_crops=12
INFO:__main__:  Trying larger batch size: 42...
INFO:__main__:✓ Batch size 42 works for max_crops=12
INFO:__main__:  Trying larger batch size: 63...
INFO:__main__:✓ Batch size 63 works for max_crops=12
INFO:__main__:Using batch size: 63 for max_crops=12
INFO:__main__:Measuring accuracy for max_crops=12...
```

This indicates:
- Estimated value 19 works
- Gradually increased to 63, all work
- Finally use 63 for full accuracy test

## Result Recording

The result JSON file records the actual batch size used for each configuration:

```json
{
  "summary": [
    {
      "max_crops": 12,
      "accuracy": 0.85,
      "batch_size_used": 63,  // Actual batch size used
      ...
    }
  ]
}
```

## Tuning Recommendations

If you find the auto-adjusted batch size is too conservative or too aggressive, you can modify the scale factors in the estimation functions.

For example, if max_crops=12 with batch_size=32 is very safe, you can adjust:

```python
elif max_crops <= 12:
    scale_factor = 0.5  # Increase from 0.3 to 0.5
```

## Troubleshooting

### Issue 1: Auto adjustment always fails

**Possible causes**:
- Base batch size too large
- GPU memory insufficient

**Solutions**:
- Reduce `--batch_size` parameter
- Check GPU memory usage

### Issue 2: Auto adjustment too slow

**Possible causes**:
- Too many test attempts (max_attempts=5)

**Solutions**:
- Modify `max_attempts` parameter in code (default 5)
- Or disable auto adjustment, manually set smaller batch size

### Issue 3: Some configurations still OOM

**Possible causes**:
- Auto adjustment found working batch size, but actual generation needs more memory

**Solutions**:
- Auto adjustment is conservative, but generation may need more memory
- If encountered, manually reduce batch size for that configuration

## Summary

1. **Not downward testing**: Start from estimate, search upward for maximum usable value
2. **Independent per configuration**: Don't affect each other, each finds optimal value
3. **Smart estimation**: Automatically estimates based on configuration, usually accurate
4. **Binary search**: Efficiently finds maximum usable batch size
5. **Safe testing**: Includes generation test, ensuring actual runs won't OOM

**Your command**:
- `--batch_size 64`: This is the **upper limit**, won't exceed this value
- `--start_from_max_crops 12`: Start testing from max_crops=12
- `--auto_adjust_batch_size`: Automatically find optimal batch size for each max_crops

## Code References

- **Implementation**: `experiments/base_experiment.py`
  - `_find_optimal_batch_size()`: Main batch size optimization logic
  - `_estimate_batch_size_for_max_crops()`: Estimation for max_crops
  - `_estimate_batch_size_for_top_k()`: Estimation for top_k
  - `_estimate_batch_size_for_num_blocks()`: Estimation for num_active_blocks

## Related Documents

- `../knobs/vision_tokens_knob.md`: Vision tokens control and limits
- `../knobs/moe_topk_knob.md`: MoE top-K control
- `../knobs/transformer_blocks_knob.md`: Transformer blocks control

