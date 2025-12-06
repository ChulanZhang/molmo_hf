# Experiment 3: Transformer Blocks Mask Profiling

## Overview

This experiment profiles the impact of the number of active transformer blocks on model latency. Unlike early exit mechanisms, this experiment uses a **mask-based approach** to skip certain transformer blocks during the forward pass without removing them from the model structure.

## Motivation

Understanding how model depth (number of transformer blocks) affects latency is crucial for:
- **Model optimization**: Identifying the optimal depth for a given latency budget
- **Dynamic depth adaptation**: Enabling runtime depth adjustment based on computational constraints
- **Performance scaling analysis**: Understanding the relationship between depth and latency

This experiment complements the other profiling experiments:
1. **Input tokens** (vision tokens) - controls input size
2. **MoE topK** - controls model width (active experts)
3. **Transformer blocks mask** - controls model depth (active blocks) ← **This experiment**
4. **Output tokens** - controls generation length

## Experimental Design

### Key Concept: Mask-Based Block Skipping

Instead of removing blocks from the model (which would require model reconstruction), we use a **mask mechanism** that:
- Keeps all blocks in the model structure
- Skips computation for masked blocks during forward pass
- Passes input directly through skipped blocks (identity function)
- Maintains proper cache handling for generation

### Implementation Approach

The experiment uses a `BlockMaskWrapper` class that:
1. **Monkey patches** the `MolmoModel.forward` method
2. **Intercepts** the blocks iteration loop
3. **Checks the mask** before each block execution
4. **Skips computation** for masked blocks (pass-through)
5. **Handles cache properly** for both active and skipped blocks

### Mask Strategy

The experiment supports two modes:

**Mode 1: Count-based (Default)**
- Activates blocks **from early to late** (first N blocks)
- **Default behavior**: Tests all block counts from 1 to total_blocks (e.g., 1, 2, 3, ..., 16)
- This provides comprehensive profiling of how latency scales with depth
- Can be customized with `--num_active_blocks` to test specific counts

This strategy is chosen because:
- Early blocks typically process more general features
- Later blocks focus on task-specific refinements
- This matches common early-exit strategies in the literature

**Mode 2: Specific Block Indices**
- Allows specifying exact block indices to activate
- Useful for testing different block combinations (e.g., early blocks, middle blocks, late blocks, or uniform sampling)
- Example: `--active_block_indices 0 5 10 15` activates blocks at positions 0, 5, 10, and 15

## Usage

### Mode 1: Count-based (Default)
Activate the first N blocks (from early to late):

```bash
# Use default: test all block counts from 1 to total_blocks (e.g., 1, 2, 3, ..., 16)
python experiments/profiling/knob3_layers/exp_transformer_blocks_mask.py \
    --model_path checkpoints \
    --output_dir ./results/transformer_blocks_mask \
    --num_samples 50

# Specify exact counts
python experiments/profiling/knob3_layers/exp_transformer_blocks_mask.py \
    --model_path checkpoints \
    --output_dir ./results/transformer_blocks_mask \
    --num_samples 50 \
    --num_active_blocks 4 8 12 16
```

### Mode 2: Specific Block Indices
Activate specific blocks by their indices:

```bash
# Test different block combinations
python experiments/profiling/knob3_layers/exp_transformer_blocks_mask.py \
    --model_path checkpoints \
    --output_dir ./results/transformer_blocks_mask \
    --num_samples 50 \
    --active_block_indices 0 1 2 3 \
    --active_block_indices 0 5 10 15 \
    --active_block_indices 4 8 12 16

# Test early, middle, and late blocks separately
python experiments/profiling/knob3_layers/exp_transformer_blocks_mask.py \
    --model_path checkpoints \
    --output_dir ./results/transformer_blocks_mask \
    --num_samples 50 \
    --active_block_indices 0 1 2 3 \
    --active_block_indices 6 7 8 9 \
    --active_block_indices 12 13 14 15
```

**Note**: `--active_block_indices` takes precedence over `--num_active_blocks` if both are provided.

### Bash Script Usage

```bash
# Count-based mode
bash experiments/profiling/run_exp3_transformer_blocks_mask.sh 0 \
    --num_samples 50 \
    --num_active_blocks 4 8 12 16

# Specific indices mode
bash experiments/profiling/run_exp3_transformer_blocks_mask.sh 0 \
    --num_samples 50 \
    --active_block_indices 0 1 2 3 \
    --active_block_indices 0 5 10 15
```

### Arguments

- `--model_path`: Path to model checkpoint directory (default: `checkpoints`)
- `--output_dir`: Output directory for results (default: `./results/transformer_blocks_mask`)
- `--num_samples`: Number of measurement repetitions per configuration (default: 50)
- `--num_active_blocks`: List of numbers of active blocks to test. If not provided, defaults to testing various fractions (full, 3/4, 1/2, 1/4) of total blocks. This activates the first N blocks (from early to late).
- `--active_block_indices`: List of specific block indices to activate. Can be specified multiple times to test different block combinations. This takes precedence over `--num_active_blocks`.

## Implementation Details

### BlockMaskWrapper Class

The `BlockMaskWrapper` class provides the core functionality:

```python
class BlockMaskWrapper:
    def __init__(self, model: MolmoModel, block_mask: torch.Tensor):
        """
        Args:
            model: The MolmoModel instance
            block_mask: Boolean tensor of shape (n_layers,) where True means the block is active
        """
```

**Key Methods:**
- `apply()`: Replaces `model.forward` with masked version
- `remove()`: Restores original `model.forward`
- `_masked_forward()`: Modified forward pass that respects the mask

### Masked Forward Pass

The masked forward pass:
1. **Replicates** the original `MolmoModel.forward` logic up to the blocks loop
2. **Modifies** the blocks iteration to check the mask:
   ```python
   for block_idx, layer in enumerate(model.transformer.blocks):
       if not self.block_mask[block_idx]:
           # Skip: pass through x without computation
           cache = ...  # Handle cache appropriately
       else:
           # Normal execution
           x, cache = layer(...)
   ```
3. **Handles cache** properly for skipped blocks (maintains structure for `use_cache=True`)
4. **Continues** with the rest of the forward pass (final layer norm, etc.)

### Cache Handling

For skipped blocks when `use_cache=True`:
- If `past_key_values` exists for the block, reuse it
- Otherwise, create a dummy cache with appropriate shape
- This ensures the cache structure remains consistent

**Note**: The dummy cache for skipped blocks is not used in actual computation but maintains the expected data structure.

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "summary": [
    {
      "num_active_blocks": 24,
      "num_total_blocks": 24,
      "active_block_indices": [0, 1, 2, ..., 23],
      "prefill": {
        "P50": 123.45,
        "P95": 145.67,
        "P99": 156.78,
        "mean": 125.12,
        "std": 8.34,
        "min": 110.23,
        "max": 160.45
      },
      "decode": {
        "P50": 12.34,
        "P95": 14.56,
        "P99": 15.67,
        "mean": 12.45,
        "std": 0.89,
        "min": 11.23,
        "max": 16.45
      }
    },
    ...
  ],
  "all_samples": [
    {
      "sample_id": 0,
      "num_active_blocks": 24,
      "num_total_blocks": 24,
      "active_block_indices": [0, 1, 2, ..., 23],
      "T_LLM_prefill": 123.45,
      "T_LLM_decode": 12.34,
      "T_vision_total": 45.67,
      "T_total": 181.46,
      "num_input_text_tokens": 10,
      "num_vision_tokens": 144,
      "num_output_tokens": 10,
      ...
    },
    ...
  ]
}
```

- **`summary`**: Statistical summary for each configuration (mean, std, percentiles)
- **`all_samples`**: Complete per-sample results with all metrics from `measure_inference_latency`

## Expected Results

### Latency Scaling

We expect to observe:
- **Linear scaling** (approximately) for prefill latency with number of active blocks
- **Linear scaling** (approximately) for decode latency per token with number of active blocks
- **Diminishing returns** in model quality as depth increases (not measured in this experiment, but important for full analysis)

### Typical Observations

For a model with 24 blocks:
- **24 blocks (full)**: Baseline latency
- **18 blocks (75%)**: ~75% of full latency
- **12 blocks (50%)**: ~50% of full latency
- **6 blocks (25%)**: ~25% of full latency

**Note**: Actual scaling may not be perfectly linear due to:
- Overhead from mask checking
- Cache management overhead
- GPU utilization patterns
- Memory bandwidth constraints

## Comparison with Early Exit

This experiment differs from **early exit** mechanisms:

| Aspect | Mask-Based (This Experiment) | Early Exit |
|--------|-------------------------------|------------|
| **Model Structure** | All blocks remain in model | All blocks remain in model |
| **Computation** | Blocks are skipped via mask | Computation stops at exit point |
| **Cache Handling** | All blocks maintain cache structure | Only active blocks have cache |
| **Use Case** | Profiling and analysis | Runtime optimization |
| **Flexibility** | Can skip any subset of blocks | Typically exits at fixed points |

## Technical Considerations

### Performance Overhead

The mask-based approach introduces minimal overhead:
- **Mask checking**: O(1) per block (boolean tensor access)
- **Conditional branching**: Negligible in modern CPUs/GPUs
- **Cache handling**: Slight overhead for skipped blocks

### Limitations

1. **Cache shape assumptions**: Dummy caches for skipped blocks may not perfectly match actual cache shapes (especially for multi-query/group-query attention)
2. **Not production-ready**: This is a profiling tool, not optimized for production use
3. **Mask strategy**: Currently only supports "first N blocks" strategy; other strategies (uniform, last N, etc.) can be added

### Future Improvements

Potential enhancements:
1. **Multiple mask strategies**: Uniform sampling, last N blocks, learned masks
2. **Cache shape detection**: Automatically detect correct cache shapes for dummy caches
3. **Production optimization**: Optimize for lower overhead in production settings
4. **Quality metrics**: Add model quality measurements (accuracy, perplexity) alongside latency

## Related Experiments

- **Exp 1**: Context Scaling (input tokens)
- **Exp 2**: MoE Top-K Analysis (model width)
- **Exp 4**: Output Tokens (generation length)

## References

- See `PROFILING_EXPERIMENTS.md` for overview of all profiling experiments
- See `experiments/profiling/knob2_topk/exp_moe_topk.py` for similar experiment structure
- See `experiments/motivate/base_experiment.py` for base experiment class

## File Structure

```
experiments/profiling/knob3_layers/
    ├── exp_transformer_blocks_mask.py  # Main experiment script
    └── exp_layer_skipping.py           # Alternative implementation (removes blocks)

docs/
    └── EXP3_TRANSFORMER_BLOCKS_MASK.md  # This document
```

## Troubleshooting

### Common Issues

1. **Cache shape mismatch**: If you encounter cache shape errors, check that dummy caches match the expected shape for your model's attention mechanism.

2. **Mask not applied**: Ensure `mask_wrapper.apply()` is called before measurements and `mask_wrapper.remove()` is called after.

3. **Unexpected latency**: If latency doesn't scale as expected, check:
   - GPU utilization
   - Memory bandwidth
   - Overhead from mask checking

### Debug Tips

- Enable detailed logging: `logging.basicConfig(level=logging.DEBUG)`
- Check mask application: Log `block_mask.tolist()` to verify mask is correct
- Verify block skipping: Add print statements in masked forward to confirm blocks are skipped

