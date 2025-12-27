# MoE Top-K Control Knob

## Overview

The MoE (Mixture of Experts) top-K knob controls the number of active experts per token in the MoE layers, which directly affects model width and computation cost. This knob allows dynamic adjustment of the model's computational budget by selecting how many experts to activate for each token.

## Key Concepts

### MoE Architecture

The model uses a sparse MoE architecture where:
- **Total experts**: `moe_num_experts` (default: 64 in MolmoE models)
- **Active experts per token**: `moe_top_k` (configurable, default: 2)
- **Expert selection**: Based on router logits (learned during training)

### Top-K Selection

For each token, the router computes logits for all experts, then:
1. Selects the top-K experts with highest logits
2. Routes the token to these K experts
3. Aggregates outputs from selected experts

**Key formula**:
```
Active computation = (top_k / moe_num_experts) × Full model computation
```

## Configuration

### Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `moe_num_experts` | 64 | Total number of experts |
| `moe_top_k` | 2 | Number of experts to activate per token |
| `moe_mlp_impl` | "sparse" | MoE implementation type |

### Valid Range

- **Minimum**: `top_k = 1` (most efficient, lowest quality)
- **Maximum**: `top_k = moe_num_experts` (full model, highest quality)
- **Typical range**: 1, 2, 4, 8, 16, 32 (for 64-expert model)

## Dynamic Top-K Adjustment

### Setting Top-K at Runtime

The top-K value can be changed dynamically without model reconstruction:

```python
def set_moe_top_k(model, k: int):
    """
    Set top_k for all MoE blocks in the model.
    
    Args:
        model: MolmoModel instance
        k: Top-K value (must be between 1 and moe_num_experts)
    """
    assert 1 <= k <= model.config.moe_num_experts, \
        f"top_k must be between 1 and {model.config.moe_num_experts}"
    
    # Update config
    model.config.moe_top_k = k
    
    # Update each MoE block
    transformer = model.model.transformer
    if isinstance(transformer, torch.nn.ModuleDict):
        blocks = transformer["blocks"] if "blocks" in transformer else []
    elif hasattr(transformer, 'blocks'):
        blocks = transformer.blocks
    else:
        blocks = []
    
    moe_blocks_found = 0
    for block in blocks:
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'top_k'):
            mlp_type = type(block.mlp)
            mlp_type_name = mlp_type.__name__ if hasattr(mlp_type, '__name__') else str(mlp_type)
            
            is_moe_block = (
                isinstance(block.mlp, MolmoeSparseMoeBlock) or 
                'MolmoeSparseMoeBlock' in mlp_type_name or
                'SparseMoe' in mlp_type_name
            )
            
            if is_moe_block:
                block.mlp.top_k = k
                moe_blocks_found += 1
    
    return moe_blocks_found
```

### Implementation Details

**Code location**: `experiments/profiling/knob2_topk/exp2_accuracy.py`, `experiments/profiling/knob5_combined/exp5_accuracy.py`

**Key points**:
1. Update `model.config.moe_top_k` for consistency
2. Iterate through all transformer blocks
3. Identify MoE blocks (check for `MolmoeSparseMoeBlock`)
4. Set `block.mlp.top_k = k` directly

## Performance Impact

### Computation Scaling

Top-K affects computation in MoE layers:

| Top-K | Relative Computation | Notes |
|-------|---------------------|-------|
| 1 | ~1.6% | Minimal (1/64 of experts) |
| 2 | ~3.1% | Default (2/64 of experts) |
| 4 | ~6.2% | Moderate |
| 8 | ~12.5% | High |
| 16 | ~25% | Very high |
| 32 | ~50% | Half of experts |

**Note**: Actual computation may vary due to:
- Load balancing across experts
- Token routing patterns
- Hardware utilization

### Latency Impact

Top-K primarily affects:
- **Prefill latency**: Linear scaling with top-K (approximately)
- **Decode latency**: Linear scaling with top-K per token (approximately)

**Typical scaling**:
- `top_k=1`: ~50% of `top_k=2` latency
- `top_k=4`: ~200% of `top_k=2` latency
- `top_k=8`: ~400% of `top_k=2` latency

### Quality Impact

- **Lower top-K** (1-2): Faster, but may reduce quality for complex tasks
- **Higher top-K** (4-8): Better quality, but slower
- **Very high top-K** (16+): Diminishing returns, significant latency increase

## Usage Examples

### Example 1: Basic Top-K Setting

```python
from molmo.models.modeling_molmoe import MolmoForCausalLM

# Load model
model = MolmoForCausalLM.from_pretrained("checkpoints")

# Set top_k to 4
set_moe_top_k(model, k=4)

# Run inference
outputs = model.generate(input_ids, images=images, ...)
```

### Example 2: Adaptive Top-K Based on Latency Budget

```python
def adaptive_top_k(latency_budget_ms: float, base_latency_ms: float) -> int:
    """
    Select top_k based on latency budget.
    
    Args:
        latency_budget_ms: Target latency in milliseconds
        base_latency_ms: Baseline latency with top_k=2
    
    Returns:
        Recommended top_k value
    """
    if latency_budget_ms < base_latency_ms * 0.6:
        return 1  # Fast mode
    elif latency_budget_ms < base_latency_ms * 1.2:
        return 2  # Default mode
    elif latency_budget_ms < base_latency_ms * 2.0:
        return 4  # Quality mode
    else:
        return 8  # High quality mode

# Usage
target_latency = 100.0  # ms
baseline_latency = 80.0  # ms
top_k = adaptive_top_k(target_latency, baseline_latency)
set_moe_top_k(model, top_k)
```

### Example 3: Per-Request Top-K Selection

```python
def select_top_k_for_request(
    request_complexity: str,
    latency_budget_ms: float
) -> int:
    """
    Select top_k based on request characteristics.
    
    Args:
        request_complexity: "simple", "medium", "complex"
        latency_budget_ms: Available latency budget
    
    Returns:
        Top_k value
    """
    complexity_map = {
        "simple": 1,
        "medium": 2,
        "complex": 4
    }
    
    base_top_k = complexity_map.get(request_complexity, 2)
    
    # Adjust based on latency budget
    if latency_budget_ms < 50:
        return max(1, base_top_k - 1)
    elif latency_budget_ms > 200:
        return min(8, base_top_k + 2)
    else:
        return base_top_k
```

## Experimental Profiling

### Profiling Script

Use `experiments/profiling/knob2_topk/exp_moe_topk.py` to profile different top-K values:

```bash
python experiments/profiling/knob2_topk/exp_moe_topk.py \
    --model_path checkpoints \
    --output_dir ./results/moe_topk \
    --num_samples 50 \
    --top_k_values 1 2 4 8 16 32
```

### Accuracy Measurement

Use `experiments/profiling/knob2_topk/exp2_accuracy.py` to measure accuracy vs. top-K:

```bash
python experiments/profiling/knob2_topk/exp2_accuracy.py \
    --model_path checkpoints \
    --batch_size 64 \
    --top_k_list 1 2 4 8 \
    --auto_adjust_batch_size
```

## Implementation Notes

### Current Implementation

The current MoE implementation:
- Uses standard sparse MoE (not MegaBlocks)
- Iterates through all experts (even unselected ones)
- Uses `index_add_` to aggregate expert outputs
- Supports dynamic top-K adjustment at runtime

### Limitations

1. **Not optimized for production**: Current implementation may have overhead
2. **Load balancing**: May not be perfectly balanced across experts
3. **Memory**: All expert parameters remain in memory

### Future Optimizations

Potential improvements:
- **MegaBlocks integration**: Block-sparse operations for better efficiency
- **Dynamic expert selection**: Only compute selected experts
- **Better load balancing**: More uniform expert utilization

**Note**: MegaBlocks may not support dynamic top-K changes (requires fixed sparsity pattern at compile time). Current implementation is better for runtime flexibility.

## Code References

- **MoE block**: `molmo/models/modeling_molmoe.py`
  - `MolmoeSparseMoeBlock`: Lines 813-863
  - Router and expert selection logic

- **Configuration**: `molmo/config.py`
  - `moe_num_experts`: Line 716
  - `moe_top_k`: Line 721

- **Experiments**:
  - `experiments/profiling/knob2_topk/exp_moe_topk.py`: Latency profiling
  - `experiments/profiling/knob2_topk/exp2_accuracy.py`: Accuracy measurement

## Related Documents

- `../mechanisms/model_inference_flow.md`: Complete inference pipeline including MoE
- `vision_tokens_knob.md`: Vision tokens control knob
- `transformer_blocks_knob.md`: Transformer blocks control knob

