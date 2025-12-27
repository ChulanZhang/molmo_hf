# Transformer Blocks Control Knob

## Overview

The transformer blocks knob controls model depth by selectively activating or masking transformer blocks during inference. This knob allows dynamic adjustment of computational cost by skipping less important blocks based on importance scores.

## Key Concepts

### Block Masking vs. Early Exit

Unlike early exit mechanisms that stop computation at a fixed point, block masking:
- **Keeps all blocks** in the model structure
- **Skips computation** for masked blocks (pass-through)
- **Maintains cache structure** for proper generation
- **Allows flexible selection** of which blocks to activate

### Importance Score-Based Selection

Blocks can be selected based on importance scores computed during a forward pass:

1. **Compute importance scores** for each block (e.g., based on activation magnitude, gradient, or attention patterns)
2. **Rank blocks** by importance
3. **Select top-N blocks** or blocks above a threshold
4. **Create mask** to activate only selected blocks

## Implementation

### BlockMaskWrapper Class

The `BlockMaskWrapper` class provides the core functionality:

```python
class BlockMaskWrapper:
    """
    Wrapper to apply block masks during forward pass.
    Allows skipping certain transformer blocks without removing them from the model.
    """
    
    def __init__(self, model: MolmoModel, block_mask: torch.Tensor):
        """
        Args:
            model: The MolmoModel instance
            block_mask: Boolean tensor of shape (n_layers,) where True means active
        """
        self.model = model
        self.block_mask = block_mask
        self.original_forward = None
        self.n_layers = len(model.transformer.blocks)
        
    def apply(self):
        """Apply mask by replacing model.forward with masked version."""
        self.original_forward = self.model.forward
        self.model.forward = self._masked_forward
        
    def remove(self):
        """Remove mask and restore original forward."""
        if self.original_forward is not None:
            self.model.forward = self.original_forward
```

### Masked Forward Pass

The masked forward pass:
1. Replicates original `MolmoModel.forward` logic up to blocks loop
2. Checks mask before each block execution
3. Skips computation for masked blocks (identity pass-through)
4. Handles cache properly for both active and skipped blocks

```python
def _masked_forward(self, *args, **kwargs):
    """Modified forward pass that skips masked blocks."""
    # ... setup code ...
    
    # Apply blocks with mask
    for block_idx, layer in enumerate(self.model.transformer.blocks):
        if not self.block_mask[block_idx]:
            # Skip: pass through x without computation
            if use_cache:
                # Handle cache for skipped blocks
                cache = self._get_dummy_cache(...)
            else:
                cache = None
        else:
            # Normal execution
            x, cache = layer(x, ...)
        
        if attn_key_values is not None:
            attn_key_values.append(cache)
    
    # ... rest of forward pass ...
```

## Importance Score Calculation

### Method 1: Activation Magnitude

Compute importance based on activation magnitudes:

```python
def compute_activation_importance(model, batch, num_samples=100):
    """
    Compute importance scores based on activation magnitudes.
    
    Args:
        model: MolmoModel instance
        batch: Input batch
        num_samples: Number of samples to use for averaging
    
    Returns:
        importance_scores: Tensor of shape (n_layers,) with importance scores
    """
    model.eval()
    importance_scores = torch.zeros(len(model.transformer.blocks))
    
    # Register hooks to capture activations
    activations = {}
    
    def hook_fn(block_idx):
        def hook(module, input, output):
            if block_idx not in activations:
                activations[block_idx] = []
            # Store activation magnitude (L2 norm)
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            activations[block_idx].append(x.norm().item())
        return hook
    
    hooks = []
    for idx, block in enumerate(model.transformer.blocks):
        hook = block.register_forward_hook(hook_fn(idx))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        for _ in range(num_samples):
            _ = model(**batch)
    
    # Compute average importance
    for block_idx, acts in activations.items():
        importance_scores[block_idx] = np.mean(acts)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return importance_scores
```

### Method 2: Gradient-Based Importance

Compute importance based on gradients:

```python
def compute_gradient_importance(model, batch, loss_fn):
    """
    Compute importance scores based on gradients.
    
    Args:
        model: MolmoModel instance
        batch: Input batch with labels
        loss_fn: Loss function
    
    Returns:
        importance_scores: Tensor of shape (n_layers,) with importance scores
    """
    model.train()
    importance_scores = torch.zeros(len(model.transformer.blocks))
    
    # Forward and backward pass
    outputs = model(**batch)
    loss = loss_fn(outputs, batch['labels'])
    loss.backward()
    
    # Compute importance from gradients
    for idx, block in enumerate(model.transformer.blocks):
        # Sum of absolute gradients for block parameters
        grad_norm = 0.0
        for param in block.parameters():
            if param.grad is not None:
                grad_norm += param.grad.abs().sum().item()
        importance_scores[idx] = grad_norm
    
    model.zero_grad()
    return importance_scores
```

### Method 3: Accuracy Drop (Recommended)

Compute importance based on accuracy drop when block is removed:

```python
def compute_accuracy_drop_importance(
    model,
    dataloader,
    baseline_accuracy: float,
    num_samples: int = 5000
) -> torch.Tensor:
    """
    Compute importance scores based on accuracy drop when each block is removed.
    
    This is the most reliable method as it directly measures impact on task performance.
    
    Args:
        model: MolmoModel instance
        dataloader: DataLoader for evaluation
        baseline_accuracy: Baseline accuracy with all blocks
        num_samples: Number of samples to evaluate
    
    Returns:
        importance_scores: Tensor of shape (n_layers,) with importance scores
    """
    model.eval()
    importance_scores = torch.zeros(len(model.transformer.blocks))
    
    for layer_idx in range(len(model.transformer.blocks)):
        # Create mask: all blocks active except this one
        block_mask = torch.ones(len(model.transformer.blocks), dtype=torch.bool)
        block_mask[layer_idx] = False  # Remove this block
        
        # Ensure first and last blocks are always active
        block_mask[0] = True
        block_mask[-1] = True
        
        # Apply mask
        mask_wrapper = BlockMaskWrapper(model, block_mask)
        mask_wrapper.apply()
        
        # Compute accuracy without this block
        ablated_accuracy = compute_accuracy(model, dataloader, num_samples)
        
        # Importance = accuracy drop (higher drop = more important)
        importance_score = baseline_accuracy - ablated_accuracy
        importance_scores[layer_idx] = importance_score
        
        # Remove mask
        mask_wrapper.remove()
    
    return importance_scores
```

**Advantages**:
- **Direct measurement**: Measures actual impact on task performance
- **Task-specific**: Importance scores reflect importance for specific task
- **Reliable**: More accurate than proxy metrics

**Disadvantages**:
- **Computationally expensive**: Requires full evaluation for each block
- **Task-dependent**: Scores may vary across different tasks

### Method 4: Attention Pattern Importance

Compute importance based on attention patterns:

```python
def compute_attention_importance(model, batch):
    """
    Compute importance scores based on attention patterns.
    
    Args:
        model: MolmoModel instance
        batch: Input batch
    
    Returns:
        importance_scores: Tensor of shape (n_layers,) with importance scores
    """
    model.eval()
    importance_scores = torch.zeros(len(model.transformer.blocks))
    
    # Register hooks to capture attention weights
    attention_weights = {}
    
    def hook_fn(block_idx):
        def hook(module, input, output):
            if block_idx not in attention_weights:
                attention_weights[block_idx] = []
            # Extract attention weights from attention module
            # (implementation depends on attention mechanism)
            if hasattr(module, 'attn') and hasattr(module.attn, 'attention_weights'):
                attn_weights = module.attn.attention_weights
                # Compute entropy or variance as importance
                entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1).mean()
                attention_weights[block_idx].append(entropy.item())
        return hook
    
    hooks = []
    for idx, block in enumerate(model.transformer.blocks):
        hook = block.register_forward_hook(hook_fn(idx))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(**batch)
    
    # Compute average importance
    for block_idx, attns in attention_weights.items():
        importance_scores[block_idx] = np.mean(attns)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return importance_scores
```

## Block Selection Strategies

### Strategy 1: Top-K by Importance

Select top-K most important blocks:

```python
def select_top_k_blocks(importance_scores: torch.Tensor, k: int) -> torch.Tensor:
    """
    Select top-K blocks by importance.
    
    Args:
        importance_scores: Importance scores for each block
        k: Number of blocks to select
    
    Returns:
        block_mask: Boolean tensor, True for selected blocks
    """
    _, top_indices = torch.topk(importance_scores, k)
    block_mask = torch.zeros_like(importance_scores, dtype=torch.bool)
    block_mask[top_indices] = True
    return block_mask
```

### Strategy 2: Threshold-Based Selection

Select blocks above importance threshold:

```python
def select_by_threshold(importance_scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Select blocks with importance above threshold.
    
    Args:
        importance_scores: Importance scores for each block
        threshold: Importance threshold (relative to max)
    
    Returns:
        block_mask: Boolean tensor, True for selected blocks
    """
    max_importance = importance_scores.max()
    threshold_value = max_importance * threshold
    block_mask = importance_scores >= threshold_value
    return block_mask
```

### Strategy 3: Sequential Selection (Early to Late)

Select first N blocks (simple strategy, no importance needed):

```python
def select_sequential_blocks(n_layers: int, num_active: int) -> torch.Tensor:
    """
    Select first N blocks sequentially.
    
    Args:
        n_layers: Total number of layers
        num_active: Number of active blocks
    
    Returns:
        block_mask: Boolean tensor, True for first num_active blocks
    """
    block_mask = torch.zeros(n_layers, dtype=torch.bool)
    block_mask[:num_active] = True
    return block_mask
```

## Complete Workflow

### Step 1: Compute Importance Scores

```python
# Compute importance scores
importance_scores = compute_activation_importance(model, batch, num_samples=100)
```

### Step 2: Select Blocks

```python
# Select top-K blocks
num_active_blocks = 12
block_mask = select_top_k_blocks(importance_scores, num_active_blocks)
```

### Step 3: Apply Mask

```python
# Create and apply mask wrapper
mask_wrapper = BlockMaskWrapper(model, block_mask)
mask_wrapper.apply()

# Run inference with masked blocks
outputs = model.generate(input_ids, images=images, ...)

# Remove mask when done
mask_wrapper.remove()
```

## Performance Impact

### Latency Scaling

Block masking affects latency approximately linearly:

| Active Blocks | Relative Latency | Notes |
|---------------|------------------|-------|
| 50% (12/24) | ~50% | Half depth |
| 75% (18/24) | ~75% | Three-quarters depth |
| 100% (24/24) | 100% | Full depth |

**Note**: Actual scaling may vary due to:
- Cache overhead for skipped blocks
- GPU utilization patterns
- Memory bandwidth constraints

### Quality Impact

- **Early blocks** (0-8): General feature extraction, critical for all tasks
- **Middle blocks** (8-16): Task-specific processing, important for complex tasks
- **Late blocks** (16-24): Fine-tuning and refinement, may be skipped for simple tasks

## Usage Examples

### Example 1: Importance-Based Block Selection

```python
from experiments.profiling.knob3_layers.exp_transformer_blocks_mask import BlockMaskWrapper

# Compute importance scores
importance_scores = compute_activation_importance(model, batch)

# Select top-12 blocks
block_mask = select_top_k_blocks(importance_scores, k=12)

# Apply mask
mask_wrapper = BlockMaskWrapper(model, block_mask)
mask_wrapper.apply()

# Run inference
outputs = model.generate(input_ids, images=images)

# Clean up
mask_wrapper.remove()
```

### Example 2: Adaptive Block Selection Based on Latency Budget

```python
def adaptive_block_selection(
    model,
    batch,
    latency_budget_ms: float,
    baseline_latency_ms: float
) -> BlockMaskWrapper:
    """
    Select blocks based on latency budget.
    
    Args:
        model: MolmoModel instance
        batch: Input batch
        latency_budget_ms: Target latency
        baseline_latency_ms: Full model latency
    
    Returns:
        BlockMaskWrapper instance
    """
    # Compute importance
    importance_scores = compute_activation_importance(model, batch)
    
    # Estimate required depth
    depth_ratio = latency_budget_ms / baseline_latency_ms
    num_active = int(len(model.transformer.blocks) * depth_ratio)
    num_active = max(1, min(num_active, len(model.transformer.blocks)))
    
    # Select blocks
    block_mask = select_top_k_blocks(importance_scores, num_active)
    
    # Create wrapper
    return BlockMaskWrapper(model, block_mask)
```

### Example 3: Per-Request Block Selection

```python
def select_blocks_for_request(
    model,
    batch,
    request_complexity: str,
    latency_budget_ms: float
) -> BlockMaskWrapper:
    """
    Select blocks based on request characteristics.
    
    Args:
        model: MolmoModel instance
        batch: Input batch
        request_complexity: "simple", "medium", "complex"
        latency_budget_ms: Available latency budget
    
    Returns:
        BlockMaskWrapper instance
    """
    # Compute importance
    importance_scores = compute_activation_importance(model, batch)
    
    # Base selection by complexity
    complexity_map = {
        "simple": 0.5,   # 50% of blocks
        "medium": 0.75,  # 75% of blocks
        "complex": 1.0   # All blocks
    }
    
    base_ratio = complexity_map.get(request_complexity, 0.75)
    num_active = int(len(model.transformer.blocks) * base_ratio)
    
    # Adjust based on latency budget
    if latency_budget_ms < 50:
        num_active = max(1, num_active - 2)
    elif latency_budget_ms > 200:
        num_active = min(len(model.transformer.blocks), num_active + 2)
    
    # Select blocks
    block_mask = select_top_k_blocks(importance_scores, num_active)
    
    return BlockMaskWrapper(model, block_mask)
```

## Cache Handling

### Skipped Block Cache

For skipped blocks when `use_cache=True`:
- If `past_key_values` exists for the block, reuse it
- Otherwise, create dummy cache with appropriate shape
- Ensures cache structure remains consistent

**Note**: Dummy cache for skipped blocks is not used in computation but maintains expected data structure.

## Code References

- **BlockMaskWrapper**: `experiments/profiling/knob3_layers/exp_transformer_blocks_mask.py`
  - Lines 27-236: Complete implementation

- **Model forward**: `molmo/models/modeling_molmoe.py`
  - Lines 1996-2007: Blocks iteration loop

- **Experiments**:
  - `experiments/profiling/knob3_layers/exp_transformer_blocks_mask.py`: Latency profiling
  - `experiments/profiling/knob3_layers/exp3_accuracy.py`: Accuracy measurement

## Related Documents

- `../mechanisms/model_inference_flow.md`: Complete inference pipeline
- `vision_tokens_knob.md`: Vision tokens control knob
- `moe_topk_knob.md`: MoE top-K control knob

