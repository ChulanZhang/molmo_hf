# Control Knobs Documentation

This directory contains comprehensive documentation for the three main control knobs used for adaptive inference in Molmo models.

## Overview

The three control knobs allow fine-grained control over model computation:

1. **Vision Tokens Knob** - Controls input size (number of vision tokens)
2. **MoE Top-K Knob** - Controls model width (number of active experts)
3. **Transformer Blocks Knob** - Controls model depth (number of active blocks)

## Documents

### 1. Vision Tokens Knob

**Main file**: `vision_tokens_knob.md`  
**Examples file**: `vision_tokens_knob_examples.md`

Controls the number of vision tokens by managing:
- Number of image crops
- Tiling configuration (rows × cols) - **adaptively selected based on image aspect ratio**
- Image resolution

**Key formula**: `Total Vision Tokens = (num_crops + 1) × 144`

**Key innovation**: Uses `vision_tokens_list` instead of `image_size_list` for adaptive tiling selection, minimizing image distortion.

**Use cases**:
- Reduce prefill latency by limiting vision tokens
- Balance quality vs. speed for different image sizes
- Adaptive image resizing based on latency budget
- Preserve aspect ratio while controlling vision token count

**See also**:
- `vision_tokens_knob_examples.md`: Real-world examples with actual images from VQA v2 dataset
- Vision tokens approach vs image size approach: See "Key innovation" in `vision_tokens_knob.md`

### 2. MoE Top-K Knob

**File**: `moe_topk_knob.md`

Controls the number of active experts per token in MoE layers:
- Dynamic top-K adjustment at runtime
- Linear scaling of computation with top-K
- Quality vs. speed tradeoff

**Key formula**: `Active computation = (top_k / moe_num_experts) × Full model computation`

**Use cases**:
- Adjust model width based on latency budget
- Balance quality vs. speed for different tasks
- Per-request expert selection

### 3. Transformer Blocks Knob

**File**: `transformer_blocks_knob.md`

Controls model depth by selectively activating blocks:
- Importance score-based block selection
- Mask-based block skipping
- Linear scaling of latency with depth

**Key formula**: `Relative Latency ≈ (active_blocks / total_blocks) × Full model latency`

**Use cases**:
- Reduce model depth for faster inference
- Importance-based block selection
- Adaptive depth based on task complexity

## Quick Reference

### Vision Tokens → Image Resolution

```python
target_tokens = 432  # 2 crops + global
num_crops = (target_tokens // 144) - 1  # = 2
tiling = crops_to_tiling(num_crops, aspect_ratio)
resolution = tiling_to_resolution(tiling)  # e.g., (336, 560)
```

### MoE Top-K Setting

```python
set_moe_top_k(model, k=4)  # Activate top-4 experts
```

### Transformer Blocks Selection

```python
importance_scores = compute_accuracy_drop_importance(model, dataloader, baseline_acc)
block_mask = select_top_k_blocks(importance_scores, k=12)
mask_wrapper = BlockMaskWrapper(model, block_mask)
mask_wrapper.apply()
```

## Combined Usage

All three knobs can be used together for maximum flexibility:

```python
# 1. Set vision tokens (reduce input size)
target_tokens = 432
resized_image = resize_for_target_vision_tokens(image, target_tokens)

# 2. Set MoE top-K (reduce model width)
set_moe_top_k(model, k=2)

# 3. Set transformer blocks (reduce model depth)
importance_scores = compute_importance(model, batch)
block_mask = select_top_k_blocks(importance_scores, k=12)
mask_wrapper = BlockMaskWrapper(model, block_mask)
mask_wrapper.apply()

# Run inference
outputs = model.generate(input_ids, images=resized_image, ...)
```

## Related Documents

- `../mechanisms/model_inference_flow.md`: Complete inference pipeline
- `../experiments/profiling_experiments.md`: Profiling experiments using these knobs
- `../experiments/motivation_experiments.md`: Motivation experiments

