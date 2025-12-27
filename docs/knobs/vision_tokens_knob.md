# Vision Tokens Control Knob

## Overview

The vision tokens knob controls the number of vision tokens processed by the model, which directly affects prefill latency. This knob works by controlling the number of image crops, which in turn determines the total vision token count.

## Key Formula

```
Total Vision Tokens = (num_crops + 1) × 144
```

Where:
- `num_crops`: Number of image crops (determined by tiling configuration)
- `+1`: Global image (always processed)
- `144`: Post-pooling tokens per crop/global image (12×12 grid)

## Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| `base_image_input_size` | 336×336 px | Fixed size per crop |
| `image_patch_size` | 14 px | Vision patch size (ViT-L/14 style) |
| `vision_grid_per_crop` | 24×24 | Pre-pooling patches per crop |
| `vision_patches_per_crop` | 576 | Pre-pooling patches (24×24) |
| `image_pooling_h/w` | 2×2 | Pooling size |
| `vision_tokens_per_crop` | 144 | Post-pooling tokens per crop (12×12) |
| `global_image_tokens` | 144 | Global image tokens |
| `overlap_margins` | (4, 4) | Overlap margins in patches |
| `max_crops` | 12 | Maximum number of crops (default) |

## Vision Token Calculation Pipeline

### Step 1: Image Resize and Tiling

The original image is resized and divided into multiple crops using the `select_tiling` algorithm:

```python
# Calculate tiling configuration
tiling = select_tiling(
    original_image_h - total_margin_pixels,
    original_image_w - total_margin_pixels,
    crop_window_size,
    max_crops
)

# Resize image to target size
resized_size = [
    tiling[0] * crop_window_size + total_margin_pixels,
    tiling[1] * crop_window_size + total_margin_pixels
]
```

**Key parameter calculations**:
- `crop_window_size = crop_window_patches * patch_size = 16 * 14 = 224`
- `crop_window_patches = crop_patches - (right_margin + left_margin) = 24 - 8 = 16`
- `crop_patches = base_image_input_size[0] // patch_size = 336 // 14 = 24`
- `total_margin_pixels = (right_margin + left_margin) * patch_size = (4 + 4) * 14 = 112`

### Step 2: The `select_tiling` Algorithm

The `select_tiling` function determines the tiling configuration (rows × cols) that minimizes image upscaling:

**Algorithm**:
1. **Generate candidates**: All possible tiling configurations (i×j) where i×j ≤ max_crops
2. **Calculate scaling**: For each candidate, calculate required scaling to fit the image
3. **Select optimal**: Choose the tiling requiring the **least upscaling** (or least downscaling if all require downscaling)

**Key properties**:
- **Aspect ratio preservation**: Selects tiling closest to original image aspect ratio
- **Minimize upscaling**: Avoids excessive image enlargement (may cause quality loss)
- **Considers all configurations**: Includes 1×N, N×1, 2×N, N×2, etc.

**Exact crops mode** (recommended for precise control):
- If `exact_num_crops` is specified, `select_tiling` will **force** selection of exactly that many crops
- Among all tilings that result in exactly `exact_num_crops` crops, it selects the one closest to the image aspect ratio
- This ensures the **number of crops** matches the target exactly
- **Note**: Even with `exact_num_crops`, actual vision tokens may be slightly less than `(num_crops + 1) × 144` due to invalid patches (see "Why actual vision tokens may not be exactly 144 × (num_crops + 1)" section below)
- Used in `combined_profiling.py` to ensure precise crop count control

**Example**:
- Image size: 640×425 (width > height, landscape)
- Effective size: 528×313 (after subtracting margins)
- Candidate tilings: 1×1, 1×2, 2×1, 1×3, 3×1, 2×2, ...
- For 1×2 tiling: requires 224×448, scale = min(224/313, 448/528) ≈ 0.72 (downscale)
- For 2×1 tiling: requires 448×224, scale = min(448/313, 224/528) ≈ 0.42 (downscale)
- **Selected: 1×2** (less downscaling: 0.72 > 0.42)

### Step 3: Per-Crop Processing

Each crop is processed at fixed size `base_image_input_size = (336, 336)`:

```python
# Patch grid per crop
vision_grid_h = base_image_input_size[0] // image_patch_size  # 336 // 14 = 24
vision_grid_w = base_image_input_size[1] // image_patch_size  # 336 // 14 = 24
vision_patches_per_crop = vision_grid_h * vision_grid_w  # 24 * 24 = 576
```

### Step 4: Vision Encoder and Pooling

1. **Vision Encoder (ViT)**: Processes 576 patches per crop → 576 features per crop
2. **2D Pooling**: Reduces 24×24 patches to 12×12 tokens via 2×2 pooling
   ```python
   llm_patches_per_crop_h = (24 + 2 - 1) // 2 = 12
   llm_patches_per_crop_w = (24 + 2 - 1) // 2 = 12
   vision_tokens_per_crop = 12 * 12 = 144
   ```
3. **Global Image**: Always processed, produces 144 tokens (same as each crop)

### Step 5: Total Vision Tokens

```python
total_vision_tokens = (num_crops + 1) * vision_tokens_per_crop
                     = (num_crops + 1) * 144
```

**Actual counting in code**:
```python
# Use image_input_idx to count valid vision tokens
num_vision_tokens = (batch["image_input_idx"] >= 0).sum().item()
```

Note: `image_input_idx` already reflects post-pooling token count (144), not pre-pooling patch count (576).

**Why actual vision tokens may not be exactly 144 × (num_crops + 1)**:
- Invalid patches are marked as `-100` in `image_input_idx`
- These patches are excluded from the count: `(image_input_idx >= 0).sum()`
- Common causes of invalid patches:
  1. **Image boundary handling**: Patches that extend beyond image boundaries
  2. **Padding regions**: Patches in padding areas (especially for non-square images)
  3. **Tiling edge cases**: Some crops may have partial patches at edges

**Practical impact**: For most images, the deviation is small (typically 0-24 tokens, or 0-5% of theoretical value). The deviation is consistent across similar images, so it doesn't significantly affect experimental comparisons.

## Controlling Vision Tokens

### Method 1: Target Vision Tokens → Calculate Required Crops (Recommended)

Given a target number of vision tokens, calculate the required number of crops and use `exact_num_crops` to force precise crop selection:

```python
def tokens_to_crops(target_tokens: int) -> int:
    """
    Calculate number of crops needed for target vision tokens.
    
    Formula: target_tokens = (num_crops + 1) * 144
    Solve: num_crops = (target_tokens / 144) - 1
    """
    num_crops = (target_tokens // 144) - 1
    return max(1, num_crops)  # At least 1 crop

# Usage in MultiModalPreprocessor
num_crops = tokens_to_crops(target_vision_tokens)
mm_preprocessor = MultiModalPreprocessor(
    tokenizer=tokenizer,
    crop_mode="resize",
    max_crops=num_crops,  # Set max_crops to num_crops
    exact_num_crops=num_crops,  # Force exact number of crops
    # ... other parameters
)
```

**Example**:
- Target: 432 vision tokens
- Required crops: (432 // 144) - 1 = 2 crops
- Set `max_crops=2` and `exact_num_crops=2`
- Theoretical tokens: (2 + 1) × 144 = 432 ✓

**Important Note**: Even with `exact_num_crops`, actual vision tokens may be **slightly less** than the theoretical value because:
- Some patches may be marked as invalid (-100) if they exceed image boundaries
- Padding may cause some patches to be invalid
- Tiling configuration may result in partial crops

**Typical deviation**: For most images, actual vision tokens are within 0-5% of the theoretical value. The deviation is usually small and consistent across similar images.

**Best Practice for Precise Control**:
1. Use `exact_num_crops` to ensure exact crop count: `exact_num_crops = (target_tokens // 144) - 1`
2. Set `max_crops` to the same value: `max_crops = exact_num_crops`
3. Accept that actual vision tokens may be slightly less than theoretical (typically 0-24 tokens less)
4. For experimental comparisons, use `actual_vision_tokens` (from `image_input_idx`) rather than theoretical value
5. The deviation is consistent across similar images, so it doesn't significantly affect experimental comparisons

### Method 2: Calculate Tiling Configuration

Given number of crops, find appropriate tiling configuration:

```python
def crops_to_tiling(num_crops: int, aspect_ratio: float = 1.0) -> Tuple[int, int]:
    """
    Find tiling configuration for given number of crops.
    
    Args:
        num_crops: Target number of crops
        aspect_ratio: Image aspect ratio (width/height)
    
    Returns:
        (rows, cols) tiling configuration
    """
    best_tiling = None
    best_match = float('inf')
    
    for i in range(1, num_crops + 1):
        if num_crops % i == 0:
            j = num_crops // i
            tiling = (i, j)
            
            # Calculate tiling aspect ratio
            resized_h = i * 224 + 112
            resized_w = j * 224 + 112
            tiling_aspect = resized_w / resized_h
            
            # Select closest to target aspect ratio
            if abs(tiling_aspect - aspect_ratio) < best_match:
                best_match = abs(tiling_aspect - aspect_ratio)
                best_tiling = tiling
    
    return best_tiling if best_tiling else (1, num_crops)
```

### Method 3: Calculate Target Image Resolution

Given tiling configuration, calculate required image resolution:

```python
def tiling_to_resolution(tiling: Tuple[int, int], 
                        crop_window_size: int = 224,
                        total_margin_pixels: int = 112) -> Tuple[int, int]:
    """
    Calculate image resolution for given tiling.
    
    Args:
        tiling: (rows, cols) tiling configuration
        crop_window_size: Size of each crop window (default: 224)
        total_margin_pixels: Total margin pixels (default: 112)
    
    Returns:
        (target_h, target_w) resolution
    """
    rows, cols = tiling
    target_h = rows * crop_window_size + total_margin_pixels
    target_w = cols * crop_window_size + total_margin_pixels
    return target_h, target_w
```

### Complete Workflow: Vision Tokens → Image Resolution

```python
def vision_tokens_to_image_resolution(
    target_tokens: int,
    original_aspect_ratio: float,
    crop_window_size: int = 224,
    total_margin_pixels: int = 112
) -> Tuple[int, int]:
    """
    Complete workflow: target vision tokens → required image resolution.
    
    Args:
        target_tokens: Target number of vision tokens
        original_aspect_ratio: Original image aspect ratio (width/height)
        crop_window_size: Size of each crop window
        total_margin_pixels: Total margin pixels
    
    Returns:
        (target_h, target_w) resolution to resize image
    """
    # Step 1: Calculate required crops
    num_crops = tokens_to_crops(target_tokens)
    
    # Step 2: Find best tiling for aspect ratio
    tiling = crops_to_tiling(num_crops, original_aspect_ratio)
    
    # Step 3: Calculate target resolution
    target_h, target_w = tiling_to_resolution(tiling, crop_window_size, total_margin_pixels)
    
    return target_h, target_w
```

## Resolution to Vision Tokens Mapping

### Common Mappings

| Image Resolution (H×W) | Tiling | Crops | Vision Tokens |
|------------------------|--------|-------|---------------|
| 336×336 | 1×1 | 1 | 288 |
| 336×560 | 1×2 | 2 | 432 |
| 560×336 | 2×1 | 2 | 432 |
| 336×784 | 1×3 | 3 | 576 |
| 560×560 | 2×2 | 4 | 720 |
| 784×784 | 3×3 | 9 | 1440 |
| 1008×784 | 4×3 | 12 | 1872 |
| 784×1008 | 3×4 | 12 | 1872 |

### Aspect Ratio Considerations

**Recommended aspect ratio range**: 0.5 to 2.0 (1:2 to 2:1)

- **Square images** (aspect ratio ≈ 1.0): Use N×N tilings (e.g., 2×2, 3×3)
- **Wide images** (aspect ratio > 1.0): Use 1×N or 2×N tilings
- **Tall images** (aspect ratio < 1.0): Use N×1 or N×2 tilings

**Avoid extreme aspect ratios** (< 0.33 or > 3.0) as they cause significant image distortion.

## Implementation Example

```python
from PIL import Image

def resize_for_target_vision_tokens(
    image: Image.Image,
    target_tokens: int,
    preserve_aspect_ratio: bool = True
) -> Image.Image:
    """
    Resize image to achieve target number of vision tokens.
    
    Args:
        image: Input PIL Image
        target_tokens: Target number of vision tokens
        preserve_aspect_ratio: Whether to preserve original aspect ratio
    
    Returns:
        Resized image
    """
    orig_w, orig_h = image.size
    orig_aspect = orig_w / orig_h
    
    if preserve_aspect_ratio:
        # Calculate target resolution
        target_h, target_w = vision_tokens_to_image_resolution(
            target_tokens, orig_aspect
        )
    else:
        # Use first available resolution for target tokens
        num_crops = tokens_to_crops(target_tokens)
        tiling = crops_to_tiling(num_crops, 1.0)  # Use square as default
        target_h, target_w = tiling_to_resolution(tiling)
    
    # Resize image
    resized = image.resize((target_w, target_h), Image.BILINEAR)
    return resized
```

## Limits and Constraints

### Sequence Length Limit (Primary Constraint)

The model has `max_sequence_length = 4096`, which limits the maximum vision tokens:

| Max Crops | Vision Tokens (estimated) | Remaining for Text | Feasible |
|-----------|---------------------------|-------------------|----------|
| 12 | ~1,728 | ~2,368 | ✅ Safe |
| 20 | ~2,880 | ~1,216 | ⚠️ Tight |
| 25 | ~3,600 | ~496 | ⚠️ Very tight |
| 28 | ~4,032 | ~64 | ❌ Almost impossible |
| 30+ | >4,096 | <0 | ❌ Exceeds limit |

**Note**: Actual tokens per crop may be slightly more than 144 due to special tokens (e.g., `image_start_token`, `image_end_token`, `image_col_token`), so estimates use 144-150 tokens per crop.

**Recommendation**: 
- **Recommended**: `max_crops ≤ 20` for practical use
- **Feasible but requires caution**: `20 < max_crops ≤ 25`
- **Not recommended**: `max_crops > 25`
- **Maximum theoretical**: 25-28 with very small batch sizes (1-4)

### Memory Constraints

Vision tokens memory usage:
- Each crop's vision features: `(batch_size, num_crops, 144, hidden_dim)`
- Attention computation: `O(sequence_length²)`, where sequence_length includes all vision tokens

**Memory impact by crop count**:
- `max_crops=12`: Vision tokens ~1,728, batch_size=64 is feasible
- `max_crops=20`: Vision tokens ~2,880, batch_size must be reduced to 16-32
- `max_crops=28`: Vision tokens ~4,032, batch_size may need to be 8-16

### Computation Complexity

Attention computation complexity is `O(sequence_length²)`:
- `max_crops=12`: sequence_length ≈ 1,728 + text_tokens → attention complexity ≈ 3M
- `max_crops=20`: sequence_length ≈ 2,880 + text_tokens → attention complexity ≈ 8M
- `max_crops=28`: sequence_length ≈ 4,032 + text_tokens → attention complexity ≈ 16M

Larger `max_crops` leads to:
- Slower inference speed
- Higher memory usage
- May trigger CUDA kernel limits (e.g., "invalid configuration argument" error)

### Code-Level Constraints

The `select_tiling` function has no hard-coded maximum, but:
- Loop complexity is `O(max_num_crops²)`
- For very large values (>100), it becomes slow

**Practical limits**:
- Default experimental range: `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]`
- This is experimental configuration, not a code limit

### Usage Recommendations by Scenario

1. **Standard use** (e.g., VQA v2):
   - `max_crops = 12` (default)
   - Sufficient for most images

2. **High-resolution images**:
   - `max_crops = 15-18`
   - Requires reduced batch_size

3. **Very high-resolution images**:
   - `max_crops = 20-24`
   - Requires significantly reduced batch_size (8-16)
   - Monitor memory usage

4. **Experimental/Research**:
   - Can try `max_crops > 24`, but requires:
     - Very small batch_size (1-4)
     - Monitor sequence length (must not exceed 4096)
     - Accept slower inference speed

### Testing Larger max_crops

To test larger `max_crops` values:

```bash
# Test max_crops=20
python experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --max_crops_list 20 \
    --batch_size 16 \
    --auto_adjust_batch_size

# Test max_crops=25 (requires smaller batch size)
python experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --max_crops_list 25 \
    --batch_size 8 \
    --auto_adjust_batch_size
```

**Precautions**:
1. Start with smaller batch_size (8-16)
2. Enable `--auto_adjust_batch_size` for automatic adjustment
3. Monitor memory usage and sequence length
4. If encountering "invalid configuration argument" error, batch_size or max_crops is too large

### Monitoring Limits

**Method 1: Check sequence length**
```python
input_len = batch["input_ids"].shape[1]
if input_len > 4000:
    log.warning(f"Sequence length {input_len} is very close to max_sequence_length=4096!")
```

**Method 2: Monitor memory**
```python
import torch
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9  # GB
    reserved = torch.cuda.memory_reserved() / 1e9     # GB
    log.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

**Method 3: Test incrementally**
Gradually increase `max_crops` and observe:
- Whether OOM occurs
- Whether sequence length exceeds 4096
- Inference speed changes

### Summary Table

| Constraint Type | Practical Limit | Notes |
|----------------|-----------------|-------|
| **Hard-coded limit** | ❌ None | No hard-coded maximum in code |
| **Sequence length** | ~25-28 | Limited by `max_sequence_length=4096` |
| **Memory** | ~20-24 | Depends on GPU memory and batch_size |
| **Computation complexity** | ~20-24 | Attention complexity O(n²) |
| **Recommended value** | **≤ 20** | Balance between performance and practicality |

**Conclusion**: While there's no hard-coded limit, practical use recommends `max_crops ≤ 20`, maximum 25-28 (requires very small batch_size).

## Code References

- **Image preprocessing**: `molmo/preprocessors/image_preprocessing_molmo.py`
  - `select_tiling()`: Lines 103-128
  - `image_to_patches_and_tokens()`: Lines 170-350

- **Vision encoding**: `molmo/models/modeling_molmoe.py`
  - `encode_image()`: Lines 1594-1627
  - `forward()` (VisionBackbone): Lines 1629-1709

- **Configuration**: `molmo/config.py`
  - `llm_patches_per_crop()`: Lines 898-903
  - `image_num_patch` property: Lines 309-312

## Related Documents

- `../mechanisms/model_inference_flow.md`: Complete inference pipeline
- `moe_topk_knob.md`: MoE top-K control knob
- `transformer_blocks_knob.md`: Transformer blocks control knob

