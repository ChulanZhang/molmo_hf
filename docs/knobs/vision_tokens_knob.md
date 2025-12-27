# Vision Tokens Control Knob

## Overview

The vision tokens knob is the **primary control knob** for managing vision input size in Molmo. It directly controls the number of vision tokens processed by the model, which significantly affects both accuracy and latency (especially prefill latency).

**Key Principle**: Instead of fixing image dimensions (which can cause aspect ratio mismatches), we specify a **target number of vision tokens** and let the system automatically adapt the tiling configuration to each image's aspect ratio.

## Core Formula

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
| `crop_window_size` | 224 px | Effective crop window (16×14 patches) |
| `total_margin_pixels` | 112 px | Total margin (8×14 patches) |

## Vision Token Calculation Pipeline

### Step 1: Target Vision Tokens → Number of Crops

Given a target number of vision tokens, we calculate the required number of crops:

```python
def tokens_to_crops(target_tokens: int) -> int:
    """
    Calculate number of crops needed for target vision tokens.
    
    Formula: target_tokens = (num_crops + 1) * 144
    Solve: num_crops = (target_tokens / 144) - 1
    """
    num_crops = (target_tokens // 144) - 1
    return max(1, num_crops)  # At least 1 crop
```

**Example**:
- Target: 432 vision tokens
- Required crops: (432 // 144) - 1 = 2 crops
- Theoretical tokens: (2 + 1) × 144 = 432 ✓

### Step 2: Adaptive Tiling Selection (Key Innovation)

For each image, the system automatically selects the best tiling configuration based on the image's **original aspect ratio**:

```python
def select_tiling(
    original_h: int,
    original_w: int,
    crop_window_size: int = 224,
    max_num_crops: int,
    exact_num_crops: Optional[int] = None
) -> Tuple[int, int]:
    """
    Select optimal tiling (rows, cols) for given number of crops.
    
    When exact_num_crops is specified:
    1. Find all possible tilings that result in exactly exact_num_crops
    2. Calculate aspect ratio for each tiling
    3. Select the tiling closest to the original image's aspect ratio
    """
    if exact_num_crops is not None:
        # Find all factorizations of exact_num_crops
        possible_tilings = []
        for i in range(1, exact_num_crops + 1):
            if exact_num_crops % i == 0:
                j = exact_num_crops // i
                possible_tilings.append((i, j))
        
        # Calculate aspect ratio for original image
        original_aspect = original_w / original_h if original_h > 0 else 1.0
        
        # Select tiling with closest aspect ratio
        best_tiling = None
        best_match = float('inf')
        for rows, cols in possible_tilings:
            # Tiling aspect ratio = (cols * crop_window_size) / (rows * crop_window_size)
            tiling_aspect = cols / rows
            aspect_diff = abs(tiling_aspect - original_aspect)
            if aspect_diff < best_match:
                best_match = aspect_diff
                best_tiling = (rows, cols)
        
        return best_tiling
```

**Key Innovation**: Instead of fixing the tiling upfront, we let `select_tiling` adapt to each image's aspect ratio, minimizing distortion.

### Step 3: Image Resize to Target Resolution

Once the tiling is selected, we calculate the target resolution:

```python
def tiling_to_resolution(tiling: Tuple[int, int]) -> Tuple[int, int]:
    """
    Calculate target image resolution for given tiling.
    
    Formula:
    target_h = rows * crop_window_size + total_margin_pixels
    target_w = cols * crop_window_size + total_margin_pixels
    """
    rows, cols = tiling
    target_h = rows * 224 + 112
    target_w = cols * 224 + 112
    return target_h, target_w
```

**Example**:
- Tiling: (3, 2) for 6 crops
- Target resolution: (3×224+112, 2×224+112) = (784, 560)

### Step 4: Resize with Aspect Ratio Preservation

The image is resized to the target resolution while preserving aspect ratio:

```python
def resize_and_crop_to_fill(
    image: np.ndarray,
    target_h: int,
    target_w: int
) -> np.ndarray:
    """
    Resize image to fill target canvas, then crop/pad to exact dimensions.
    
    This ensures:
    1. Small images are upscaled to fill the canvas (when resize_to_fill=True)
    2. Aspect ratio is preserved during resize
    3. Final dimensions match target exactly (via crop/pad)
    """
    # Calculate scale to fill target (preserving aspect ratio)
    scale = max(target_h / image_h, target_w / image_w)
    
    # Resize
    resized_h = int(image_h * scale)
    resized_w = int(image_w * scale)
    resized = cv2.resize(image, (resized_w, resized_h))
    
    # Crop or pad to exact target dimensions
    # ... (crop if too large, pad if too small)
    
    return final_image
```

### Step 5: Per-Crop Processing

Each crop is processed at fixed size `base_image_input_size = (336, 336)`:

```python
# Patch grid per crop
vision_grid_h = 336 // 14 = 24
vision_grid_w = 336 // 14 = 24
vision_patches_per_crop = 24 * 24 = 576
```

### Step 6: Vision Encoder and Pooling

1. **Vision Encoder (ViT)**: Processes 576 patches per crop → 576 features per crop
2. **2D Pooling**: Reduces 24×24 patches to 12×12 tokens via 2×2 pooling
   ```python
   llm_patches_per_crop_h = (24 + 2 - 1) // 2 = 12
   llm_patches_per_crop_w = (24 + 2 - 1) // 2 = 12
   vision_tokens_per_crop = 12 * 12 = 144
   ```
3. **Global Image**: Always processed, produces 144 tokens (same as each crop)

### Step 7: Total Vision Tokens

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

## Why Actual Vision Tokens May Differ from Theoretical

**Important**: Even with precise control, actual vision tokens may be **slightly less** than the theoretical value `(num_crops + 1) × 144`.

### Causes

1. **Invalid patches**: Patches that extend beyond image boundaries are marked as `-100` in `image_input_idx`
2. **Padding regions**: Patches in padding areas (especially for non-square images) may be invalid
3. **Tiling edge cases**: Some crops may have partial patches at edges

### Practical Impact

- **Typical deviation**: 0-24 tokens (0-5% of theoretical value)
- **Consistency**: Deviation is consistent across similar images
- **Experimental impact**: Does not significantly affect experimental comparisons

**Best Practice**: Use `actual_vision_tokens` (from `image_input_idx`) for analysis rather than theoretical value.

## Control Methods: Vision Tokens List vs Image Size List

### Method 1: Vision Tokens List (Recommended) ✅

**Principle**: Specify target vision tokens, let the system adapt tiling to each image's aspect ratio.

**Workflow**:
```
Target Vision Tokens (e.g., 1008)
  → Calculate num_crops (6)
  → For each image:
      → Select best tiling based on original aspect ratio
      → Resize to optimal dimensions
      → Process with exact_num_crops=6
```

**Example Configuration**:
```python
vision_tokens_list = [432, 720, 1008, 1440]

# Corresponds to:
# 432 tokens  → 2 crops  (small images)
# 720 tokens  → 4 crops  (medium images)
# 1008 tokens → 6 crops  (large images)
# 1440 tokens → 9 crops  (very large images)
```

**Implementation**:
```python
# In acc_lat_profiling.py
target_vision_tokens = 1008
num_crops = tokens_to_crops(target_vision_tokens)  # = 6

mm_preprocessor = MultiModalPreprocessor(
    tokenizer=tokenizer,
    crop_mode="resize",
    max_crops=num_crops,  # = 6
    exact_num_crops=num_crops,  # Force exactly 6 crops
    resize_to_fill=True,  # Upscale small images to fill canvas
    # ... other parameters
)
```

**Advantages**:
- ✅ **Adaptive**: Each image gets the best tiling for its aspect ratio
- ✅ **Minimal distortion**: Aspect ratio is preserved as much as possible
- ✅ **Simple configuration**: Just specify vision token values
- ✅ **Consistent experiments**: All configs use same vision token targets

### Method 2: Image Size List (Legacy) ⚠️

**Principle**: Fix target image dimensions upfront, derive vision tokens from fixed tiling.

**Workflow**:
```
Fixed Image Size (e.g., 560×784)
  → Infer fixed tiling (2×3 = 6 crops)
  → Calculate vision tokens (1008)
  → Force all images to resize to 560×784
```

**Example Configuration**:
```python
image_size_list = ["560x336", "560x560", "560x784", "784x784"]
```

**Problems**:
- ❌ **Aspect ratio mismatch**: Fixed dimensions may not match original image aspect ratio
- ❌ **Image distortion**: Images are forced to resize to fixed dimensions, causing stretching/squashing
- ❌ **Padding overhead**: Large padding regions reduce effective vision tokens
- ❌ **Inconsistent tiling**: `select_tiling` may choose different tiling than inferred, causing confusion

**Example of Problem**:
```
Original image: 640×480 (aspect 0.75, tall)
Target size: 560×784 (aspect 1.4, wide)
  → Fixed tiling: 2×3 (6 crops)
  → select_tiling may choose (3,2) instead (better match for tall image)
  → Resize to 784×560 (aspect 0.71)
  → Result: Inconsistent! Target was 560×784 but actual is 784×560
```

## Detailed Comparison: Vision Tokens List vs Image Size List

### Example 1: Tall Image (640×480, aspect 0.75)

#### Using `vision_tokens_list = [1008]` ✅

```
Target: 1008 tokens (6 crops)
  → num_crops = 6
  → exact_num_crops = 6
  
For image 640×480 (aspect 0.75):
  → select_tiling finds possible 6-crop tilings:
     - (1,6): aspect 6.0 (too wide)
     - (2,3): aspect 1.5 (too wide)
     - (3,2): aspect 0.67 (closest to 0.75!) ✓
     - (6,1): aspect 0.17 (too narrow)
  → Selects (3,2) tiling
  → Target resolution: 784×560 (aspect 0.71)
  → Resize: 640×480 → 784×560 (minimal distortion)
  → Result: Consistent, minimal distortion
```

#### Using `image_size_list = ["560x784"]` ❌

```
Target: 560×784 (aspect 1.4)
  → Inferred tiling: 2×3 (6 crops)
  → exact_num_crops = 6
  
For image 640×480 (aspect 0.75):
  → select_tiling finds possible 6-crop tilings:
     - (3,2): aspect 0.67 (closest to 0.75!)
  → Selects (3,2) tiling (different from inferred!)
  → Target resolution: 784×560 (aspect 0.71)
  → But we wanted 560×784!
  → Result: Inconsistent! Target was 560×784 but actual is 784×560
```

### Example 2: Wide Image (1024×768, aspect 1.33)

#### Using `vision_tokens_list = [1008]` ✅

```
Target: 1008 tokens (6 crops)
  → num_crops = 6
  
For image 1024×768 (aspect 1.33):
  → select_tiling selects (2,3) tiling (aspect 1.5, closest to 1.33)
  → Target resolution: 560×784 (aspect 1.4)
  → Resize: 1024×768 → 560×784 (minimal distortion)
  → Result: Consistent, minimal distortion
```

#### Using `image_size_list = ["560x784"]` ⚠️

```
Target: 560×784 (aspect 1.4)
  → Inferred tiling: 2×3 (6 crops)
  
For image 1024×768 (aspect 1.33):
  → select_tiling selects (2,3) tiling (matches inferred)
  → Target resolution: 560×784 (aspect 1.4)
  → Resize: 1024×768 → 560×784 (acceptable)
  → Result: Works, but aspect ratio still has slight mismatch
```

### Example 3: Square Image (512×512, aspect 1.0)

#### Using `vision_tokens_list = [720]` ✅

```
Target: 720 tokens (4 crops)
  → num_crops = 4
  
For image 512×512 (aspect 1.0):
  → select_tiling selects (2,2) tiling (aspect 1.0, perfect match!)
  → Target resolution: 560×560 (aspect 1.0)
  → Resize: 512×512 → 560×560 (perfect match, no distortion)
  → Result: Perfect! Square tiling for square image
```

#### Using `image_size_list = ["560x784"]` ❌

```
Target: 560×784 (aspect 1.4)
  → Inferred tiling: 2×3 (6 crops)
  
For image 512×512 (aspect 1.0):
  → select_tiling must choose 6-crop tiling:
     - (2,3): aspect 1.5 (closest to 1.0)
  → Target resolution: 560×784 (aspect 1.4)
  → Resize: 512×512 → 560×784 (severe distortion!)
  → Result: Square image forced into wide format, severe distortion
```

## Resize to Fill: Handling Small Images

### Problem: Small Images Don't Use Full Token Budget

When a small image (e.g., 200×150) is processed with a large vision token target (e.g., 1440 tokens = 9 crops), the image may not fill the target canvas, resulting in:
- Large padding regions
- Reduced effective vision tokens
- Wasted computation

### Solution: `resize_to_fill=True` (Default)

When `resize_to_fill=True`, small images are **upscaled** to fill the target canvas before tiling:

```python
def resize_and_crop_to_fill(
    image: np.ndarray,
    target_h: int,
    target_w: int
) -> np.ndarray:
    """
    Scale image to fill target canvas, then crop/pad to exact dimensions.
    
    This ensures small images are upscaled to fully utilize the vision token budget.
    """
    # Calculate scale to FILL target (may upscale small images)
    scale = max(target_h / image_h, target_w / image_w)
    
    # Resize (may upscale)
    resized_h = int(image_h * scale)
    resized_w = int(image_w * scale)
    resized = cv2.resize(image, (resized_w, resized_h))
    
    # Crop or pad to exact target dimensions
    # ... (crop if too large, pad if too small)
    
    return final_image
```

**Example**:
```
Small image: 200×150 (aspect 0.75)
Target: 1440 tokens (9 crops) → tiling (3,3) → resolution 784×784

Without resize_to_fill:
  → Resize: 200×150 → 784×588 (downscale to fit, then pad)
  → Large padding regions, reduced effective tokens

With resize_to_fill=True:
  → Scale: max(784/200, 784/150) = 5.23 (upscale!)
  → Resize: 200×150 → 1046×784 (upscaled to fill)
  → Crop: 1046×784 → 784×784 (crop excess)
  → Result: Full utilization of vision token budget
```

**Trade-off**: Upscaling small images may introduce some artifacts, but ensures full utilization of the vision token budget, which is important for accuracy.

## Implementation in Experiments

### Configuration in Scripts

**H100 Script** (`run_multi_datasets_h100.sh`):
```bash
# Primary knob: vision tokens (target)
VISION_TOKENS_LIST="${VISION_TOKENS_LIST:-432 720 1008 1440}"

# Control whether to upscale small images
RESIZE_TO_FILL="${RESIZE_TO_FILL:-true}"

# Usage
torchrun --nproc-per-node=${NUM_GPUS} experiments/core_exp/acc_lat_profiling.py \
    --vision_tokens_list ${VISION_TOKENS_LIST} \
    --resize_to_fill  # If RESIZE_TO_FILL=true
```

**A100 Script** (`run_multi_datasets_a100.sh`):
```bash
# Same configuration
VISION_TOKENS_LIST="${VISION_TOKENS_LIST:-432 720 1008 1440}"
RESIZE_TO_FILL="${RESIZE_TO_FILL:-true}"
```

### Code Implementation

**In `acc_lat_profiling.py`**:
```python
def run(
    self,
    vision_tokens_list: List[int] = None,
    resize_to_fill: bool = True,
    ...
):
    # Step 1: Parse vision_tokens_list
    if vision_tokens_list is None:
        vision_tokens_list = [432, 720, 1008, 1296, 1584]
    
    # Step 2: Generate combinations
    combinations = self._generate_sparse_combinations(
        vision_tokens_list, top_k_list, num_active_blocks_list, ...
    )
    
    # Step 3: For each combination
    for target_vision_tokens, top_k, num_active_blocks in combinations:
        # Calculate num_crops
        num_crops = tokens_to_crops(target_vision_tokens)
        max_crops = num_crops
        
        # Step 4: Create preprocessor with exact_num_crops
        mm_preprocessor = MultiModalPreprocessor(
            tokenizer=self.tokenizer,
            crop_mode=self.model.config.crop_mode,
            max_crops=max_crops,
            exact_num_crops=num_crops,  # Force exact number of crops
            resize_to_fill=resize_to_fill,  # Upscale small images
            ...
        )
        
        # Step 5: Process images
        # For each image, select_tiling will:
        #   1. Find all tilings that result in exactly num_crops
        #   2. Select the one closest to original aspect ratio
        #   3. Resize image to optimal dimensions
```

### Result Recording

**Per-sample results**:
```json
{
  "target_vision_tokens": 1008,
  "target_crops": 6,
  "actual_vision_tokens": 1002,  // May be slightly less
  "theoretical_num_crops": 6,
  "theoretical_tiling": [3, 2],  // Selected based on image aspect ratio
  "theoretical_image_size": [784, 560],  // Calculated from tiling
  "actual_num_crops": 6,
  "actual_tiling": [3, 2],  // Should match theoretical
  "actual_image_size": [784, 560],  // Should match theoretical
  ...
}
```

**Config-level results**:
```json
{
  "target_vision_tokens": 1008,
  "target_crops": 6,
  "actual_vision_tokens_mean": 1001.5,  // Average across samples
  "theoretical_num_crops": 6,
  "theoretical_tiling": [3, 2],  // From first sample (may vary)
  "theoretical_image_size": [784, 560],
  "aggregate_stats": {
    "vision_tokens_mean": 1001.5,
    "vision_tokens_std": 12.3,
    "vision_tokens_diff_mean": 6.5,  // Theoretical - Actual
    ...
  },
  "per_sample_results": [...]
}
```

## Common Vision Token Values

### Recommended Values

| Vision Tokens | Crops | Typical Use Case | Example Resolution |
|---------------|-------|------------------|-------------------|
| 288 | 1 | Very small images | 336×336 |
| 432 | 2 | Small images | 336×560 or 560×336 |
| 576 | 3 | Small-medium images | 336×784 or 784×336 |
| 720 | 4 | Medium images | 560×560 |
| 1008 | 6 | Large images | 560×784 or 784×560 |
| 1296 | 8 | Very large images | 560×1008 or 1008×560 |
| 1440 | 9 | Very large images | 784×784 |
| 1584 | 10 | Extremely large images | 784×1008 or 1008×784 |
| 1872 | 12 | Maximum practical | 1008×1008 |

### Default Configuration

**H100 experiments**:
```bash
VISION_TOKENS_LIST="432 720 1008 1440"
# 2, 4, 6, 9 crops
```

**A100 experiments**:
```bash
VISION_TOKENS_LIST="432 720 1008 1440"
# Same as H100, but may use smaller batch sizes
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

**Recommendation**: 
- **Recommended**: `max_crops ≤ 20` for practical use
- **Feasible but requires caution**: `20 < max_crops ≤ 25`
- **Not recommended**: `max_crops > 25`

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

## Summary: Why Vision Tokens List is Better

### Problems with Image Size List

1. **Aspect ratio mismatch**: Fixed dimensions may not match original image aspect ratio
2. **Image distortion**: Images are forced to resize to fixed dimensions
3. **Padding overhead**: Large padding regions reduce effective vision tokens
4. **Inconsistent tiling**: `select_tiling` may choose different tiling than inferred
5. **Complex configuration**: Need to manually select multiple sizes to cover different aspect ratios

### Benefits of Vision Tokens List

1. ✅ **Adaptive tiling**: Each image gets the best tiling for its aspect ratio
2. ✅ **Minimal distortion**: Aspect ratio is preserved as much as possible
3. ✅ **Simple configuration**: Just specify vision token values
4. ✅ **Consistent experiments**: All configs use same vision token targets
5. ✅ **Better accuracy**: More vision tokens → better accuracy (when resize_to_fill=True)

### Trade-offs

1. **Upscaling small images**: `resize_to_fill=True` may introduce artifacts, but ensures full token utilization
2. **Variable image sizes**: Different images may have different final sizes (but same vision tokens)
3. **Theoretical vs actual**: Actual vision tokens may be slightly less than theoretical (0-5% deviation)

## Code References

- **Image preprocessing**: `molmo/data/model_preprocessor.py`
  - `select_tiling()`: Adaptive tiling selection
  - `resize_and_crop_to_fill()`: Image resizing with aspect ratio preservation
  - `image_to_patches_and_tokens()`: Vision token generation

- **Experiment code**: `experiments/core_exp/acc_lat_profiling.py`
  - `tokens_to_crops()`: Vision tokens → num_crops conversion
  - `crops_to_tiling()`: Num_crops → tiling selection
  - `calculate_theoretical_values()`: Theoretical value calculation

- **Vision encoding**: `molmo/models/modeling_molmoe.py`
  - `encode_image()`: Vision encoder processing
  - `forward()` (VisionBackbone): Complete vision encoding pipeline

## Real-World Examples

For detailed examples with real images from VQA v2 dataset, see:
- **`vision_tokens_knob_examples.md`**: Real image examples showing how the vision tokens control knob works with different image resolutions and aspect ratios
  - Example 1: Small tall image (480×640) with 432 and 1008 vision tokens
  - Example 2: Medium square image (640×640) with 720 vision tokens
  - Example 3: Large wide image (1024×768) with 1440 and 1008 vision tokens
  - Example 4: Very small image (200×150) with resize_to_fill
  - Practical recommendations based on real examples

## Related Documents

- `vision_tokens_knob_examples.md`: Real-world examples with actual images
- `vision_tokens_knob_qa.md`: Q&A on naming and tier-based design
- `vision_tokens_knob_tier_design_discussion.md`: Detailed discussion on tier-based design
- `vision_tokens_knob_tier_design_summary.md`: Quick summary of tier-based design options
- `../core_exp/vision_tokens_list_vs_image_size_list.md`: Detailed comparison
- `moe_topk_knob.md`: MoE top-K control knob
- `transformer_blocks_knob.md`: Transformer blocks control knob
- `../core_exp/migration_to_vision_tokens_list.md`: Migration guide
