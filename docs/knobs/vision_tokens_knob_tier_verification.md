# Vision Tokens Control Knob: Tier-Based Selection Verification

## Question

**If a small image is processed with a high tier (e.g., 9-15 crops), will the image be resized to a larger size and divided into more crops?**

## Answer: **YES** ✅

Let's verify this step-by-step with a concrete example.

## Example: Small Image with High Tier

### Input
- **Original image**: 200×150 pixels (small, aspect ratio 0.75)
- **Tier**: High tier (9-15 crops, preferred: [9, 12, 15])
- **Configuration**:
  - `crop_mode = "overlap-and-resize-c2"`
  - `resize_to_fill = True` (default)
  - `base_image_input_size = (336, 336)`
  - `overlap_margins = (4, 4)`
  - `image_patch_size = 14`
  - `crop_window_size = 224` (after accounting for margins)

### Step-by-Step Processing

#### Step 1: Calculate Image Dimensions After Margins

```python
# From code: molmo/preprocessors/multimodal_preprocessor.py:549
total_margin_pixels = base_image_input_d * (right_margin + left_margin)
# = 14 * (4 + 4) = 112 pixels

original_image_h = 200
original_image_w = 150

h_after_margins = 200 - 112 = 88
w_after_margins = 150 - 112 = 38
image_aspect_ratio = 38 / 88 = 0.43 (very tall, narrow image)
```

#### Step 2: Tier-Based Tiling Selection

```python
# From code: molmo/preprocessors/multimodal_preprocessor.py:302-391
tier = {
    "min_crops": 9,
    "max_crops": 15,
    "preferred_crops": [9, 12, 15],
    "mismatch_threshold": 0.3,
}

# select_tiling is called with:
tiling = select_tiling(
    h=88,   # h_after_margins
    w=38,   # w_after_margins
    patch_size=224,  # crop_window_size
    max_num_crops=15,
    tier=tier
)
```

**Tier Selection Process**:

1. **Try preferred crop counts**:
   - **9 crops**: Possible tilings: (1,9), (3,3), (9,1)
     - (1,9): aspect = 9.0, mismatch = |9.0 - 0.43| = 8.57
     - (3,3): aspect = 1.0, mismatch = |1.0 - 0.43| = 0.57
     - (9,1): aspect = 0.11, mismatch = |0.11 - 0.43| = 0.32 ✓ (best for 9 crops)
   
   - **12 crops**: Possible tilings: (1,12), (2,6), (3,4), (4,3), (6,2), (12,1)
     - (12,1): aspect = 0.083, mismatch = |0.083 - 0.43| = 0.35
     - (6,2): aspect = 0.33, mismatch = |0.33 - 0.43| = 0.10 ✓ (best for 12 crops)
   
   - **15 crops**: Possible tilings: (1,15), (3,5), (5,3), (15,1)
     - (15,1): aspect = 0.067, mismatch = |0.067 - 0.43| = 0.36
     - (5,3): aspect = 0.6, mismatch = |0.6 - 0.43| = 0.17
     - (3,5): aspect = 1.67, mismatch = |1.67 - 0.43| = 1.24
     - Best: (5,3) with mismatch = 0.17

2. **Select best**: 12 crops with (6,2) tiling (mismatch = 0.10) ✓

**Result**: 
- **Selected crops**: 12
- **Selected tiling**: (6, 2) - 6 rows, 2 columns
- **Aspect ratio mismatch**: 0.10 (excellent match!)

#### Step 3: Calculate Target Resolution

```python
# From code: molmo/preprocessors/multimodal_preprocessor.py:563-568
target_h = tiling[0] * crop_window_size + total_margin_pixels
target_w = tiling[1] * crop_window_size + total_margin_pixels

target_h = 6 * 224 + 112 = 1344 + 112 = 1456
target_w = 2 * 224 + 112 = 448 + 112 = 560

target_resolution = (1456, 560)  # Much larger than original 200×150!
```

**Key Point**: The target resolution (1456×560) is **much larger** than the original image (200×150):
- Height: 1456 / 200 = **7.28× larger**
- Width: 560 / 150 = **3.73× larger**

#### Step 4: Resize Image (with `resize_to_fill=True`)

```python
# From code: molmo/preprocessors/multimodal_preprocessor.py:563-568
src, img_mask = self.resize_image(
    image,  # Original: 200×150
    [target_h, target_w],  # Target: 1456×560
    is_training,
    rng
)

# resize_image calls resize_and_crop_to_fill when resize_to_fill=True
# From code: molmo/preprocessors/multimodal_preprocessor.py:130-217
```

**Resize Process** (`resize_and_crop_to_fill`):

```python
# From code: molmo/preprocessors/multimodal_preprocessor.py:147-150
desired_height = 1456
desired_width = 560
height = 200
width = 150

# Calculate scale to FILL target (use max ratio, upsample if needed)
image_scale_y = 1456 / 200 = 7.28
image_scale_x = 560 / 150 = 3.73
image_scale = max(7.28, 3.73) = 7.28  # Upscale by 7.28×!

# Resize preserving aspect ratio
scaled_height = int(200 * 7.28) = 1456
scaled_width = int(150 * 7.28) = 1092

# Resized image: 1456×1092 (aspect 0.75, preserved!)
```

**Center-Crop to Target Size**:

```python
# From code: molmo/preprocessors/multimodal_preprocessor.py:190-197
# Resized: 1456×1092
# Target: 1456×560

# Center-crop width (crop excess)
crop_left = (1092 - 560) // 2 = 266
crop_right = crop_left + 560 = 826

cropped = resized[:, 266:826]  # 1456×560
```

**Final Result**:
- **Original**: 200×150
- **Resized**: 1456×1092 (upscaled 7.28×, aspect ratio preserved)
- **Final**: 1456×560 (cropped to target, aspect ratio 0.38)

#### Step 5: Split into Crops

```python
# From code: molmo/preprocessors/multimodal_preprocessor.py:571-640
# Image is split into 12 crops (6 rows × 2 columns)
n_crops = 6 * 2 = 12

# Each crop is 336×336 (base_image_input_size)
# Crops are extracted from the 1456×560 resized image
```

**Result**: 
- ✅ Image is **upscaled** from 200×150 to 1456×560 (7.28× larger)
- ✅ Image is **divided into 12 crops** (6 rows × 2 columns)
- ✅ Each crop is 336×336 pixels
- ✅ Full utilization of vision token budget

## Verification Summary

### ✅ Confirmed: Small Image + High Tier = Upscaling + More Crops

| Aspect | Original Size | Tier | Selected Crops | Target Resolution | Upscale Factor |
|--------|--------------|------|----------------|-------------------|---------------|
| Tall (0.43) | 200×150 | High (9-15) | 12 (6×2) | 1456×560 | 7.28× |
| Square (1.0) | 200×200 | High (9-15) | 9 (3×3) | 784×784 | 3.92× |
| Wide (1.5) | 200×300 | High (9-15) | 12 (3×4) | 784×1008 | 3.36× |

### Key Points

1. **Tier selection is adaptive**: `select_tiling` chooses the best crop count within tier range based on image aspect ratio
2. **Target resolution is calculated from tiling**: `target = tiling × crop_window_size + margins`
3. **`resize_to_fill=True` upscales small images**: Uses `max(scale_x, scale_y)` to ensure full canvas coverage
4. **More crops = larger target resolution**: Higher tier → more crops → larger target → more upscaling

### Code Flow

```
Small Image (200×150)
    ↓
select_tiling(tier=high) → Selects 12 crops, tiling (6,2)
    ↓
Calculate target: 1456×560
    ↓
resize_and_crop_to_fill(image, target) → Upscale 7.28×
    ↓
Split into 12 crops (6×2 grid)
    ↓
Result: 12 crops of 336×336 each
```

## Conclusion

**YES**, when a small image is processed with a high tier:
1. ✅ The image **will be resized to a larger size** (upscaled to fill target canvas)
2. ✅ The image **will be divided into more crops** (based on tier selection)
3. ✅ The upscaling factor depends on the selected tiling and target resolution
4. ✅ `resize_to_fill=True` ensures full token budget utilization

This is the **intended behavior** to ensure that:
- Small images fully utilize the vision token budget
- More vision tokens can bring accuracy benefits
- Aspect ratio is preserved during upscaling (minimal distortion)

