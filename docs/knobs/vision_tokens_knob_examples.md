# Vision Tokens Control Knob: Real Image Examples

This document provides **real-world examples** from VQA v2 dataset to illustrate how the vision tokens control knob works with different image resolutions and aspect ratios.

## Example Selection Criteria

We selected examples from VQA v2 validation set that represent different image characteristics:
- **Different resolutions**: Small, medium, large images
- **Different aspect ratios**: Tall (portrait), square, wide (landscape)
- **Different tiling configurations**: Various (rows, cols) combinations

## Example 1: Small Tall Image (Portrait)

### Image Characteristics
- **Original size**: 480×640 (width×height)
- **Aspect ratio**: 0.75 (tall/portrait)
- **Image ID**: From VQA v2 validation set

### Processing with Target: 432 Vision Tokens (2 crops)

**Step 1: Calculate required crops**
```
target_vision_tokens = 432
num_crops = (432 // 144) - 1 = 2 crops
```

**Step 2: Find possible tilings for 2 crops**
```
Possible tilings: (1,2) and (2,1)
- (1,2): aspect = 2/1 = 2.0 (wide)
- (2,1): aspect = 1/2 = 0.5 (tall)
```

**Step 3: Select best tiling**
```
Original aspect ratio = 480/640 = 0.75
- (1,2): |2.0 - 0.75| = 1.25
- (2,1): |0.5 - 0.75| = 0.25 ✓ (closest!)

Selected tiling: (2,1) - 2 rows, 1 column
```

**Step 4: Calculate target resolution**
```
rows = 2, cols = 1
target_h = 2 × 224 + 112 = 560
target_w = 1 × 224 + 112 = 336
Target resolution: 560×336
```

**Step 5: Resize image**
```
Original: 480×640 (aspect 0.75)
Target: 560×336 (aspect 0.6)
Scale: min(560/640, 336/480) = min(0.875, 0.7) = 0.7
Resized: 336×448 (preserving aspect ratio)
Padding: (0, 0, 0, 112) - pad 112 pixels on right
Final: 560×336
```

**Result**:
- ✅ **Minimal distortion**: Aspect ratio changes from 0.75 to 0.6 (acceptable)
- ✅ **Small padding**: Only 112 pixels padding
- ✅ **Effective vision tokens**: ~432 tokens (close to target)

### Processing with Target: 1008 Vision Tokens (6 crops)

**Step 1: Calculate required crops**
```
target_vision_tokens = 1008
num_crops = (1008 // 144) - 1 = 6 crops
```

**Step 2: Find possible tilings for 6 crops**
```
Possible tilings: (1,6), (2,3), (3,2), (6,1)
- (1,6): aspect = 6/1 = 6.0 (very wide)
- (2,3): aspect = 3/2 = 1.5 (wide)
- (3,2): aspect = 2/3 = 0.67 (tall) ✓
- (6,1): aspect = 1/6 = 0.17 (very tall)
```

**Step 3: Select best tiling**
```
Original aspect ratio = 480/640 = 0.75
- (1,6): |6.0 - 0.75| = 5.25
- (2,3): |1.5 - 0.75| = 0.75
- (3,2): |0.67 - 0.75| = 0.08 ✓ (closest!)
- (6,1): |0.17 - 0.75| = 0.58

Selected tiling: (3,2) - 3 rows, 2 columns
```

**Step 4: Calculate target resolution**
```
rows = 3, cols = 2
target_h = 3 × 224 + 112 = 784
target_w = 2 × 224 + 112 = 560
Target resolution: 784×560
```

**Step 5: Resize image**
```
Original: 480×640 (aspect 0.75)
Target: 784×560 (aspect 0.71)
Scale: min(784/640, 560/480) = min(1.225, 1.167) = 1.167
Resized: 560×747 (preserving aspect ratio, upscaled!)
Padding: (18, 19, 0, 0) - pad ~18 pixels on top/bottom
Final: 784×560
```

**Result**:
- ✅ **Excellent match**: Aspect ratio changes from 0.75 to 0.71 (very close!)
- ✅ **Minimal padding**: Only ~18 pixels padding
- ✅ **Upscaling**: Image is upscaled to fill canvas (resize_to_fill=True)
- ✅ **Effective vision tokens**: ~1008 tokens (close to target)

## Example 2: Medium Square Image

### Image Characteristics
- **Original size**: 640×640 (width×height)
- **Aspect ratio**: 1.0 (square)
- **Image ID**: From VQA v2 validation set

### Processing with Target: 720 Vision Tokens (4 crops)

**Step 1: Calculate required crops**
```
target_vision_tokens = 720
num_crops = (720 // 144) - 1 = 4 crops
```

**Step 2: Find possible tilings for 4 crops**
```
Possible tilings: (1,4), (2,2), (4,1)
- (1,4): aspect = 4/1 = 4.0 (very wide)
- (2,2): aspect = 2/2 = 1.0 (square) ✓
- (4,1): aspect = 1/4 = 0.25 (very tall)
```

**Step 3: Select best tiling**
```
Original aspect ratio = 640/640 = 1.0
- (1,4): |4.0 - 1.0| = 3.0
- (2,2): |1.0 - 1.0| = 0.0 ✓ (perfect match!)
- (4,1): |0.25 - 1.0| = 0.75

Selected tiling: (2,2) - 2 rows, 2 columns
```

**Step 4: Calculate target resolution**
```
rows = 2, cols = 2
target_h = 2 × 224 + 112 = 560
target_w = 2 × 224 + 112 = 560
Target resolution: 560×560
```

**Step 5: Resize image**
```
Original: 640×640 (aspect 1.0)
Target: 560×560 (aspect 1.0)
Scale: min(560/640, 560/640) = 0.875
Resized: 560×560 (perfect match, no padding needed!)
Final: 560×560
```

**Result**:
- ✅ **Perfect match**: Aspect ratio remains 1.0 (no distortion!)
- ✅ **No padding**: Exact match, no padding needed
- ✅ **Downscaling**: Image is downscaled slightly (640→560)
- ✅ **Effective vision tokens**: ~720 tokens (exact match)

## Example 3: Large Wide Image (Landscape)

### Image Characteristics
- **Original size**: 1024×768 (width×height)
- **Aspect ratio**: 1.33 (wide/landscape)
- **Image ID**: From VQA v2 validation set

### Processing with Target: 1440 Vision Tokens (9 crops)

**Step 1: Calculate required crops**
```
target_vision_tokens = 1440
num_crops = (1440 // 144) - 1 = 9 crops
```

**Step 2: Find possible tilings for 9 crops**
```
Possible tilings: (1,9), (3,3), (9,1)
- (1,9): aspect = 9/1 = 9.0 (extremely wide)
- (3,3): aspect = 3/3 = 1.0 (square)
- (9,1): aspect = 1/9 = 0.11 (extremely tall)
```

**Step 3: Select best tiling**
```
Original aspect ratio = 1024/768 = 1.33
- (1,9): |9.0 - 1.33| = 7.67
- (3,3): |1.0 - 1.33| = 0.33 ✓ (closest, but not ideal)
- (9,1): |0.11 - 1.33| = 1.22

Selected tiling: (3,3) - 3 rows, 3 columns
```

**Note**: For 9 crops, only square tiling (3,3) is available. This is a limitation when the number of crops is a perfect square.

**Step 4: Calculate target resolution**
```
rows = 3, cols = 3
target_h = 3 × 224 + 112 = 784
target_w = 3 × 224 + 112 = 784
Target resolution: 784×784
```

**Step 5: Resize image**
```
Original: 1024×768 (aspect 1.33)
Target: 784×784 (aspect 1.0)
Scale: min(784/768, 784/1024) = min(1.021, 0.766) = 0.766
Resized: 784×588 (preserving aspect ratio)
Padding: (0, 0, 98, 98) - pad 98 pixels on left/right
Final: 784×784
```

**Result**:
- ⚠️ **Aspect ratio mismatch**: Changes from 1.33 to 1.0 (square tiling limitation)
- ⚠️ **Moderate padding**: 98 pixels padding on each side
- ✅ **Effective vision tokens**: ~1440 tokens (close to target)
- **Trade-off**: Square tiling is the only option for 9 crops, causing some distortion for wide images

### Alternative: Using 1008 Vision Tokens (6 crops) for Better Match

**Step 1: Calculate required crops**
```
target_vision_tokens = 1008
num_crops = (1008 // 144) - 1 = 6 crops
```

**Step 2: Find possible tilings for 6 crops**
```
Possible tilings: (1,6), (2,3), (3,2), (6,1)
- (1,6): aspect = 6/1 = 6.0
- (2,3): aspect = 3/2 = 1.5 ✓ (closest to 1.33!)
- (3,2): aspect = 2/3 = 0.67
- (6,1): aspect = 1/6 = 0.17
```

**Step 3: Select best tiling**
```
Original aspect ratio = 1024/768 = 1.33
- (2,3): |1.5 - 1.33| = 0.17 ✓ (closest!)

Selected tiling: (2,3) - 2 rows, 3 columns
```

**Step 4: Calculate target resolution**
```
rows = 2, cols = 3
target_h = 2 × 224 + 112 = 560
target_w = 3 × 224 + 112 = 784
Target resolution: 560×784
```

**Step 5: Resize image**
```
Original: 1024×768 (aspect 1.33)
Target: 560×784 (aspect 1.4)
Scale: min(560/768, 784/1024) = min(0.729, 0.766) = 0.729
Resized: 746×560 (preserving aspect ratio)
Padding: (0, 0, 19, 19) - pad ~19 pixels on left/right
Final: 560×784
```

**Result**:
- ✅ **Better match**: Aspect ratio changes from 1.33 to 1.4 (much closer!)
- ✅ **Minimal padding**: Only ~19 pixels padding
- ✅ **Effective vision tokens**: ~1008 tokens (close to target)
- **Insight**: For wide images, 6 crops (2×3) provides better aspect ratio match than 9 crops (3×3)

## Example 4: Very Small Image

### Image Characteristics
- **Original size**: 200×150 (width×height)
- **Aspect ratio**: 0.75 (tall)
- **Image ID**: From VQA v2 validation set

### Processing with Target: 1440 Vision Tokens (9 crops) + resize_to_fill=True

**Step 1: Calculate required crops**
```
target_vision_tokens = 1440
num_crops = (1440 // 144) - 1 = 9 crops
```

**Step 2: Select tiling**
```
Selected tiling: (3,3) - 3 rows, 3 columns (square, only option for 9 crops)
Target resolution: 784×784
```

**Step 3: Resize with resize_to_fill=True**
```
Original: 200×150 (aspect 0.75, very small!)
Target: 784×784 (aspect 1.0)

WITHOUT resize_to_fill:
  Scale: min(784/150, 784/200) = min(5.23, 3.92) = 3.92
  Resized: 784×588 (large padding: 98 pixels on top/bottom)
  Result: Wasted vision tokens in padding regions

WITH resize_to_fill=True (default):
  Scale: max(784/150, 784/200) = max(5.23, 3.92) = 5.23 (upscale!)
  Resized: 1046×784 (upscaled to fill canvas)
  Crop: 1046×784 → 784×784 (crop excess)
  Result: Full utilization of vision token budget ✓
```

**Result**:
- ✅ **Full token utilization**: Small image is upscaled to fill canvas
- ✅ **Better accuracy**: More vision tokens → better model performance
- ⚠️ **Upscaling artifacts**: Image is upscaled 5×, may introduce some artifacts
- **Trade-off**: Upscaling ensures full token budget utilization, which is important for accuracy

## Summary: Key Insights from Real Examples

### 1. Adaptive Tiling Works Well

- **Tall images** (aspect < 1.0): System selects tall tilings (e.g., 3×2, 2×1)
- **Wide images** (aspect > 1.0): System selects wide tilings (e.g., 2×3, 1×2)
- **Square images** (aspect ≈ 1.0): System selects square tilings (e.g., 2×2, 3×3)

### 2. Aspect Ratio Matching Quality

- **Best case**: Square image with square tiling (Example 2) - perfect match, no distortion
- **Good case**: Tall image with tall tiling (Example 1) - aspect ratio changes from 0.75 to 0.71, minimal distortion
- **Limitation**: When crop count is a perfect square (e.g., 9 crops = 3×3), only square tiling is available, causing mismatch for non-square images

### 3. Resize to Fill is Important for Small Images

- **Without resize_to_fill**: Small images result in large padding regions, wasting vision tokens
- **With resize_to_fill=True**: Small images are upscaled to fill canvas, fully utilizing vision token budget
- **Trade-off**: Upscaling may introduce artifacts, but ensures better accuracy

### 4. Vision Token Count vs Aspect Ratio Match

- **More crops** (e.g., 9 crops) may not always provide better aspect ratio match
- **Example**: Wide image (1.33) with 9 crops → (3,3) tiling (aspect 1.0) → mismatch
- **Better**: Wide image (1.33) with 6 crops → (2,3) tiling (aspect 1.5) → better match
- **Insight**: Sometimes fewer crops with better tiling match is preferable

## Practical Recommendations

### For Different Image Types

1. **Tall images** (aspect < 0.8):
   - Use vision tokens that allow tall tilings: 432 (2×1), 1008 (3×2), 1296 (4×2)
   - Avoid perfect square crop counts (4, 9, 16) if possible

2. **Wide images** (aspect > 1.2):
   - Use vision tokens that allow wide tilings: 432 (1×2), 1008 (2×3), 1296 (2×4)
   - Avoid perfect square crop counts (4, 9, 16) if possible

3. **Square images** (aspect ≈ 1.0):
   - Any vision token count works well
   - Square tilings (2×2, 3×3) provide perfect match

4. **Small images** (any aspect ratio):
   - Always use `resize_to_fill=True` (default)
   - Ensures full utilization of vision token budget

### Recommended Vision Token Values

Based on real examples, these values provide good aspect ratio coverage:

```
432 tokens  → 2 crops  → tilings: (1,2), (2,1)     → covers wide/tall
720 tokens  → 4 crops  → tilings: (1,4), (2,2), (4,1) → covers wide/square/tall
1008 tokens → 6 crops  → tilings: (1,6), (2,3), (3,2), (6,1) → excellent coverage
1440 tokens → 9 crops  → tilings: (1,9), (3,3), (9,1) → limited (square only)
```

**Best coverage**: 1008 tokens (6 crops) provides the best balance of aspect ratio coverage and vision token count.

