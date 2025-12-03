# Experiment 6: Crop Overlap Analysis

## Overview

Experiment 6 analyzes how images are divided into crops and quantifies the overlap and redundant computation. This experiment is designed to help understand how to effectively control the number of vision tokens by examining different methods of crop management.

## Goal

**Primary Goal**: Understand how images are tiled into crops and analyze overlap patterns to inform strategies for controlling vision token count.

**Key Questions**:
1. How does the `select_tiling` function determine the number of crops?
2. What methods can be used to control the number of vision tokens?
3. Which method is most practical and effective?

## Methodology

### Image Selection

The experiment selects images from Exp 2 results based on crop counts:

- **Selection Strategy**: Evenly distributed across the crop count range
- **Default**: 5 images with different crop counts (e.g., 2, 3, 4, 5, 10 crops)
- **Target Max Crops**: Configurable (default: 12) - will select closest available if not found
- **Source**: `exp2_component_profiling.json` results

### Analysis Components

For each selected image, the experiment:

1. **Crop Analysis**: 
   - Calculates tiling configuration (rows × cols)
   - Determines crop boundaries in both original and resized image space
   - Computes overlap statistics

2. **Overlap Quantification**:
   - **Overlap Ratio**: Percentage of patches that are covered by multiple crops
   - **Redundancy Ratio**: Total patch area / Unique patch area
   - **Average Overlap Count**: Average number of times each patch is covered

3. **Crop Control Analysis**:
   - Analyzes all possible tiling configurations
   - Evaluates alternative `max_crops` values
   - Calculates image size requirements for different crop counts
   - Documents methods for controlling vision tokens

## How `num_crops` is Determined

### The `select_tiling` Function

The number of crops is determined by the `select_tiling` function in `molmo/data/model_preprocessor.py`:

```python
def select_tiling(h, w, patch_size, max_num_crops):
    """Divide image of size [w, h] into up to max_num_crops of size patch_size"""
```

**Algorithm**:
1. **Generate Candidates**: Consider all possible tiling configurations (i×j) where i×j ≤ max_crops
2. **Calculate Scaling**: For each candidate, calculate required scaling to fit the image
3. **Select Optimal**: Choose the tiling that requires the **least upscaling** (or least downscaling if all require downscaling)

**Key Parameters**:
- `h, w`: Effective image dimensions (after subtracting margin pixels)
- `patch_size`: Crop window size (typically 224 pixels)
- `max_num_crops`: Maximum allowed crops (default: 12)

**Example**:
- For an image of 640×425 pixels with `max_crops=12`:
  - Effective size: 640-112 = 528 (width), 425-112 = 313 (height)
  - Candidate tilings: 1×1, 1×2, 2×1, 1×3, 3×1, ..., up to 3×4, 4×3, etc.
  - Selected: The tiling requiring least upscaling (e.g., 1×2 = 2 crops)

## Methods to Control Vision Token Count

Based on the analysis, there are four main methods to control the number of vision tokens:

### Method 1: Change `max_crops` Parameter

**How it works**: Adjust the `max_crops` parameter in model configuration.

**Pros**:
- Simple: Single parameter change
- Direct: Directly limits maximum crops
- No image modification needed

**Cons**:
- Coarse control: Only sets upper bound
- May not achieve desired exact count
- Requires model reconfiguration

**Implementation**:
```python
# In model config
max_crops = 6  # or 9, 12, 15, etc.
```

**Effectiveness**: ⭐⭐⭐⭐ (4/5) - Direct but coarse

### Method 2: Resize Image

**How it works**: Resize the input image to trigger different tiling configurations.

**Pros**:
- Fine-grained control: Can target specific crop counts
- No model changes: Works with existing model
- Flexible: Can be done at preprocessing stage

**Cons**:
- Requires calculation: Need to determine target size
- May affect quality: Upscaling can introduce artifacts
- Complex: Need to understand tiling algorithm

**Implementation**:
```python
# Calculate target size for desired crop count
# For 12 crops with 3×4 tiling:
target_h = 3 * crop_window_size + total_margin_pixels
target_w = 4 * crop_window_size + total_margin_pixels
resized_image = resize(image, (target_w, target_h))
```

**Effectiveness**: ⭐⭐⭐⭐⭐ (5/5) - Most flexible and precise

### Method 3: Change `overlap_margins`

**How it works**: Adjust the overlap margins to change `crop_window_size`.

**Pros**:
- Affects tiling: Changes effective crop size
- Can reduce overlap: Smaller margins = less overlap

**Cons**:
- Limited impact: Only affects crop window size
- May affect quality: Too small margins may lose context
- Requires model changes: Needs configuration update

**Implementation**:
```python
# In model config
overlap_margins = (2, 2)  # Smaller margins = larger crop_window_size
```

**Effectiveness**: ⭐⭐ (2/5) - Limited and indirect

### Method 4: Change `base_image_input_size`

**How it works**: Adjust the base image input size to change `crop_window_size`.

**Pros**:
- Direct control: Affects crop size directly
- Can be tuned: Fine-tune for specific use cases

**Cons**:
- Model-dependent: Requires model architecture changes
- Complex: Affects multiple components
- Not practical: Usually fixed in model design

**Implementation**:
```python
# In model config
base_image_input_size = (448, 448)  # Larger = larger crops
```

**Effectiveness**: ⭐ (1/5) - Not practical for runtime control

## Recommended Approach

### For Runtime Control: **Method 2 (Resize Image)**

**Why**:
1. **Flexibility**: Can achieve any desired crop count
2. **No Model Changes**: Works with existing model
3. **Precise**: Can target exact crop counts
4. **Practical**: Can be implemented at preprocessing stage

**Implementation Strategy**:
```python
def resize_for_target_crops(image, target_crops, max_crops=12, 
                            crop_window_size=224, overlap_margins=(4, 4)):
    """
    Resize image to achieve target number of crops.
    
    Args:
        image: Input image (H, W, 3)
        target_crops: Desired number of crops
        max_crops: Maximum allowed crops
        crop_window_size: Size of each crop window
        overlap_margins: Overlap margins (left, right)
    
    Returns:
        Resized image that will produce target_crops
    """
    # Find tiling configuration for target_crops
    # For target_crops=12, possible tilings: 1×12, 2×6, 3×4, 4×3, 6×2, 12×1
    # Select based on image aspect ratio
    
    base_image_input_d = 14  # patch size
    left_margin, right_margin = overlap_margins
    total_margin_pixels = base_image_input_d * (right_margin + left_margin)
    
    # Calculate target size for desired tiling
    # Example: 3×4 tiling for 12 crops
    rows, cols = find_optimal_tiling(target_crops, image.shape[:2])
    
    target_h = rows * crop_window_size + total_margin_pixels
    target_w = cols * crop_window_size + total_margin_pixels
    
    # Resize image
    resized = resize(image, (target_w, target_h))
    return resized
```

### For Model Configuration: **Method 1 (Change max_crops)**

**Why**:
1. **Simple**: Single parameter
2. **Direct**: Sets upper bound
3. **Consistent**: Same behavior across all images

**Use Case**: When you want to limit maximum computation but don't need fine-grained control.

## Output Files

### Visualizations

1. **`exp6_crop_overlap_image_{image_id}_crops.png`**
   - Original image with crop boundaries overlaid
   - Shows how image is divided into crops
   - Color-coded boundaries for each crop

2. **`exp6_crop_overlap_image_{image_id}_comparison.png`**
   - **Left**: Original image with crop boundaries
   - **Right**: Stitched crops with grid lines
   - Side-by-side comparison for easy understanding

3. **`exp6_crop_overlap_image_{image_id}_statistics.png`**
   - Bar chart showing patch coverage statistics
   - Overlap ratio and redundancy metrics

### Data Files

1. **`exp6_crop_overlap_analysis.json`**
   - Complete analysis results for all images
   - Includes:
     - Crop boundaries and coordinates
     - Overlap statistics
     - Crop control information
     - All possible tiling configurations

2. **`crops/{prefix}_original.png`**
   - Original image for each analyzed image

3. **`crops/{prefix}_crop_{idx:02d}.png`**
   - Individual crop images for each crop

## Key Insights

### Overlap Patterns

1. **Rectangular Tilings (1×N or N×1)**:
   - **Low Overlap**: Typically 0-15% overlap
   - **Efficient**: Minimal redundant computation
   - **Common**: For images with high aspect ratio

2. **Square Tilings (N×N)**:
   - **High Overlap**: Can reach 30-40% overlap
   - **More Redundancy**: Higher computational cost
   - **Better Coverage**: More complete image coverage

### Redundancy Analysis

- **Redundancy Ratio**: Typically 1.0x to 1.6x
  - 1.0x = No overlap (perfect tiling)
  - 1.6x = 60% redundant computation
- **Impact**: Higher redundancy = more vision tokens = higher latency

### Crop Count Distribution

From Exp 2 results on VQA v2 validation set:
- **2 crops**: 17 images (0.1%)
- **3 crops**: 255 images (1.3%)
- **4 crops**: 56 images (0.3%)
- **5 crops**: 466 images (2.4%)
- **7 crops**: 3,980 images (20.5%) - **Most common**
- **10 crops**: 226 images (1.2%)

**Observation**: Most images use 7 crops, suggesting the default `max_crops=12` is rarely fully utilized.

## Usage

### Basic Usage

```bash
# Use default settings (5 images, target 12 crops)
bash experiments/motivate/run_exp6.sh

# Customize selection
bash experiments/motivate/run_exp6.sh 0 --num_images 7 --target_max_crops 12
```

### Python API

```python
from experiments.motivate.exp6_crop_overlap_analysis import CropOverlapAnalysis

experiment = CropOverlapAnalysis(
    model_path="checkpoints",
    output_dir="./results/motivation/exp6"
)

results = experiment.run(
    dataset_name="coco_2014_vqa",
    split="validation",
    exp2_results_path="./results/motivation/exp2/exp2_component_profiling.json",
    num_images=5,
    target_max_crops=12,
    save_crop_images=True
)
```

## Practical Recommendations

### For Adaptive Inference Systems

1. **Dynamic Resizing** (Recommended):
   - Implement Method 2 (resize image) for runtime control
   - Create lookup table: `target_crops → target_image_size`
   - Resize based on desired latency/quality tradeoff

2. **Hybrid Approach**:
   - Use Method 1 (max_crops) for coarse control
   - Use Method 2 (resize) for fine-tuning within that bound

3. **Quality-Aware Resizing**:
   - Monitor image quality after resizing
   - Use adaptive resizing based on image content
   - Consider aspect ratio when selecting tiling

### Implementation Example

```python
def adaptive_crop_control(image, target_latency, quality_threshold):
    """
    Adaptively control crops based on target latency.
    """
    # Estimate current latency based on image size
    current_crops = estimate_crops(image)
    current_latency = estimate_latency(current_crops)
    
    if current_latency <= target_latency:
        return image  # No change needed
    
    # Calculate target crops for desired latency
    target_crops = calculate_target_crops(target_latency)
    
    # Resize to achieve target crops
    resized = resize_for_target_crops(image, target_crops)
    
    # Verify quality
    if calculate_quality(resized) >= quality_threshold:
        return resized
    else:
        # Fallback: Use max_crops limit
        return limit_max_crops(image, target_crops)
```

## Conclusion

**Best Method for Controlling Vision Tokens**: **Resize Image (Method 2)**

- Most flexible and precise
- No model changes required
- Can be implemented at preprocessing stage
- Allows fine-grained control

**Practical Implementation**:
1. Create a mapping from target crop counts to image sizes
2. Implement dynamic resizing based on latency/quality requirements
3. Monitor overlap and redundancy to optimize efficiency

This approach provides the best balance between control, flexibility, and practicality for adaptive inference systems.


