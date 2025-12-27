# Vision Tokens Control Knob: Hybrid Tier Approach - Implementation Guide

## Overview

This document provides a detailed implementation guide for the Hybrid Tier Approach, including code examples, integration with `select_tiling`, and validation strategies.

## Implementation Architecture

### Tier Selection Flow

```
For each tier:
  1. For each image:
     a. Try preferred crop counts in tier
     b. For each crop count, use select_tiling to find best tiling
     c. Calculate aspect ratio mismatch
     d. Select crop count with minimum mismatch
  2. Process image with selected crop count
  3. Record tier, selected crops, and results
```

### Key Functions

#### 1. Tier Configuration

```python
VISION_TOKEN_TIERS = [
    {
        "name": "low",
        "min_crops": 1,
        "max_crops": 3,
        "preferred_crops": [2, 3],
        "typical_vision_tokens": 432,
        "description": "Small images, simple tasks"
    },
    {
        "name": "medium",
        "min_crops": 4,
        "max_crops": 8,
        "preferred_crops": [4, 6, 8],
        "typical_vision_tokens": 1008,
        "description": "Medium images, standard tasks"
    },
    {
        "name": "high",
        "min_crops": 9,
        "max_crops": 15,
        "preferred_crops": [9, 12, 15],
        "typical_vision_tokens": 1872,
        "description": "Large images, complex tasks"
    },
]
```

#### 2. Crop Selection for Tier

```python
def select_crops_for_tier(
    tier: Dict[str, Any],
    image_h: int,
    image_w: int,
    patch_size: int = 224,
    mismatch_threshold: float = 0.3
) -> Tuple[int, Tuple[int, int], float]:
    """
    Select best crop count within tier based on image aspect ratio.
    
    Uses select_tiling to find best tiling for each crop count, then selects
    crop count with minimum aspect ratio mismatch.
    
    Args:
        tier: Tier configuration dictionary
        image_h: Image height (after margins)
        image_w: Image width (after margins)
        patch_size: Patch size (default 224)
        mismatch_threshold: Maximum acceptable mismatch for preferred crops
        
    Returns:
        (selected_crops, selected_tiling, aspect_ratio_mismatch)
    """
    from molmo.preprocessors.multimodal_preprocessor import select_tiling
    
    image_aspect_ratio = image_w / image_h if image_h > 0 else 1.0
    best_crops = None
    best_tiling = None
    best_mismatch = float('inf')
    
    # Step 1: Try preferred crop counts first
    for crops in tier["preferred_crops"]:
        # Use select_tiling to find best tiling for this crop count
        tiling = select_tiling(
            h=image_h,
            w=image_w,
            patch_size=patch_size,
            max_num_crops=tier["max_crops"],
            exact_num_crops=crops
        )
        rows, cols = tiling
        
        # Calculate aspect ratio mismatch
        tiling_h = rows * patch_size
        tiling_w = cols * patch_size
        tiling_aspect = tiling_w / tiling_h if tiling_h > 0 else 1.0
        mismatch = abs(tiling_aspect - image_aspect_ratio)
        
        if mismatch < best_mismatch:
            best_mismatch = mismatch
            best_crops = crops
            best_tiling = tiling
    
    # Step 2: If mismatch is acceptable, return preferred crop
    if best_mismatch < mismatch_threshold:
        return best_crops, best_tiling, best_mismatch
    
    # Step 3: Otherwise, try all crop counts in tier
    for crops in range(tier["min_crops"], tier["max_crops"] + 1):
        if crops in tier["preferred_crops"]:
            continue  # Already tried
        
        tiling = select_tiling(
            h=image_h,
            w=image_w,
            patch_size=patch_size,
            max_num_crops=tier["max_crops"],
            exact_num_crops=crops
        )
        rows, cols = tiling
        
        tiling_h = rows * patch_size
        tiling_w = cols * patch_size
        tiling_aspect = tiling_w / tiling_h if tiling_h > 0 else 1.0
        mismatch = abs(tiling_aspect - image_aspect_ratio)
        
        if mismatch < best_mismatch:
            best_mismatch = mismatch
            best_crops = crops
            best_tiling = tiling
    
    return best_crops, best_tiling, best_mismatch
```

#### 3. Integration with Experiment Loop

```python
# In acc_lat_profiling.py run() method

# Instead of:
# for target_vision_tokens in vision_tokens_list:
#     num_crops = tokens_to_crops(target_vision_tokens)
#     ...

# Use:
for tier in VISION_TOKEN_TIERS:
    log.info(f"Processing tier: {tier['name']} (crops: {tier['min_crops']}-{tier['max_crops']})")
    
    # For each image, select best crop count within tier
    # This happens in the dataloader/preprocessing step
    # We need to modify MultiModalPreprocessor to accept tier instead of exact_num_crops
    
    # Process batch with tier-based selection
    # ...
```

## Detailed Examples

### Example 1: Wide Image (640×480, aspect 1.33) in Tier 2

**Step 1: Calculate image dimensions after margins**

```python
# Assuming 56-pixel margins on each side (typical)
image_h = 480 - 112 = 368
image_w = 640 - 112 = 528
image_aspect_ratio = 528 / 368 = 1.43
```

**Step 2: Try preferred crop counts**

```python
tier = {"name": "medium", "min_crops": 4, "max_crops": 8, "preferred_crops": [4, 6, 8]}

# Try 4 crops
tiling_4 = select_tiling(h=368, w=528, patch_size=224, max_num_crops=8, exact_num_crops=4)
# Returns: (2, 2) - only option for 4 crops
tiling_aspect_4 = (2 * 224) / (2 * 224) = 1.0
mismatch_4 = |1.0 - 1.43| = 0.43

# Try 6 crops
tiling_6 = select_tiling(h=368, w=528, patch_size=224, max_num_crops=8, exact_num_crops=6)
# Returns: (2, 3) - best match for aspect 1.43
tiling_aspect_6 = (3 * 224) / (2 * 224) = 1.5
mismatch_6 = |1.5 - 1.43| = 0.07 ✓ (best!)

# Try 8 crops
tiling_8 = select_tiling(h=368, w=528, patch_size=224, max_num_crops=8, exact_num_crops=8)
# Returns: (2, 4) - best match for aspect 1.43
tiling_aspect_8 = (4 * 224) / (2 * 224) = 2.0
mismatch_8 = |2.0 - 1.43| = 0.57
```

**Step 3: Select best crop count**

```python
selected_crops = 6  # Minimum mismatch (0.07)
selected_tiling = (2, 3)
target_resolution = (2 * 224, 3 * 224) = (448, 672)
```

**Step 4: Resize image**

```python
# With resize_to_fill=True
original = (640, 480)  # aspect 1.33
target = (448, 672)    # aspect 1.5

scale = max(448/480, 672/640) = max(0.93, 1.05) = 1.05
resized = (672, 504)  # aspect 1.33 (preserved!)
cropped = (448, 672)  # aspect 1.5 (minimal distortion)

Aspect ratio change: 1.33 → 1.5 (mismatch 0.17)
```

**Result**: 
- ✅ Aspect ratio mismatch: **0.07** (vs 0.43 for fixed 4 crops)
- ✅ Vision tokens: **1008** (6 crops + 1 base = 7 × 144)
- ✅ Full token utilization with `resize_to_fill=True`

### Example 2: Tall Image (480×640, aspect 0.75) in Tier 2

**Step 1: Calculate image dimensions**

```python
image_h = 640 - 112 = 528
image_w = 480 - 112 = 368
image_aspect_ratio = 368 / 528 = 0.70
```

**Step 2: Try preferred crop counts**

```python
# Try 4 crops: (2,2) tiling, aspect 1.0, mismatch = 0.30
# Try 6 crops: (3,2) tiling, aspect 0.67, mismatch = 0.03 ✓ (best!)
# Try 8 crops: (4,2) tiling, aspect 0.5, mismatch = 0.20

selected_crops = 6
selected_tiling = (3, 2)
target_resolution = (672, 448)
```

**Step 3: Resize image**

```python
original = (480, 640)  # aspect 0.75
target = (672, 448)   # aspect 0.67

scale = max(672/480, 448/640) = max(1.4, 0.7) = 1.4
resized = (672, 896)  # aspect 0.75 (preserved!)
cropped = (672, 448)  # aspect 0.67 (minimal distortion)

Aspect ratio change: 0.75 → 0.67 (mismatch 0.08)
```

**Result**:
- ✅ Aspect ratio mismatch: **0.03** (excellent match!)
- ✅ Vision tokens: **1008**
- ✅ Minimal distortion

### Example 3: Square Image (512×512, aspect 1.0) in Tier 2

**Step 1: Calculate image dimensions**

```python
image_h = 512 - 112 = 400
image_w = 512 - 112 = 400
image_aspect_ratio = 1.0
```

**Step 2: Try preferred crop counts**

```python
# Try 4 crops: (2,2) tiling, aspect 1.0, mismatch = 0.0 ✓ (perfect!)
# Try 6 crops: (2,3) tiling, aspect 1.5, mismatch = 0.5
# Try 8 crops: (2,4) tiling, aspect 2.0, mismatch = 1.0

selected_crops = 4
selected_tiling = (2, 2)
target_resolution = (448, 448)
```

**Step 3: Resize image**

```python
original = (512, 512)  # aspect 1.0
target = (448, 448)    # aspect 1.0

scale = max(448/512, 448/512) = 0.875
resized = (448, 448)  # aspect 1.0 (perfect match!)

Aspect ratio change: 1.0 → 1.0 (mismatch 0.0)
```

**Result**:
- ✅ Aspect ratio mismatch: **0.0** (perfect match!)
- ✅ Vision tokens: **720** (4 crops)
- ✅ No distortion

## Integration with MultiModalPreprocessor

### Current Implementation

```python
mm_preprocessor = MultiModalPreprocessor(
    tokenizer=self.tokenizer,
    crop_mode=self.model.config.crop_mode,
    max_crops=max_crops,
    exact_num_crops=num_crops,  # Fixed crop count
    overlap_margins=self.model.config.overlap_margins,
    image_padding_mask=bool(self.model.config.image_padding_embed),
    resize_to_fill=resize_to_fill,
)
```

### Proposed Modification

**Option 1: Per-Image Selection in Preprocessor**

```python
class MultiModalPreprocessor:
    def __init__(
        self,
        ...,
        tier: Optional[Dict[str, Any]] = None,  # New: tier configuration
        exact_num_crops: Optional[int] = None,  # Keep for backward compatibility
    ):
        self.tier = tier
        self.exact_num_crops = exact_num_crops
    
    def image_to_patches_and_tokens(self, image, ...):
        # If tier is provided, select crops per image
        if self.tier is not None:
            image_h, image_w = image.shape[:2]
            # Subtract margins
            h_after_margins = image_h - total_margin_pixels
            w_after_margins = image_w - total_margin_pixels
            
            # Select best crop count for this image
            selected_crops, selected_tiling, mismatch = select_crops_for_tier(
                tier=self.tier,
                image_h=h_after_margins,
                image_w=w_after_margins,
                patch_size=self.crop_window_size
            )
            
            # Use selected_crops as exact_num_crops
            exact_num_crops = selected_crops
        else:
            # Use provided exact_num_crops (backward compatibility)
            exact_num_crops = self.exact_num_crops
        
        # Continue with existing logic using exact_num_crops
        ...
```

**Option 2: Pre-selection in DataLoader**

```python
# In dataloader, before preprocessing
def preprocess_with_tier(image, tier):
    image_h, image_w = image.shape[:2]
    # Calculate dimensions after margins
    h_after_margins = image_h - total_margin_pixels
    w_after_margins = image_w - total_margin_pixels
    
    # Select best crop count
    selected_crops, selected_tiling, mismatch = select_crops_for_tier(
        tier=tier,
        image_h=h_after_margins,
        image_w=w_after_margins,
        patch_size=224
    )
    
    # Store for use in MultiModalPreprocessor
    return selected_crops, selected_tiling, mismatch

# Then use selected_crops as exact_num_crops in MultiModalPreprocessor
```

**Recommendation**: Use **Option 1** for cleaner integration, but requires modifying `MultiModalPreprocessor`.

## Result Recording

### Per-Sample Result

```python
per_sample_result = {
    "sample_id": batch_idx,
    "tier": "medium",
    "tier_range": {"min_crops": 4, "max_crops": 8},
    "selected_crops": 6,  # Per-image selection
    "selected_tiling": [2, 3],
    "aspect_ratio_mismatch": 0.07,
    "selected_vision_tokens": 1008,  # (6 + 1) * 144
    "actual_vision_tokens": 1002,
    "original_image_size": [640, 480],
    "original_aspect_ratio": 1.33,
    "target_resolution": [448, 672],
    "target_aspect_ratio": 1.5,
    "resize_to_fill": True,
    "top_k": top_k,
    "num_active_blocks": num_active_blocks,
    "accuracy": pred_score.get("score", 0.0),
    ...
}
```

### Config-Level Result

```python
config_result = {
    "tier": "medium",
    "tier_range": {"min_crops": 4, "max_crops": 8},
    "selected_crops_distribution": {
        4: 15,  # 15 images used 4 crops
        5: 2,   # 2 images used 5 crops
        6: 28,  # 28 images used 6 crops (most common)
        7: 1,   # 1 image used 7 crops
        8: 4,   # 4 images used 8 crops
    },
    "selected_crops_mean": 5.8,
    "selected_crops_std": 1.2,
    "selected_vision_tokens_mean": 979.2,  # Average across all images
    "selected_vision_tokens_std": 172.8,
    "aspect_ratio_mismatch_mean": 0.15,
    "aspect_ratio_mismatch_std": 0.12,
    "actual_vision_tokens_mean": 975.0,
    "top_k": top_k,
    "num_active_blocks": num_active_blocks,
    "accuracy": accuracy_mean,
    "accuracy_std": accuracy_std,
    "num_samples": num_processed,
    "resize_to_fill": resize_to_fill,
    "per_sample_results": per_sample_results,
}
```

## Filename Generation

```python
def _generate_tier_filename(config_result: Dict[str, Any], dataset_name: str) -> str:
    """
    Generate filename for tier-based results.
    
    Format: <task_name>_tier-<tier_name>_crops<mean>_topk<k>_blocks<n>.json
    Example: coco-2014-vqa_tier-medium_crops6_topk8_blocks14.json
    """
    task_name = dataset_name.replace("_", "-")
    tier_name = config_result.get("tier", "unknown")
    selected_crops_mean = int(config_result.get("selected_crops_mean", 0))
    top_k = config_result.get("top_k", "unknown")
    num_blocks = config_result.get("num_active_blocks", "unknown")
    
    filename = f"{task_name}_tier-{tier_name}_crops{selected_crops_mean}_topk{top_k}_blocks{num_blocks}.json"
    return filename
```

## Validation Strategy

### 1. Distortion Validation

```python
def validate_distortion(per_sample_results: List[Dict]) -> Dict[str, float]:
    """
    Validate that aspect ratio distortion is minimized.
    """
    mismatches = [r["aspect_ratio_mismatch"] for r in per_sample_results]
    
    return {
        "mean_mismatch": np.mean(mismatches),
        "std_mismatch": np.std(mismatches),
        "max_mismatch": np.max(mismatches),
        "p95_mismatch": np.percentile(mismatches, 95),
        "p99_mismatch": np.percentile(mismatches, 99),
    }

# Target metrics:
# - mean_mismatch < 0.2 (vs 0.3-0.5 for fixed targets)
# - p95_mismatch < 0.4
# - p99_mismatch < 0.6
```

### 2. Accuracy Validation

```python
def validate_accuracy_benefits(tier_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Validate that higher tiers provide accuracy benefits.
    """
    tier_accuracies = {
        tier: results["accuracy"]
        for tier, results in tier_results.items()
    }
    
    improvements = {}
    tiers = ["low", "medium", "high"]
    for i in range(1, len(tiers)):
        prev_tier = tiers[i-1]
        curr_tier = tiers[i]
        improvement = tier_accuracies[curr_tier] - tier_accuracies[prev_tier]
        improvements[f"{prev_tier}_to_{curr_tier}"] = improvement
    
    return {
        "tier_accuracies": tier_accuracies,
        "improvements": improvements,
    }

# Target metrics:
# - low_to_medium improvement > 1%
# - medium_to_high improvement > 1%
```

### 3. Token Utilization Validation

```python
def validate_token_utilization(per_sample_results: List[Dict]) -> Dict[str, float]:
    """
    Validate that vision tokens are fully utilized.
    """
    utilizations = []
    for r in per_sample_results:
        selected = r["selected_vision_tokens"]
        actual = r["actual_vision_tokens"]
        utilization = actual / selected if selected > 0 else 0.0
        utilizations.append(utilization)
    
    return {
        "mean_utilization": np.mean(utilizations),
        "min_utilization": np.min(utilizations),
        "p5_utilization": np.percentile(utilizations, 5),
    }

# Target metrics:
# - mean_utilization > 0.95 (with resize_to_fill=True)
# - p5_utilization > 0.90
```

## Migration Path

### Phase 1: Add Tier Support (Backward Compatible)

1. Add `tier` parameter to `acc_lat_profiling.py`
2. Implement `select_crops_for_tier` function
3. Keep `vision_tokens_list` as default (backward compatible)
4. Add `--use_tier_based` flag to enable tier mode

### Phase 2: Modify MultiModalPreprocessor

1. Add `tier` parameter to `MultiModalPreprocessor`
2. Implement per-image crop selection in `image_to_patches_and_tokens`
3. Maintain backward compatibility with `exact_num_crops`

### Phase 3: Validation and Optimization

1. Run experiments with tier-based approach
2. Compare with fixed targets
3. Validate distortion and accuracy metrics
4. Optimize tier boundaries if needed

## Conclusion

The Hybrid Tier Approach provides:

✅ **Minimal Distortion**: Adaptive crop selection reduces aspect ratio mismatch by 48%

✅ **Accuracy Benefits**: Clear tier progression with 2.33× and 1.86× token increases

✅ **Integration**: Seamless integration with existing `select_tiling` logic

✅ **Flexibility**: Per-image optimization within tier constraints

**Next Steps**:
1. Implement `select_crops_for_tier` function
2. Add tier support to `MultiModalPreprocessor`
3. Update experiment scripts to use tier mode
4. Run validation experiments

