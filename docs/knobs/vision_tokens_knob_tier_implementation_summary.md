# Vision Tokens Control Knob: Tier-Based Implementation Summary

## Overview

The tier-based approach has been integrated directly into `select_tiling` function, replacing the `exact_num_crops` parameter. This provides a cleaner, more flexible design that allows adaptive crop selection within tier ranges while maintaining backward compatibility.

## Implementation Details

### 1. Modified `select_tiling` Function

**Location**: `molmo/preprocessors/multimodal_preprocessor.py`

**Changes**:
- **Removed**: `exact_num_crops` parameter
- **Added**: `tier` parameter (Optional[Dict[str, Any]])

**New Signature**:
```python
def select_tiling(h, w, patch_size, max_num_crops, tier=None):
    """
    Select optimal tiling configuration for an image.
    
    Args:
        h: Image height (after subtracting margins)
        w: Image width (after subtracting margins)
        patch_size: Crop window size (e.g., 224)
        max_num_crops: Maximum number of crops allowed
        tier: Optional dict with keys:
            - min_crops: Minimum number of crops (default: 1)
            - max_crops: Maximum number of crops (default: max_num_crops)
            - preferred_crops: List of preferred crop counts to try first
            - mismatch_threshold: Maximum acceptable mismatch (default: 0.3)
            If provided, selects best crop count within tier range based on aspect ratio.
            If None, uses adaptive selection based on image size.
    
    Returns:
        (rows, cols) tiling configuration
    """
```

**Logic Flow**:
1. **If `tier` is provided**:
   - Try preferred crop counts first
   - For each crop count, find all possible tilings
   - Select tiling with minimum aspect ratio mismatch
   - If mismatch < threshold, return immediately
   - Otherwise, try all crop counts in tier range
   - Return best tiling found

2. **If `tier` is None**:
   - Use original adaptive selection logic
   - Selects tiling based on image size and minimal upscaling

### 2. Modified `MultiModalPreprocessor` Class

**Location**: `molmo/preprocessors/multimodal_preprocessor.py`

**Changes**:
- **Removed**: `exact_num_crops` field
- **Added**: `tier` field (Optional[Dict[str, Any]])

**Updated Field**:
```python
@dataclasses.dataclass
class MultiModalPreprocessor:
    # ...
    max_crops: int = 12
    tier: Optional[Dict[str, Any]] = None  # Tier configuration for adaptive crop selection
    # ...
```

**Updated Call**:
```python
tiling = select_tiling(
    original_image_h - total_margin_pixels,
    original_image_w - total_margin_pixels,
    crop_window_size,
    max_crops,
    tier=self.tier  # Pass tier instead of exact_num_crops
)
```

### 3. Updated Experiment Script

**Location**: `experiments/core_exp/acc_lat_profiling.py`

**Changes**:
- Removed `exact_num_crops=num_crops` parameter
- Added `tier=None` parameter (for future tier-based experiments)

**Current Usage**:
```python
mm_preprocessor = MultiModalPreprocessor(
    tokenizer=self.tokenizer,
    crop_mode=self.model.config.crop_mode,
    max_crops=max_crops,  # Set to num_crops to approximate exact crop count
    tier=None,  # For now, use adaptive selection (can be changed to tier-based later)
    # ...
)
```

## Tier Configuration Format

When using tier-based selection, pass a dictionary with the following structure:

```python
tier = {
    "min_crops": 4,           # Minimum number of crops
    "max_crops": 8,            # Maximum number of crops
    "preferred_crops": [4, 6, 8],  # Preferred crop counts to try first
    "mismatch_threshold": 0.3,     # Maximum acceptable mismatch (default: 0.3)
}
```

**Example Tier Definitions**:
```python
TIER_LOW = {
    "min_crops": 1,
    "max_crops": 3,
    "preferred_crops": [2, 3],
    "mismatch_threshold": 0.3,
}

TIER_MEDIUM = {
    "min_crops": 4,
    "max_crops": 8,
    "preferred_crops": [4, 6, 8],
    "mismatch_threshold": 0.3,
}

TIER_HIGH = {
    "min_crops": 9,
    "max_crops": 15,
    "preferred_crops": [9, 12, 15],
    "mismatch_threshold": 0.3,
}
```

## Benefits of This Design

### 1. **Cleaner API**
- Single parameter (`tier`) instead of separate `exact_num_crops`
- More flexible: allows range-based selection instead of fixed count
- Better semantics: tier represents a "quality level" rather than exact count

### 2. **Better Aspect Ratio Matching**
- Adaptive selection within tier range
- Prioritizes preferred crop counts (which have better tiling options)
- Falls back to other crop counts if mismatch is too high

### 3. **Backward Compatible**
- When `tier=None`, uses original adaptive selection logic
- Existing code continues to work without changes
- Gradual migration path: can add tier support incrementally

### 4. **Centralized Logic**
- All tiling selection logic in one place (`select_tiling`)
- No need for separate `select_crops_for_tier` function
- Easier to maintain and test

## Usage Examples

### Example 1: Fixed Crop Count (Current Approach)

```python
# Current: approximate exact crop count by setting max_crops
mm_preprocessor = MultiModalPreprocessor(
    max_crops=6,  # Will select tiling with ~6 crops
    tier=None,    # Use adaptive selection
)
```

### Example 2: Tier-Based Selection (Future)

```python
# Future: use tier-based selection
tier_medium = {
    "min_crops": 4,
    "max_crops": 8,
    "preferred_crops": [4, 6, 8],
}

mm_preprocessor = MultiModalPreprocessor(
    max_crops=12,  # Upper bound
    tier=tier_medium,  # Select best crop count within tier
)
```

### Example 3: Per-Image Tier Selection

```python
# For each image, select appropriate tier based on image characteristics
def select_tier_for_image(image):
    h, w = image.shape[:2]
    if h * w < 200000:  # Small image
        return TIER_LOW
    elif h * w < 500000:  # Medium image
        return TIER_MEDIUM
    else:  # Large image
        return TIER_HIGH

# In preprocessing loop:
tier = select_tier_for_image(image)
mm_preprocessor = MultiModalPreprocessor(
    max_crops=15,
    tier=tier,
)
```

## Migration Path

### Phase 1: Current (Completed)
- ✅ Removed `exact_num_crops` from `select_tiling`
- ✅ Added `tier` parameter to `select_tiling`
- ✅ Updated `MultiModalPreprocessor` to use `tier`
- ✅ Updated experiment scripts to remove `exact_num_crops`

### Phase 2: Future (To Be Implemented)
- [ ] Add tier configuration to experiment scripts
- [ ] Implement tier-based experiments
- [ ] Validate tier-based selection vs fixed targets
- [ ] Update documentation with tier-based results

## Testing

To test tier-based selection:

```python
from molmo.preprocessors.multimodal_preprocessor import select_tiling

# Test tier-based selection
tier = {
    "min_crops": 4,
    "max_crops": 8,
    "preferred_crops": [4, 6, 8],
}

# Wide image (640×480, aspect 1.33)
tiling = select_tiling(
    h=480,
    w=640,
    patch_size=224,
    max_num_crops=12,
    tier=tier
)
# Expected: (2, 3) tiling (6 crops, aspect 1.5, mismatch 0.17)

# Square image (512×512, aspect 1.0)
tiling = select_tiling(
    h=512,
    w=512,
    patch_size=224,
    max_num_crops=12,
    tier=tier
)
# Expected: (2, 2) tiling (4 crops, aspect 1.0, mismatch 0.0)
```

## Conclusion

The tier-based approach is now integrated into `select_tiling`, providing a clean, flexible API for adaptive crop selection. The implementation maintains backward compatibility while enabling future tier-based experiments.

