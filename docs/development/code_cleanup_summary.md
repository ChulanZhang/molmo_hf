# Code Cleanup Summary: Duplicate Function Removal

## Overview

This document summarizes the code cleanup performed to remove duplicate `select_tiling` and `resize_and_pad` functions from the codebase.

## Problem Identified

Two duplicate implementations of `select_tiling` and `resize_and_pad` were found:

1. **`molmo/data/model_preprocessor.py`** (New version, actively used)
   - `select_tiling(h, w, patch_size, max_num_crops, exact_num_crops=None)` - Supports `exact_num_crops` parameter
   - `resize_and_pad(image, desired_output_size, is_training=False, resize_method="torch-bilinear", pad_value=0, rng=np.random)` - More flexible signature
   - Used by `MultiModalPreprocessor` (main data processing pipeline)

2. **`molmo/preprocessors/image_preprocessing_molmo.py`** (Merged into `multimodal_preprocessor.py`)
   - `select_tiling(h, w, patch_size, max_num_patches)` - No `exact_num_crops` support
   - `resize_and_pad(image, desired_output_size, resize_method="torch-bilinear", pad_value=0, normalize=True, ...)` - Different signature with `normalize` parameter
   - Used by `MolmoImageProcessor` (used in some profiling experiments)

## Solution Implemented

### 1. Removed Duplicate Functions

- Deleted `select_tiling` from `image_preprocessing_molmo.py`
- Deleted `resize_and_pad` from `image_preprocessing_molmo.py`

### 2. Imported Shared Functions

Updated `image_preprocessing_molmo.py` to import from `model_preprocessor.py`:

```python
# Import shared functions from model_preprocessor to avoid code duplication
from molmo.preprocessors.multimodal_preprocessor import select_tiling, resize_and_pad
```

### 3. Created Compatibility Wrapper

Created `_resize_and_pad_with_normalize` wrapper function to maintain backward compatibility with `MolmoImageProcessor`:

```python
def _resize_and_pad_with_normalize(
    image,
    desired_output_size,
    resize_method="torch-bilinear",
    pad_value=0,
    normalize=True,
    image_mean=OPENAI_CLIP_MEAN,
    image_std=OPENAI_CLIP_STD,
):
    """
    Wrapper around resize_and_pad from model_preprocessor to support normalize parameter.
    This maintains backward compatibility with MolmoImageProcessor.
    """
    # Call the shared resize_and_pad (without normalize)
    image, image_mask = resize_and_pad(
        image=image,
        desired_output_size=desired_output_size,
        is_training=False,
        resize_method=resize_method,
        pad_value=pad_value,
        rng=np.random
    )
    
    # Apply normalization if requested
    if normalize:
        image = normalize_image(image, offset=image_mean, scale=image_std)
    
    return image, image_mask
```

### 4. Updated Function Calls

Updated `MolmoImageProcessor.image_to_patches_and_tokens` to:
- Use shared `select_tiling` with correct signature
- Use `_resize_and_pad_with_normalize` wrapper for backward compatibility

### 5. Fixed Bug in `model_preprocessor.py`

Fixed a syntax error in `select_tiling`:
```python
# Before (bug):
required_scale_d = candidate_resolutions.astype(np.float32) / original_size,

# After (fixed):
required_scale_d = candidate_resolutions.astype(np.float32) / original_size
```

## Code Structure After Cleanup

### Shared Functions (Single Source of Truth)

**`molmo/data/model_preprocessor.py`**:
- `select_tiling(h, w, patch_size, max_num_crops, exact_num_crops=None)` - Main implementation
- `resize_and_pad(image, desired_output_size, is_training=False, ...)` - Main implementation
- `resize_and_crop_to_fill(image, desired_output_size, ...)` - Unique to this file
- `pixels_to_patches(array, patch_size)` - Unique to this file
- `batch_pixels_to_patches(array, patch_size)` - Unique to this file

**`molmo/preprocessors/multimodal_preprocessor.py`** (now contains `MolmoImageProcessor`):
- Imports `select_tiling` and `resize_and_pad` from `model_preprocessor`
- `_resize_and_pad_with_normalize(...)` - Compatibility wrapper
- `pad_to_bounding_box(...)` - Unique utility function
- `normalize_image(...)` - Unique utility function

## Usage Analysis

### Primary Data Pipeline (Most Experiments)

**`MultiModalPreprocessor`** (`molmo/data/model_preprocessor.py`):
- Used by all main experiments (`acc_lat_profiling.py`, `exp5_accuracy.py`, etc.)
- Directly uses shared functions
- Supports `exact_num_crops` for precise vision token control
- Supports `resize_to_fill` for better token utilization

### Legacy Pipeline (Some Profiling Experiments)

**`MolmoImageProcessor`** (`molmo/preprocessors/multimodal_preprocessor.py`):
- Used by some profiling experiments (`exp_context_scaling.py`, `exp_moe_topk.py`, etc.)
- Uses shared functions via compatibility wrapper
- Maintains backward compatibility with existing code

## Benefits

1. **Single Source of Truth**: All image processing logic is centralized in `model_preprocessor.py`
2. **Reduced Code Duplication**: Eliminated ~100 lines of duplicate code
3. **Easier Maintenance**: Future improvements only need to be made in one place
4. **Backward Compatibility**: Legacy code continues to work via compatibility wrapper
5. **Bug Fixes**: Fixed syntax error in `select_tiling` that could cause issues

## Testing Recommendations

1. **Test Main Experiments**: Verify `acc_lat_profiling.py` still works correctly
2. **Test Legacy Experiments**: Verify profiling experiments using `MolmoImageProcessor` still work
3. **Test Vision Token Control**: Verify `exact_num_crops` functionality is preserved
4. **Test Resize Functions**: Verify both `resize_and_pad` and `resize_and_crop_to_fill` work correctly

## Files Modified

1. `molmo/preprocessors/multimodal_preprocessor.py` (contains both `MultiModalPreprocessor` and `MolmoImageProcessor`)
   - Removed duplicate `select_tiling` function
   - Removed duplicate `resize_and_pad` function
   - Added import from `model_preprocessor`
   - Added `_resize_and_pad_with_normalize` wrapper
   - Updated function calls to use shared functions

2. `molmo/data/model_preprocessor.py`
   - Fixed syntax error in `select_tiling` (removed trailing comma)

## Notes

- `pixels_to_patches` and `batch_pixels_to_patches` are unique to `model_preprocessor.py` and not duplicated
- `resize_and_crop_to_fill` is unique to `model_preprocessor.py` and not duplicated
- All other image processing functions are unique to their respective files

