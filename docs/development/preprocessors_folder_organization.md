# Preprocessors Folder Organization

## Current Structure

```
molmo/preprocessors/
├── __init__.py
├── multimodal_preprocessor.py      (Core preprocessing - moved from data/)
├── preprocessing_molmo.py          (HuggingFace-compatible processor)
└── multimodal_preprocessor.py    (Contains both MultiModalPreprocessor and MolmoImageProcessor)
```

## File Responsibilities

### 1. `multimodal_preprocessor.py` (Core Implementation)
**Purpose**: Core data preprocessing implementation
- `MultiModalPreprocessor` - Main preprocessing class (image + text → tensors)
- `Preprocessor` - Wrapper combining `DataFormatter` + `MultiModalPreprocessor`
- Utility functions: `select_tiling`, `resize_and_pad`, `resize_and_crop_to_fill`, `pixels_to_patches`, `load_image`, `setup_pil`
- **Used by**: All main experiments, training code, evaluation code

### 2. `preprocessing_molmo.py` (HuggingFace Compatibility Layer)
**Purpose**: HuggingFace-compatible processor for HF API compatibility
- `MolmoProcessor` - HuggingFace `ProcessorMixin` compatible
- `MolmoTextKwargs`, `MolmoProcessorKwargs` - Type hints for HF API
- **Used by**: Some profiling experiments, HF Hub compatibility

### 3. `multimodal_preprocessor.py` (Core Preprocessing + HuggingFace Compatibility)
**Purpose**: HuggingFace-compatible image processor for HF API compatibility
- `MolmoImageProcessor` - HuggingFace `BaseImageProcessor` compatible
- `MolmoImagesKwargs` - Type hints for HF API
- Uses functions from `multimodal_preprocessor.py` (via import)
- **Used by**: `MolmoProcessor`, some profiling experiments

## Relationship Between Files

```
┌─────────────────────────────────────┐
│  multimodal_preprocessor.py          │
│  (Core Implementation)               │
│  - MultiModalPreprocessor            │
│  - Preprocessor                      │
│  - Utility functions                 │
└──────────────┬──────────────────────┘
               │
               │ (all in same file)
               │
┌──────────────▼──────────────────────┐
│  multimodal_preprocessor.py          │
│  (Core + HF Compatibility)           │
│  - MultiModalPreprocessor            │
│  - Preprocessor                       │
│  - MolmoImageProcessor               │
│  - Utility functions                 │
└──────────────┬──────────────────────┘
               │
               │ uses
               │
┌──────────────▼──────────────────────┐
│  preprocessing_molmo.py               │
│  (HF Compatibility Layer)             │
│  - MolmoProcessor                    │
│  - Uses: MolmoImageProcessor         │
└──────────────────────────────────────┘
```

## Organization Rationale

### Current Organization is Good ✅

1. **Clear Separation of Concerns**:
   - Core implementation (`multimodal_preprocessor.py`) - single source of truth
   - HF compatibility layers (`preprocessing_molmo.py`, `MolmoImageProcessor` in `multimodal_preprocessor.py`) - thin wrappers

2. **Dependency Flow**:
   - Core → HF compatibility (one-way dependency)
   - No circular dependencies

3. **File Naming**:
   - `multimodal_preprocessor.py` - Clear name indicating core functionality
   - `preprocessing_molmo.py` - HF-compatible processor
   - `MolmoImageProcessor` (in `multimodal_preprocessor.py`) - HF-compatible image processor

## Potential Improvements

### Option 1: Keep Current Structure (Recommended) ✅
**Pros**:
- Clear and simple
- Easy to understand
- No unnecessary complexity

**Cons**:
- None significant

### Option 2: Subfolder Organization
```
preprocessors/
├── __init__.py
├── core/
│   └── multimodal_preprocessor.py
└── hf_compat/
    ├── preprocessing_molmo.py
    └── multimodal_preprocessor.py (contains both MultiModalPreprocessor and MolmoImageProcessor)
```
**Pros**:
- More explicit separation
- Could add more HF-compatible processors later

**Cons**:
- More complex import paths
- May be over-engineering for current needs

### Option 3: Rename for Clarity
- `multimodal_preprocessor.py` → `core_preprocessor.py` or `base_preprocessor.py`
- Keep HF compatibility files as-is

**Recommendation**: Keep current structure (Option 1)

## Import Patterns

### Recommended Import Patterns

**For Core Functionality** (Most experiments):
```python
from molmo.preprocessors.multimodal_preprocessor import (
    MultiModalPreprocessor,
    Preprocessor,
    select_tiling,
    resize_and_pad,
)
```

**For HuggingFace Compatibility** (Some profiling experiments):
```python
from molmo.preprocessors.preprocessing_molmo import MolmoProcessor
from molmo.preprocessors.multimodal_preprocessor import MolmoImageProcessor
```

**Via __init__.py** (Convenience):
```python
from molmo.preprocessors import (
    MultiModalPreprocessor,
    Preprocessor,
    MolmoProcessor,
    MolmoImageProcessor,
)
```

## Summary

The current organization is **well-structured and logical**:
- ✅ Core implementation separate from compatibility layers
- ✅ Clear dependency flow
- ✅ Appropriate file naming
- ✅ Easy to maintain and extend

**No reorganization needed** - the current structure is optimal for the codebase's needs.

