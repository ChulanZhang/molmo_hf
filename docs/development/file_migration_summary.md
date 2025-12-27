# File Migration Summary: model_preprocessor.py → preprocessors/

## Migration Completed ✅

### File Moved
- **From**: `molmo/data/model_preprocessor.py`
- **To**: `molmo/preprocessors/multimodal_preprocessor.py`
- **Reason**: Better organization - preprocessing code belongs in `preprocessors/` folder, not `data/` folder

### Files Updated (20+ files)

#### Core Files
1. `molmo/data/__init__.py` - Updated import path
2. `molmo/preprocessors/__init__.py` - Added exports for new location
3. `molmo/preprocessors/image_preprocessing_molmo.py` - Merged into `multimodal_preprocessor.py` and deleted
4. `molmo/data/download_urls.py` - Updated import for `setup_pil`

#### Experiment Files (15 files)
5. `experiments/base_experiment.py`
6. `experiments/motivate/base_experiment.py`
7. `experiments/core_exp/acc_lat_profiling.py`
8. `experiments/controller/data_preparation.py`
9. `experiments/profiling/knob5_combined/exp5_accuracy.py`
10. `experiments/profiling/knob5_combined/exp6_accuracy.py`
11. `experiments/profiling/knob2_topk/exp2_accuracy.py`
12. `experiments/profiling/knob1_tokens/exp1_accuracy.py`
13. `experiments/profiling/knob3_layers/exp3_accuracy.py`
14. `experiments/profiling/knob3_layers/exp3_accuracy_sensitivity.py`
15. `experiments/profiling/utils/analyze_vqa_max_crops.py`
16. `experiments/motivate/exp6_crop_overlap_analysis.py`

#### Test Files (2 files)
17. `tests/test_data.py`
18. `tests/test_data_functional.py`

#### Documentation Files (2 files)
19. `docs/development/code_cleanup_summary.md`
20. `docs/knobs/vision_tokens_knob_hybrid_tier_implementation.md`

## New File Organization

### `molmo/data/` (Dataset Management Only)
```
data/
├── __init__.py
├── academic_datasets.py          # Dataset classes
├── pixmo_datasets.py              # Dataset classes
├── dataset.py                     # Dataset base classes
├── data_formatter.py              # Text formatting
├── collator.py                    # Batch collation
├── iterable_dataset_mixture.py    # Dataset mixture
└── download_urls.py               # Download management
```
**All files are now dataset-related** ✅

### `molmo/preprocessors/` (Data Preprocessing)
```
preprocessors/
├── __init__.py
├── multimodal_preprocessor.py     # Core preprocessing (moved from data/)
├── preprocessing_molmo.py         # HuggingFace-compatible processor
└── multimodal_preprocessor.py    # Core preprocessing + HuggingFace-compatible image processor (merged)
```
**All files are preprocessing-related** ✅

## Import Changes

### Old Import Pattern
```python
from molmo.data.model_preprocessor import MultiModalPreprocessor, Preprocessor
```

### New Import Pattern
```python
from molmo.preprocessors.multimodal_preprocessor import MultiModalPreprocessor, Preprocessor
```

### Via __init__.py (Convenience)
```python
from molmo.preprocessors import MultiModalPreprocessor, Preprocessor
```

## Benefits

1. **Better Organization**: Preprocessing code is now in the preprocessing folder
2. **Clear Separation**: Dataset management (`data/`) vs. data preprocessing (`preprocessors/`)
3. **Logical Grouping**: All preprocessing logic in one place
4. **Easier Maintenance**: Clearer file structure makes it easier to find and modify code

## Verification

All imports have been updated. The migration is complete and the codebase is now better organized.

