# Documentation Migration Summary

## Overview

This document summarizes the documentation reorganization and consolidation completed in 2024. All documents have been unified in English, consolidated to remove duplicates, and organized around the three main control knobs.

## Three Control Knobs

The documentation is now organized around three main control knobs:

1. **Vision Tokens Knob** - Controls input size (number of vision tokens)
   - Document: `knobs/vision_tokens_knob.md`
   - Controls: number of crops, tilings, and image size
   - Formula: `Total Vision Tokens = (num_crops + 1) × 144`

2. **MoE Top-K Knob** - Controls model width (number of active experts)
   - Document: `knobs/moe_topk_knob.md`
   - Controls: number of active experts per token
   - Formula: `Active computation = (top_k / moe_num_experts) × Full model computation`

3. **Transformer Blocks Knob** - Controls model depth (number of active blocks)
   - Document: `knobs/transformer_blocks_knob.md`
   - Controls: which blocks to activate based on importance scores
   - Formula: `Relative Latency ≈ (active_blocks / total_blocks) × Full model latency`

## New Document Structure

```
docs/
├── README.md                          # Main index (updated)
├── MIGRATION_SUMMARY.md               # This document
├── knobs/                              # 🆕 Control knobs (core documentation)
│   ├── README.md                      # Quick reference
│   ├── vision_tokens_knob.md         # Vision tokens control (unified, English)
│   ├── moe_topk_knob.md              # MoE top-K control (unified, English)
│   └── transformer_blocks_knob.md    # Transformer blocks control (unified, English)
├── experiments/                        # Experiment guides (all English)
│   ├── motivation_experiments.md      # Motivation experiments guide
│   └── profiling_experiments.md       # Profiling experiments guide
├── mechanisms/                         # Code mechanisms (English)
│   ├── batch_size_optimization.md     # 🆕 Unified batch size optimization
│   ├── how_to_use_eos_token.md       # EOS token usage
│   ├── max_new_tokens_knob.md        # max_new_tokens parameter
│   └── model_inference_flow.md        # Complete inference pipeline
├── analysis/                           # Result analysis (empty, content moved to knobs)
├── development/                        # Development and refactoring
│   ├── code_reorganization_plan.md
│   └── profiling_code_analysis_summary.md
└── deprecated/                         # Archived old documents
    ├── README.md                      # Migration guide
    ├── image_resolution_vision_tokens_mapping.md
    ├── exp3_transformer_blocks_mask.md
    ├── exp6_crop_overlap_analysis.md
    ├── auto_batch_size_logic.md
    ├── dynamic_batch_size_guide.md
    └── max_crops_limits.md
```

## Consolidation Summary

### Documents Consolidated into `knobs/vision_tokens_knob.md`

- `image_resolution_vision_tokens_mapping.md` - Vision tokens calculation and mapping
- `exp6_crop_overlap_analysis.md` - Crop overlap analysis and control methods
- `max_crops_limits.md` - Limits and constraints for max_crops

**Result**: Single unified document covering:
- Vision token calculation formulas
- Tiling algorithm details
- Image resolution mapping
- Limits and constraints
- Complete implementation examples

### Documents Consolidated into `knobs/transformer_blocks_knob.md`

- `exp3_transformer_blocks_mask.md` - Block masking mechanism

**Enhancements added**:
- Importance score-based selection methods
- Accuracy drop importance calculation
- Activation magnitude importance
- Gradient-based importance
- Attention pattern importance

### Documents Consolidated into `mechanisms/batch_size_optimization.md`

- `auto_batch_size_logic.md` - Automatic batch size adjustment logic
- `dynamic_batch_size_guide.md` - Dynamic batch size usage guide

**Result**: Single unified document covering:
- Complete batch size optimization algorithm
- Estimation formulas for all three knobs
- Binary search strategy
- Usage examples for all scenarios
- Troubleshooting guide

## Key Improvements

### 1. Unified Content

- **Removed duplicates**: Eliminated overlapping content across multiple documents
- **Unified terminology**: Consistent terminology throughout all documents
- **Complete coverage**: Each knob document contains all related information

### 2. English Documentation

- **All new documents**: Written in English
- **Experiments guides**: Already in English (no translation needed)
- **Consistent style**: Professional technical documentation style

### 3. Three-Knob Organization

- **Clear structure**: Each knob has its own comprehensive document
- **Complete workflows**: From target values to implementation
- **Code examples**: Complete, runnable code examples for each knob

### 4. Importance Score Support

- **Transformer blocks knob**: Now includes multiple importance score calculation methods
- **Accuracy-based**: Most reliable method using accuracy drop
- **Flexible selection**: Multiple strategies for block selection

## Migration Path

### For Vision Tokens

**Old**: Multiple documents scattered across `mechanisms/` and `analysis/`
**New**: Single document `knobs/vision_tokens_knob.md`

**What to read**:
- Vision tokens control → `knobs/vision_tokens_knob.md`
- Limits and constraints → `knobs/vision_tokens_knob.md` (section "Limits and Constraints")

### For MoE Top-K

**Old**: Information scattered in experiment guides
**New**: Single document `knobs/moe_topk_knob.md`

**What to read**:
- MoE top-K control → `knobs/moe_topk_knob.md`
- Dynamic adjustment → `knobs/moe_topk_knob.md` (section "Dynamic Top-K Adjustment")

### For Transformer Blocks

**Old**: `exp3_transformer_blocks_mask.md` in `mechanisms/`
**New**: `knobs/transformer_blocks_knob.md` with importance scores

**What to read**:
- Transformer blocks control → `knobs/transformer_blocks_knob.md`
- Importance scores → `knobs/transformer_blocks_knob.md` (section "Importance Score Calculation")

### For Batch Size

**Old**: `auto_batch_size_logic.md` and `dynamic_batch_size_guide.md`
**New**: `mechanisms/batch_size_optimization.md`

**What to read**:
- Batch size optimization → `mechanisms/batch_size_optimization.md`

## Quick Start

### Using Vision Tokens Knob

```python
# Target: 432 vision tokens
target_tokens = 432
num_crops = (target_tokens // 144) - 1  # = 2
tiling = crops_to_tiling(num_crops, aspect_ratio)
resolution = tiling_to_resolution(tiling)
```

See: `knobs/vision_tokens_knob.md`

### Using MoE Top-K Knob

```python
set_moe_top_k(model, k=4)  # Activate top-4 experts
```

See: `knobs/moe_topk_knob.md`

### Using Transformer Blocks Knob

```python
importance_scores = compute_accuracy_drop_importance(model, dataloader, baseline_acc)
block_mask = select_top_k_blocks(importance_scores, k=12)
mask_wrapper = BlockMaskWrapper(model, block_mask)
mask_wrapper.apply()
```

See: `knobs/transformer_blocks_knob.md`

## Deprecated Documents

All deprecated documents are in `deprecated/` directory with migration paths documented in `deprecated/README.md`.

**Do not use deprecated documents for new work**. They are kept only for historical reference.

## Next Steps

1. **Use new knob documents**: All new work should reference `knobs/` documents
2. **Update code comments**: Update any code comments that reference old document paths
3. **Update scripts**: Update any scripts that reference old document paths
4. **Team communication**: Inform team members about the new documentation structure

## Questions?

- **Quick reference**: See `knobs/README.md`
- **Main index**: See `README.md`
- **Migration help**: See `deprecated/README.md`

