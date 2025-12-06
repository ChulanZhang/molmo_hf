# Profiling Experiments Code Analysis - Executive Summary

## Quick Overview

This document provides a quick summary of code duplication and reorganization opportunities in the `experiments/profiling/` directory.

## Key Findings

### Code Duplication

**Total Duplicated Code**: ~1,575 lines across 5+ files

| Component | Duplication | Files Affected | Priority |
|-----------|-------------|----------------|----------|
| Distributed Initialization | ~200 lines | 5 files | High |
| Batch Size Optimization | ~450 lines | 3 files | High |
| DataLoader Creation | ~250 lines | 5 files | Medium |
| Result Saving/Merging | ~500 lines | 5 files | High |
| Time Statistics | ~100 lines | 5 files | Low |
| Generation Config | ~75 lines | 5 files | Low |

### Main Issues

1. **5 accuracy experiments** share ~80% of their code
2. **Distributed setup** is duplicated identically in all experiments
3. **Batch size optimization** logic is nearly identical
4. **Result saving/merging** follows the same pattern everywhere
5. **Documentation** is scattered and has overlapping content

## Proposed Solution

### Create Shared Base Class

**New File**: `experiments/profiling/common/accuracy_experiment_base.py`

**Extract Common Functionality**:
- Distributed initialization
- Batch size optimization (with pluggable estimation)
- DataLoader creation
- Result saving and merging
- Time statistics
- Generation config setup

### Expected Impact

- **Code Reduction**: ~30% reduction in total lines
- **Duplication Elimination**: ~100% reduction in duplicated code
- **Maintainability**: Changes to common code only need to be made once
- **New Experiments**: Can be created by inheriting from base class

## Implementation Phases

1. **Phase 1**: Create common utilities and base class (Week 1)
2. **Phase 2**: Refactor Exp1, Exp2, Exp3 (Week 2)
3. **Phase 3**: Refactor Exp5/Exp6 and reorganize directories (Week 3)
4. **Phase 4**: Reorganize documentation (Week 4)
5. **Phase 5**: Testing and cleanup (Week 5)

## Documentation Reorganization

### Current Issues
- 14 documentation files, some with overlapping content
- No clear entry point or navigation
- Scattered across root `docs/` directory

### Proposed Structure
```
docs/profiling/
├── README.md                    # Main entry point
├── running_experiments.md       # How to run experiments
├── accuracy_experiments.md      # Unified accuracy guide
├── latency_experiments.md      # Unified latency guide
├── experiments/                 # Experiment-specific docs
└── technical/                   # Technical deep-dives
```

## Files to Create/Modify

### New Files
- `experiments/profiling/common/__init__.py`
- `experiments/profiling/common/accuracy_experiment_base.py`
- `experiments/profiling/common/batch_size_utils.py`
- `experiments/profiling/common/result_utils.py`
- `experiments/profiling/common/distributed_utils.py`
- `docs/profiling/README.md`
- `docs/profiling/running_experiments.md`
- `docs/profiling/accuracy_experiments.md`
- `docs/profiling/latency_experiments.md`

### Files to Refactor
- `knob1_tokens/exp1_accuracy.py` (795 → ~300 lines)
- `knob2_topk/exp2_accuracy.py` (712 → ~350 lines)
- `knob3_layers/exp3_accuracy.py` (768 → ~400 lines)
- `knob5_combined/exp5_accuracy.py` (1000+ → ~500 lines)
- `knob5_combined/exp6_accuracy.py` (1000+ → ~500 lines)

### Files to Merge/Reorganize
- `AUTO_BATCH_SIZE_LOGIC.md` + `DYNAMIC_BATCH_SIZE_GUIDE.md` → `profiling/batch_size_optimization.md`
- `MAX_NEW_TOKENS_KNOB.md` + `MAX_NEW_TOKENS_PERFORMANCE_ANALYSIS.md` → `profiling/max_new_tokens.md`
- `EXP3_ANALYSIS.md` + `EXP3_TRANSFORMER_BLOCKS_MASK.md` → `profiling/experiments/exp3_transformer_blocks.md`

## Success Metrics

- ✅ Code duplication reduced by >90%
- ✅ All experiments produce identical results
- ✅ New experiments can be created in <100 lines
- ✅ Documentation is organized and accessible
- ✅ No performance regression

## Detailed Plan

See `CODE_REORGANIZATION_PLAN.md` for the complete detailed plan.

---

**Last Updated**: 2024-01-XX

