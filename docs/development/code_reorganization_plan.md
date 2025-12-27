# Profiling Experiments Code Reorganization Plan

## Executive Summary

This document provides a comprehensive analysis of the `experiments/profiling` directory and proposes a reorganization plan to eliminate code duplication, improve maintainability, and enhance documentation structure.

**Analysis Date**: 2024-01-XX  
**Scope**: `experiments/profiling/` directory and related documentation in `docs/`

## 1. Current Structure Analysis

### 1.1 Directory Structure

```
experiments/profiling/
├── knob1_tokens/          # Exp1: Context Scaling (Vision Tokens)
│   ├── exp_context_scaling.py      # Latency measurement
│   └── exp1_accuracy.py            # Accuracy measurement
├── knob2_topk/           # Exp2: MoE Top-K Analysis
│   ├── exp_moe_topk.py             # Latency measurement
│   └── exp2_accuracy.py            # Accuracy measurement
├── knob3_layers/         # Exp3: Transformer Blocks Mask
│   ├── exp_transformer_blocks_mask.py  # Latency measurement
│   ├── exp3_accuracy.py                 # Accuracy measurement
│   └── exp3_accuracy_sensitivity.py     # Sensitivity analysis
├── knob4_output_tokens/  # Exp4: Output Tokens Scaling
│   └── exp_output_tokens_scaling.py
├── knob5_combined/       # Exp5/Exp6: Combined Control Knobs
│   ├── exp5_accuracy.py
│   ├── exp6_accuracy.py
│   └── SPARSE_SAMPLING_STRATEGIES.md
├── utils/                # Utility scripts
│   ├── analyze_vqa_answer_lengths.py
│   ├── analyze_vqa_max_crops.py
│   ├── compare_pareto_frontiers.py
│   └── ... (many more)
├── run_*.sh              # Shell scripts for running experiments
└── plot_*.py             # Plotting scripts
```

### 1.2 Code Duplication Analysis

#### 1.2.1 High-Level Duplication Patterns

**Pattern 1: Distributed Initialization (100% duplicate across 5+ files)**
- **Files affected**: `exp1_accuracy.py`, `exp2_accuracy.py`, `exp3_accuracy.py`, `exp5_accuracy.py`, `exp6_accuracy.py`
- **Lines duplicated**: ~40 lines per file
- **Total duplication**: ~200 lines
- **Code location**: `__init__` method

```python
# Identical in all accuracy experiments:
if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    self.is_distributed = True
else:
    self.is_distributed = False

self.rank = get_global_rank()
self.world_size = get_world_size()

# Set device based on local rank if distributed
if self.is_distributed:
    local_rank = get_local_rank()
    # ... device setup code ...
```

**Pattern 2: Batch Size Optimization (95% duplicate across 3+ files)**
- **Files affected**: `exp1_accuracy.py`, `exp2_accuracy.py`, `exp3_accuracy.py`
- **Lines duplicated**: ~150 lines per file
- **Total duplication**: ~450 lines
- **Code location**: `_find_optimal_batch_size` method

The `_find_optimal_batch_size` method is nearly identical across experiments, with only minor differences in:
- Parameter names (`max_crops` vs `top_k` vs `num_active_blocks`)
- Estimation function calls (`_estimate_batch_size_for_max_crops` vs `_estimate_batch_size_for_top_k` vs `_estimate_batch_size_for_num_blocks`)
- Log messages

**Pattern 3: DataLoader Creation (90% duplicate)**
- **Files affected**: All accuracy experiments
- **Lines duplicated**: ~50 lines per file
- **Total duplication**: ~250 lines
- **Code location**: Inside `run` method, `create_dataloader` factory function

```python
# Repeated in every experiment:
def create_dataloader(bs):
    mm_preprocessor = MultiModalPreprocessor(...)
    formatter = DataFormatter(...)
    preprocessor = Preprocessor(...)
    det_dataset = DeterministicDataset(dataset, preprocessor, seed=42)
    
    if self.is_distributed:
        sampler = DistributedSampler(...)
    else:
        sampler = None
    
    return torch.utils.data.DataLoader(
        det_dataset,
        batch_size=bs,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=MMCollator(...),
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
```

**Pattern 4: Result Saving and Merging (85% duplicate)**
- **Files affected**: All accuracy experiments
- **Lines duplicated**: ~100 lines per file
- **Total duplication**: ~500 lines
- **Code location**: End of `run` method

Includes:
- Incremental saving logic (per-configuration saves)
- Distributed result gathering
- Result merging across ranks
- Final summary generation

**Pattern 5: Time Statistics (100% duplicate)**
- **Files affected**: All accuracy experiments
- **Lines duplicated**: ~20 lines per file
- **Total duplication**: ~100 lines
- **Code location**: Throughout `run` method

```python
# Repeated in every experiment:
config_start_time = time.time()
# ... experiment code ...
config_end_time = time.time()
config_duration_seconds = config_end_time - config_start_time
config_duration_hours = config_duration_seconds / 3600.0
config_duration_minutes = config_duration_seconds / 60.0
```

**Pattern 6: Generation Config Setup (90% duplicate)**
- **Files affected**: All accuracy experiments
- **Lines duplicated**: ~15 lines per file
- **Total duplication**: ~75 lines

```python
# Repeated in every experiment:
from transformers import GenerationConfig
eos_token_id = self.tokenizer.eos_token_id
if eos_token_id is None:
    eos_token_id = getattr(self.model.config, 'eos_token_id', None)

pad_token_id = self.tokenizer.pad_token_id
if pad_token_id is None:
    pad_token_id = getattr(self.model.config, 'pad_token_id', None)

generation_config = GenerationConfig(
    max_new_tokens=vqa_max_tokens,
    do_sample=False,
    use_cache=True,
    eos_token_id=eos_token_id,
    pad_token_id=pad_token_id,
)
```

#### 1.2.2 Duplication Statistics

| Pattern | Files Affected | Lines per File | Total Duplication | Priority |
|---------|----------------|----------------|-------------------|----------|
| Distributed Init | 5 | ~40 | ~200 | High |
| Batch Size Opt | 3 | ~150 | ~450 | High |
| DataLoader Creation | 5 | ~50 | ~250 | Medium |
| Result Saving | 5 | ~100 | ~500 | High |
| Time Statistics | 5 | ~20 | ~100 | Low |
| Generation Config | 5 | ~15 | ~75 | Low |
| **Total** | - | - | **~1,575 lines** | - |

### 1.3 Documentation Analysis

#### 1.3.1 Current Documentation Structure

```
docs/
├── PROFILING_EXPERIMENTS.md          # Main overview
├── ACCURACY_PROFILING_OPTIMIZATION.md # Performance optimization guide
├── AUTO_BATCH_SIZE_LOGIC.md          # Batch size auto-adjustment
├── BATCH_SIZE_ANALYSIS.md            # Batch size impact analysis
├── DYNAMIC_BATCH_SIZE_GUIDE.md       # Dynamic batch size guide
├── MAX_CROPS_LIMITS.md               # max_crops limits
├── MAX_NEW_TOKENS_KNOB.md            # max_new_tokens explanation
├── MAX_NEW_TOKENS_PERFORMANCE_ANALYSIS.md
├── EXP_MOE_TOPK_ANALYSIS.md          # Exp2 analysis
├── EXP3_ANALYSIS.md                  # Exp3 analysis
├── EXP3_TRANSFORMER_BLOCKS_MASK.md   # Exp3 details
├── EXP4_ANALYSIS.md                  # Exp4 analysis
└── EXP6_CROP_OVERLAP_ANALYSIS.md     # Exp6 analysis
```

#### 1.3.2 Documentation Issues

1. **Overlap and Redundancy**:
   - `AUTO_BATCH_SIZE_LOGIC.md` and `DYNAMIC_BATCH_SIZE_GUIDE.md` cover similar topics
   - `EXP3_ANALYSIS.md` and `EXP3_TRANSFORMER_BLOCKS_MASK.md` have overlapping content
   - Multiple documents explain `max_new_tokens` (3 separate files)

2. **Missing Documentation**:
   - No unified guide for running all experiments
   - No clear explanation of the relationship between latency and accuracy experiments
   - No documentation for Exp5/Exp6 combined experiments

3. **Organization Issues**:
   - Documentation is scattered across multiple files
   - No clear hierarchy or navigation structure
   - Some documents are very detailed, others are brief

## 2. Reorganization Plan

### 2.1 Code Refactoring Strategy

#### Phase 1: Create Shared Base Classes and Utilities

**2.1.1 Create `AccuracyExperimentBase` Class**

**Location**: `experiments/profiling/common/accuracy_experiment_base.py`

**Purpose**: Extract all common functionality from accuracy experiments into a base class.

**Extracted Components**:
1. Distributed initialization (`__init__` method)
2. Batch size optimization (`_find_optimal_batch_size` with pluggable estimation)
3. DataLoader creation factory
4. Result saving and merging logic
5. Time statistics tracking
6. Generation config setup

**Interface Design**:
```python
class AccuracyExperimentBase(BaseExperiment):
    """
    Base class for all accuracy profiling experiments.
    
    Handles:
    - Distributed setup
    - Batch size optimization
    - DataLoader creation
    - Result saving and merging
    - Time statistics
    """
    
    def __init__(self, model_path, device="cuda", output_dir="./results", ...):
        # Distributed initialization (extracted)
        ...
    
    def _estimate_batch_size(self, config_value, base_batch_size):
        """
        Abstract method to be implemented by subclasses.
        Returns estimated batch size for given configuration.
        """
        raise NotImplementedError
    
    def _find_optimal_batch_size(self, config_value, initial_batch_size, dataloader_factory):
        """
        Generic batch size optimization (works for any config type).
        """
        # Common implementation
        ...
    
    def _create_dataloader_factory(self, dataset, max_crops):
        """
        Create a factory function for DataLoader creation.
        """
        # Common implementation
        ...
    
    def _save_incremental_result(self, config_name, config_value, result_entry):
        """
        Save result for a single configuration.
        """
        # Common implementation
        ...
    
    def _merge_distributed_results(self, results_data, config_list):
        """
        Merge results from all ranks in distributed mode.
        """
        # Common implementation
        ...
    
    def _setup_generation_config(self, max_new_tokens=16):
        """
        Create and return GenerationConfig.
        """
        # Common implementation
        ...
    
    def run(self, ...):
        """
        Template method pattern - subclasses implement:
        - _get_config_list() - return list of configurations to test
        - _apply_config(config_value) - apply configuration to model
        - _get_config_name() - return name for this config (e.g., "max_crops")
        """
        # Common flow:
        # 1. Get config list
        # 2. For each config:
        #    a. Apply config
        #    b. Find optimal batch size
        #    c. Create dataloader
        #    d. Run inference
        #    e. Compute accuracy
        #    f. Save incremental result
        # 3. Merge distributed results
        # 4. Save final results
```

**2.1.2 Create Batch Size Estimation Utilities**

**Location**: `experiments/profiling/common/batch_size_utils.py`

**Purpose**: Centralize batch size estimation logic.

```python
def estimate_batch_size_for_max_crops(max_crops: int, base_batch_size: int) -> int:
    """Estimate batch size for max_crops configuration."""
    ...

def estimate_batch_size_for_top_k(top_k: int, base_batch_size: int) -> int:
    """Estimate batch size for top_k configuration."""
    ...

def estimate_batch_size_for_num_blocks(num_active: int, total: int, base_batch_size: int) -> int:
    """Estimate batch size for num_active_blocks configuration."""
    ...
```

**2.1.3 Create Result Management Utilities**

**Location**: `experiments/profiling/common/result_utils.py`

**Purpose**: Centralize result saving and merging logic.

```python
def save_incremental_result(output_dir, config_name, config_value, result_entry, rank=None):
    """Save result for a single configuration."""
    ...

def merge_distributed_results(gathered_results, config_list, config_key_func):
    """Merge results from all ranks."""
    ...

def create_summary(results_data):
    """Create summary from results data."""
    ...
```

#### Phase 2: Refactor Individual Experiments

**2.2.1 Refactor Exp1 (knob1_tokens/exp1_accuracy.py)**

**Before**: ~795 lines  
**After**: ~300 lines (estimated)

**Changes**:
- Inherit from `AccuracyExperimentBase`
- Implement only:
  - `_estimate_batch_size(max_crops, base_batch_size)` → delegate to `batch_size_utils.estimate_batch_size_for_max_crops`
  - `_apply_config(max_crops)` → set `self.model.config.max_crops = max_crops`
  - `_get_config_list()` → return `max_crops_list`
  - `_get_config_name()` → return `"max_crops"`

**2.2.2 Refactor Exp2 (knob2_topk/exp2_accuracy.py)**

**Before**: ~712 lines  
**After**: ~350 lines (estimated)

**Changes**:
- Inherit from `AccuracyExperimentBase`
- Keep `_set_top_k` method (experiment-specific)
- Implement only experiment-specific methods

**2.2.3 Refactor Exp3 (knob3_layers/exp3_accuracy.py)**

**Before**: ~768 lines  
**After**: ~400 lines (estimated)

**Changes**:
- Inherit from `AccuracyExperimentBase`
- Keep `BlockMaskWrapper` usage (experiment-specific)
- Implement only experiment-specific methods

**2.2.4 Refactor Exp5/Exp6 (knob5_combined/)**

**Before**: ~1000+ lines each  
**After**: ~500 lines each (estimated)

**Changes**:
- Inherit from `AccuracyExperimentBase`
- Implement multi-configuration logic (combinations of knobs)

### 2.2 Directory Reorganization

#### Proposed Structure

```
experiments/profiling/
├── common/                          # NEW: Shared code
│   ├── __init__.py
│   ├── accuracy_experiment_base.py  # Base class for accuracy experiments
│   ├── batch_size_utils.py          # Batch size estimation utilities
│   ├── result_utils.py              # Result saving/merging utilities
│   └── distributed_utils.py        # Distributed setup utilities
├── knob1_tokens/
│   ├── exp_context_scaling.py      # Latency (unchanged)
│   └── exp1_accuracy.py             # Accuracy (refactored)
├── knob2_topk/
│   ├── exp_moe_topk.py             # Latency (unchanged)
│   └── exp2_accuracy.py             # Accuracy (refactored)
├── knob3_layers/
│   ├── exp_transformer_blocks_mask.py  # Latency (unchanged)
│   ├── exp3_accuracy.py                 # Accuracy (refactored)
│   └── exp3_accuracy_sensitivity.py     # Sensitivity (unchanged)
├── knob4_output_tokens/
│   └── exp_output_tokens_scaling.py     # Latency (unchanged)
├── knob5_combined/
│   ├── exp5_accuracy.py            # Accuracy (refactored)
│   ├── exp6_accuracy.py            # Accuracy (refactored)
│   └── SPARSE_SAMPLING_STRATEGIES.md
├── utils/                           # Utility scripts (unchanged)
├── scripts/                         # NEW: Organized shell scripts
│   ├── run_exp1_context_scaling.sh
│   ├── run_exp2_moe_topk.sh
│   ├── run_exp2_accuracy.sh
│   ├── run_exp3_accuracy.sh
│   ├── run_exp5_accuracy.sh
│   └── run_all_experiments.sh
└── plotting/                        # NEW: Organized plotting scripts
    ├── plot_exp1_context_scaling.py
    ├── plot_exp2_moe_topk.py
    └── plot_exp1_exp2_exp3_accuracy.py
```

### 2.3 Documentation Reorganization

#### Proposed Documentation Structure

```
docs/
├── profiling/
│   ├── README.md                    # NEW: Main entry point
│   ├── overview.md                  # Overview of all profiling experiments
│   ├── running_experiments.md       # NEW: Guide for running experiments
│   ├── accuracy_experiments.md      # NEW: Unified guide for accuracy experiments
│   ├── latency_experiments.md      # NEW: Unified guide for latency experiments
│   ├── batch_size_optimization.md   # MERGED: Combine AUTO_BATCH_SIZE_LOGIC.md + DYNAMIC_BATCH_SIZE_GUIDE.md
│   ├── max_new_tokens.md            # MERGED: Combine MAX_NEW_TOKENS_KNOB.md + MAX_NEW_TOKENS_PERFORMANCE_ANALYSIS.md
│   ├── experiments/
│   │   ├── exp1_context_scaling.md
│   │   ├── exp2_moe_topk.md
│   │   ├── exp3_transformer_blocks.md
│   │   ├── exp4_output_tokens.md
│   │   ├── exp5_combined_accuracy.md
│   │   └── exp6_combined_latency.md
│   └── technical/
│       ├── batch_size_analysis.md   # Renamed from BATCH_SIZE_ANALYSIS.md
│       ├── max_crops_limits.md      # Renamed from MAX_CROPS_LIMITS.md
│       └── performance_optimization.md  # Renamed from ACCURACY_PROFILING_OPTIMIZATION.md
└── PROFILING_EXPERIMENTS.md         # DEPRECATED: Redirect to profiling/README.md
```

#### Documentation Consolidation Plan

1. **Merge Related Documents**:
   - `AUTO_BATCH_SIZE_LOGIC.md` + `DYNAMIC_BATCH_SIZE_GUIDE.md` → `profiling/batch_size_optimization.md`
   - `MAX_NEW_TOKENS_KNOB.md` + `MAX_NEW_TOKENS_PERFORMANCE_ANALYSIS.md` → `profiling/max_new_tokens.md`
   - `EXP3_ANALYSIS.md` + `EXP3_TRANSFORMER_BLOCKS_MASK.md` → `profiling/experiments/exp3_transformer_blocks.md`

2. **Create New Unified Guides**:
   - `profiling/README.md`: Main entry point with navigation
   - `profiling/running_experiments.md`: Step-by-step guide for running experiments
   - `profiling/accuracy_experiments.md`: Unified guide for all accuracy experiments
   - `profiling/latency_experiments.md`: Unified guide for all latency experiments

3. **Reorganize Existing Documents**:
   - Move experiment-specific docs to `profiling/experiments/`
   - Move technical deep-dives to `profiling/technical/`
   - Keep only high-level overview in root `docs/`

## 3. Implementation Plan

### Phase 1: Foundation (Week 1)

1. **Create Common Module Structure**
   - [ ] Create `experiments/profiling/common/` directory
   - [ ] Create `__init__.py` with exports
   - [ ] Implement `distributed_utils.py`
   - [ ] Implement `batch_size_utils.py`
   - [ ] Implement `result_utils.py`

2. **Create Base Class**
   - [ ] Implement `AccuracyExperimentBase` class
   - [ ] Extract distributed initialization
   - [ ] Extract batch size optimization
   - [ ] Extract DataLoader creation
   - [ ] Extract result saving/merging
   - [ ] Extract time statistics
   - [ ] Extract generation config setup

3. **Testing**
   - [ ] Create unit tests for common utilities
   - [ ] Test base class with a simple experiment

### Phase 2: Refactor Experiments (Week 2)

1. **Refactor Exp1**
   - [ ] Update `exp1_accuracy.py` to inherit from base class
   - [ ] Remove duplicated code
   - [ ] Test with existing data
   - [ ] Verify results match previous runs

2. **Refactor Exp2**
   - [ ] Update `exp2_accuracy.py` to inherit from base class
   - [ ] Remove duplicated code
   - [ ] Test with existing data
   - [ ] Verify results match previous runs

3. **Refactor Exp3**
   - [ ] Update `exp3_accuracy.py` to inherit from base class
   - [ ] Remove duplicated code
   - [ ] Test with existing data
   - [ ] Verify results match previous runs

### Phase 3: Advanced Refactoring (Week 3)

1. **Refactor Exp5/Exp6**
   - [ ] Update to use base class
   - [ ] Handle multi-configuration logic
   - [ ] Test with existing data

2. **Directory Reorganization**
   - [ ] Move shell scripts to `scripts/` directory
   - [ ] Move plotting scripts to `plotting/` directory
   - [ ] Update all references in scripts

### Phase 4: Documentation (Week 4)

1. **Create New Documentation Structure**
   - [ ] Create `docs/profiling/` directory structure
   - [ ] Write `profiling/README.md`
   - [ ] Write `profiling/running_experiments.md`
   - [ ] Write `profiling/accuracy_experiments.md`
   - [ ] Write `profiling/latency_experiments.md`

2. **Consolidate Existing Documentation**
   - [ ] Merge batch size documents
   - [ ] Merge max_new_tokens documents
   - [ ] Merge Exp3 documents
   - [ ] Move experiment-specific docs
   - [ ] Move technical docs

3. **Update Cross-References**
   - [ ] Update all internal links
   - [ ] Update shell script comments
   - [ ] Update code comments

### Phase 5: Cleanup and Validation (Week 5)

1. **Remove Deprecated Code**
   - [ ] Remove old duplicated code
   - [ ] Remove deprecated documentation files
   - [ ] Clean up unused imports

2. **Final Testing**
   - [ ] Run all experiments end-to-end
   - [ ] Verify all results match previous runs
   - [ ] Test distributed execution
   - [ ] Test batch size optimization

3. **Documentation Review**
   - [ ] Review all documentation for accuracy
   - [ ] Check all links work
   - [ ] Ensure examples are up-to-date

## 4. Expected Benefits

### 4.1 Code Quality

- **Reduced Duplication**: Eliminate ~1,575 lines of duplicated code
- **Improved Maintainability**: Changes to common functionality only need to be made once
- **Easier Testing**: Common functionality can be tested independently
- **Better Consistency**: All experiments use the same underlying implementation

### 4.2 Developer Experience

- **Faster Development**: New experiments can inherit from base class
- **Clearer Code**: Experiment-specific code is separated from common code
- **Better Documentation**: Unified documentation structure
- **Easier Onboarding**: Clear entry points and guides

### 4.3 Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total Lines of Code | ~5,000 | ~3,500 | -30% |
| Code Duplication | ~1,575 lines | ~0 lines | -100% |
| Number of Files | 20+ | 15+ | Better organized |
| Documentation Files | 14 | 12 | Better organized |

## 5. Risk Assessment

### 5.1 Risks

1. **Breaking Changes**: Refactoring might introduce bugs
   - **Mitigation**: Comprehensive testing at each phase
   - **Mitigation**: Keep old code until new code is verified

2. **Incompatibility**: New base class might not fit all experiments
   - **Mitigation**: Design base class to be flexible
   - **Mitigation**: Allow experiments to override methods

3. **Time Investment**: Refactoring takes time
   - **Mitigation**: Phased approach allows incremental progress
   - **Mitigation**: Benefits outweigh costs in long term

### 5.2 Rollback Plan

- Keep old code in a `_backup/` directory
- Use version control (git) for easy rollback
- Test thoroughly before removing old code

## 6. Success Criteria

1. ✅ All experiments produce identical results before and after refactoring
2. ✅ Code duplication reduced by >90%
3. ✅ All documentation is organized and accessible
4. ✅ All tests pass
5. ✅ New experiments can be created by inheriting from base class
6. ✅ No performance regression

## 7. Next Steps

1. **Review and Approval**: Get team review of this plan
2. **Prioritize**: Decide which phases to implement first
3. **Assign Tasks**: Assign team members to specific phases
4. **Start Implementation**: Begin with Phase 1

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-XX  
**Author**: Code Analysis and Reorganization Team




