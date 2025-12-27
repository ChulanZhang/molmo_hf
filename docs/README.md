# Documentation Index

This directory contains comprehensive documentation organized by content type. All documents have been unified in English and consolidated around three main control knobs.

**📋 Migration Summary**: See [MIGRATION_SUMMARY.md](./MIGRATION_SUMMARY.md) for complete reorganization details.

## Directory Structure

```
docs/
├── README.md                          # This document (index)
├── MIGRATION_SUMMARY.md               # Migration and reorganization summary
├── knobs/                              # 🎛️ Control knobs (core documentation)
│   ├── README.md                      # Quick reference
│   ├── vision_tokens_knob.md         # Vision tokens control
│   ├── moe_topk_knob.md              # MoE top-K control
│   └── transformer_blocks_knob.md    # Transformer blocks control
├── experiments/                        # Experiment guides (all English)
│   ├── motivation_experiments.md      # Motivation experiments
│   └── profiling_experiments.md       # Profiling experiments
├── mechanisms/                         # Code mechanisms (English)
│   ├── batch_size_optimization.md     # Batch size optimization
│   ├── how_to_use_eos_token.md       # EOS token usage
│   ├── max_new_tokens_knob.md        # max_new_tokens parameter
│   └── model_inference_flow.md        # Inference pipeline
├── analysis/                           # Result analysis (content moved to knobs)
├── core_exp/                           # 🆕 Core experiments documentation
│   ├── README.md                      # Experiments overview and quick reference
│   ├── e1-e6_*.md                     # Macro experiments (main results)
│   └── m1-m6_*.md                     # Micro experiments (supporting findings)
├── development/                        # Development and refactoring
└── deprecated/                         # Archived old documents
```

## Documentation Categories

### 🎛️ knobs/ - Control Knobs (Core Documentation)

Complete documentation for the three main control knobs, unified in English with all related content consolidated.

- **vision_tokens_knob.md** - Vision Tokens Control Knob
  - Vision tokens calculation formulas and mapping
  - Reverse calculation: vision tokens → crops, tilings, image size
  - Complete implementation examples and code references
  - Limits and constraints (integrated from max_crops_limits.md)

- **moe_topk_knob.md** - MoE Top-K Control Knob
  - Dynamic MoE top-K adjustment methods
  - Performance impact and quality tradeoffs
  - Usage examples and best practices

- **transformer_blocks_knob.md** - Transformer Blocks Control Knob
  - Importance score-based block selection
  - BlockMaskWrapper implementation and usage
  - Multiple importance score calculation methods (accuracy drop, activation magnitude, gradients, attention patterns)

**Quick Reference**: See `knobs/README.md` for quick reference and combined usage examples.

### 📋 experiments/ - Experiment Guides

Guides for running experiments, using experimental tools, and configuring experiment parameters. **All documents are in English.**

- **motivation_experiments.md** - Motivation Experiments Guide
  - Detailed descriptions of 6 motivation experiments
  - Experiment goals, methods, and outputs
  - Usage methods and data formats

- **profiling_experiments.md** - Profiling Experiments Guide
  - Detailed descriptions of profiling experiments
  - Experiment scripts and usage
  - Visualization tools

### 🔧 mechanisms/ - Code Mechanisms

Technical documents for understanding code implementation, model architecture, and parameter mechanisms. **All documents are in English.**

- **batch_size_optimization.md** - Batch Size Optimization Guide (Unified)
  - Automatic batch size adjustment algorithm
  - Binary search strategy
  - Estimation formulas for all knobs
  - Usage examples and troubleshooting
  - **Replaces**: auto_batch_size_logic.md, dynamic_batch_size_guide.md

- **how_to_use_eos_token.md** - EOS Token Usage Guide
  - EOS token function and mechanism
  - Early stopping implementation
  - Usage scenarios and best practices

- **max_new_tokens_knob.md** - max_new_tokens Parameter Details
  - Parameter definition and default values
  - Generation behavior and early stopping mechanism
  - Special considerations for VQA tasks

- **model_inference_flow.md** - Model Inference Pipeline
  - Complete inference flow timeline
  - Detailed explanations of each stage
  - Performance critical path analysis

**Note**: Transformer blocks and vision tokens content has been integrated into `knobs/` directory documents.

### 📊 analysis/ - Result Analysis

Analysis and interpretation of experimental results.

**Note**: Crop overlap analysis has been integrated into `knobs/vision_tokens_knob.md`. This directory is currently empty.

### 🛠️ development/ - Development and Refactoring

Documents for code refactoring, optimization, and development plans.

- **code_reorganization_plan.md** - Code Reorganization Plan
  - Code duplication analysis
  - Refactoring strategy and implementation plan
  - Expected benefits and risks

- **profiling_code_analysis_summary.md** - Profiling Code Analysis Summary
  - Code duplication statistics
  - Refactoring opportunity identification
  - Quick overview

## Quick Reference

### I Want to Use Control Knobs
👉 See `knobs/` directory (**Recommended, latest and most complete**)
- Vision tokens control → `knobs/vision_tokens_knob.md`
- MoE top-K control → `knobs/moe_topk_knob.md`
- Transformer blocks control → `knobs/transformer_blocks_knob.md`
- Quick reference → `knobs/README.md`

### I Want to Run Experiments
👉 See `experiments/` directory (**All documents in English**)
- How to run Motivation experiments → `experiments/motivation_experiments.md`
- How to run Profiling experiments → `experiments/profiling_experiments.md`
- Batch size optimization → `mechanisms/batch_size_optimization.md`

### I Want to Understand Code Mechanisms
👉 See `mechanisms/` directory
- Model inference pipeline → `mechanisms/model_inference_flow.md`
- Batch size optimization → `mechanisms/batch_size_optimization.md`
- Parameter details → `mechanisms/max_new_tokens_knob.md`, `mechanisms/how_to_use_eos_token.md`

**Note**: 
- Vision tokens and max_crops content integrated into `knobs/vision_tokens_knob.md`
- Batch size content integrated into `mechanisms/batch_size_optimization.md`

### I Want to Analyze Results
👉 See `knobs/` directory (content moved from analysis/)
- Vision tokens and crop overlap → `knobs/vision_tokens_knob.md`

### I Want to Run Core Experiments
👉 See `core_exp/` directory (**🆕 New experimental framework**)
- Experiments overview → `core_exp/README.md`
- Stage decomposition → `core_exp/e1_stage_aware_latency_decomposition.md`
- Knob coupling → `core_exp/e2_knob_coupling_pareto.md`
- Latency estimator → `core_exp/e3_latency_estimator.md`
- System evaluation → `core_exp/e4_end_to_end_evaluation.md`
- Training comparison → `core_exp/e5_training_strategy_comparison.md`
- Ablations → `core_exp/e6_ablations_portability.md`

### I Want to Contribute to Development
👉 See `development/` directory
- Code reorganization plan → `development/code_reorganization_plan.md`
- Code analysis summary → `development/profiling_code_analysis_summary.md`

## Documentation Updates

This documentation structure was reorganized in 2024, organized around three main control knobs:
1. **knobs/** - Control knobs (core documentation, unified in English)
2. **experiments/** - Experiment guides (all in English)
3. **mechanisms/** - Code mechanisms (all in English)
4. **development/** - Development and refactoring

### Naming Convention

All documents use **snake_case** naming style (lowercase with underscores), for example:
- ✅ `vision_tokens_knob.md`
- ✅ `model_inference_flow.md`
- ✅ `batch_size_optimization.md`
- ❌ `VISION_TOKENS_KNOB.md` (old style, deprecated)
- ❌ `VisionTokensKnob.md` (not compliant)

### Adding New Documents

When adding new documents:
1. Place in appropriate directory based on content type
2. Use snake_case naming style
3. Write in English for consistency
4. Update this README index
5. Update cross-references in related documents

### Migration

For information about deprecated documents and migration paths, see:
- `deprecated/README.md` - Migration guide for deprecated documents
- `MIGRATION_SUMMARY.md` - Complete reorganization summary

