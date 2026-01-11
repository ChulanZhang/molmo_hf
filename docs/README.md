# Documentation Index

This directory contains comprehensive documentation organized by content type. All documents have been unified in English and consolidated around three main control knobs.


## Directory Structure

```
docs/
├── README.md                          # This document (index)
├── knobs/                              # 🎛️ Control knobs (core documentation)
│   ├── README.md                      # Quick reference
│   ├── vision_tokens_knob.md         # Vision tokens control
│   ├── moe_topk_knob.md              # MoE top-K control
│   └── transformer_blocks_knob.md    # Transformer blocks control
├── experiments/                        # Experiment guides (all English)
│   ├── motivation_experiments.md      # Motivation experiments
│   ├── profiling_experiments.md       # Profiling experiments
│   └── all_9_datasets_data_requirements.md  # Data requirements for 9 datasets
├── mechanisms/                         # Code mechanisms (English)
│   ├── batch_size_optimization.md     # Batch size optimization
│   ├── how_to_use_eos_token.md       # EOS token usage
│   ├── max_new_tokens_knob.md        # max_new_tokens parameter
│   └── model_inference_flow.md        # Inference pipeline
├── analysis/                           # Latency measurement analysis
│   ├── README.md                      # Analysis index
│   ├── key_insights_latency_measurement.md
│   ├── latency_measurement_refactoring.md
│   └── ...                            # Other analysis documents
├── controller/                         # 🆕 Controller design and training
│   ├── README.md                      # Controller documentation index
│   ├── DESIGN.md                      # Unified design document
│   ├── JOINT_TRAINING.md              # Joint training details
│   ├── archive/                       # Archived documents
│   └── research/                      # Research documents
├── core_exp/                           # 🆕 Core experiments documentation
│   ├── README.md                      # Experiments overview and quick reference
│   ├── coco_caption_evaluation.md     # COCO Caption evaluation guide
│   ├── e1-e6_*.md                     # Macro experiments (main results)
│   └── m1-m6_*.md                     # Micro experiments (supporting findings)
```

## Documentation Categories

### 🎛️ knobs/ - Control Knobs (Core Documentation)

Complete documentation for the three main control knobs, unified in English with all related content consolidated.

- **vision_tokens_knob.md** - Vision Tokens Control Knob
  - Vision tokens calculation formulas and mapping
  - Reverse calculation: vision tokens → crops, tilings, image size
  - Complete implementation examples and code references
  - Limits and constraints (integrated from max_crops_limits.md)
  - Adaptive tiling selection and resize_to_fill mechanism

- **vision_tokens_knob_examples.md** - Real-World Examples
  - Real image examples from VQA v2 dataset
  - Different resolutions and aspect ratios
  - Step-by-step processing demonstrations
  - Practical recommendations based on real examples

- **vision_tokens_knob_qa.md** - Q&A
  - Why some result files use "imgsize" naming
  - Tier-based design vs fixed targets discussion
  - Recommendations for immediate and long-term improvements


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

### 📊 analysis/ - Latency Measurement Analysis

Analysis and interpretation of latency measurement issues, solutions, and refactoring.

**Key Documents**:
- `key_insights_latency_measurement.md` - Key insights summary
- `latency_measurement_refactoring.md` - Complete refactoring documentation
- `latency_measurement_issue_summary.md` - Issue summary
- `decode_measurement_strategy.md` - Decode measurement strategy
- `tier_fallback_analysis.md` - Tier fallback analysis

**Note**: These documents focus on latency measurement mechanisms and are valuable for understanding the measurement system.

### 🎮 controller/ - Controller Design and Training

Complete documentation for the adaptive controller system.

**Key Documents**:
- `README.md` - Documentation index
- `DESIGN.md` - Unified design document
- `JOINT_TRAINING.md` - Joint training details
- `ANALYSIS.md` - Technical analysis
- `training_guide.md` - Training guide
- `EXPERIMENTS.md` - Experiments documentation

**Subdirectories**:
- `archive/` - Archived documents (outdated but potentially useful)
- `research/` - Research documents (design exploration)

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
  - Real-world examples → `knobs/vision_tokens_knob_examples.md`
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
👉 See `analysis/` directory (latency measurement analysis)
- Key insights → `analysis/key_insights_latency_measurement.md`
- Complete refactoring → `analysis/latency_measurement_refactoring.md`

👉 See `knobs/` directory (control knob analysis)
- Vision tokens and crop overlap → `knobs/vision_tokens_knob.md`

### I Want to Understand Controller Design
👉 See `controller/` directory
- Quick start → `controller/OVERVIEW.md`
- Design document → `controller/DESIGN.md`
- Joint training → `controller/JOINT_TRAINING.md`
- Technical analysis → `controller/ANALYSIS.md`

### I Want to Run Core Experiments
👉 See `core_exp/` directory (**🆕 New experimental framework**)
- Experiments overview → `core_exp/README.md`
- Stage decomposition → `core_exp/e1_stage_aware_latency_decomposition.md`
- Knob coupling → `core_exp/e2_knob_coupling_pareto.md`
- Latency estimator → `core_exp/e3_latency_estimator.md`
- System evaluation → `core_exp/e4_end_to_end_evaluation.md`
- Training comparison → `core_exp/e5_training_strategy_comparison.md`
- Ablations → `core_exp/e6_ablations_portability.md`


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

## Documentation Standards

### File Naming Convention

All documents use **snake_case** naming style (lowercase with underscores):
- ✅ `vision_tokens_knob.md`
- ✅ `vision_tokens_knob_examples.md`
- ✅ `model_inference_flow.md`
- ✅ `batch_size_optimization.md`
- ✅ `all_9_datasets_data_requirements.md`
- ❌ `VISION_TOKENS_KNOB.md` (old style, deprecated)
- ❌ `VisionTokensKnob.md` (not compliant)
- ❌ `ALL_9_DATASETS_DATA_REQUIREMENTS.md` (old style, deprecated)

**Important**: All document filenames must be in **lowercase** with underscores (snake_case). This applies to:
- All markdown files in `docs/`
- All subdirectories (use lowercase)
- All references in code and documentation

### Document Structure Standards

#### 1. Control Knob Documents (`knobs/`)

**Main document** (e.g., `vision_tokens_knob.md`):
- **Overview**: High-level description and key principles
- **Core Formula**: Mathematical formulas and constants
- **Pipeline/Workflow**: Step-by-step explanation
- **Control Methods**: Different ways to use the knob
- **Examples**: Brief examples (detailed examples in separate file)
- **Limits and Constraints**: Practical limitations
- **Code References**: Links to relevant code
- **Related Documents**: Cross-references

**Examples document** (e.g., `vision_tokens_knob_examples.md`):
- **Example Selection Criteria**: How examples were chosen
- **Multiple Real Examples**: Actual images from datasets
  - Image characteristics (size, aspect ratio)
  - Step-by-step processing with different targets
  - Results and insights
- **Summary**: Key insights from examples
- **Practical Recommendations**: Best practices

**Format for examples**:
```markdown
## Example N: [Image Type]

### Image Characteristics
- **Original size**: W×H (width×height)
- **Aspect ratio**: X.XX
- **Image ID**: From [dataset] [split] set

### Processing with Target: [N] Vision Tokens ([M] crops)

**Step 1: [Description]**
```
[Code or calculation]
```

**Step 2: [Description]**
...
```

#### 2. Experiment Documents (`experiments/`, `core_exp/`)

- **Overview**: Experiment goals and methods
- **Usage**: How to run the experiment
- **Configuration**: Parameters and options
- **Output**: Result file formats
- **Analysis**: How to interpret results

#### 3. Mechanism Documents (`mechanisms/`)

- **Overview**: What the mechanism does
- **How It Works**: Detailed explanation
- **Usage**: How to use it
- **Code References**: Relevant code locations
- **Best Practices**: Recommendations

### Content Standards

#### 1. Use Real Examples When Possible

- **Control knobs**: Include real image examples from datasets
- **Experiments**: Show actual command-line usage
- **Mechanisms**: Provide code snippets from actual implementation

#### 2. Include Step-by-Step Explanations

For complex processes, break down into numbered steps:
```markdown
**Step 1: [Action]**
[Explanation]

**Step 2: [Action]**
[Explanation]
```

#### 3. Use Tables for Structured Data

For constants, mappings, comparisons:
```markdown
| Parameter | Value | Description |
|-----------|-------|-------------|
| ... | ... | ... |
```

#### 4. Include Code References

Always link to actual code:
```markdown
- **Function**: `molmo/data/model_preprocessor.py:202-276`
- **Class**: `experiments/core_exp/acc_lat_profiling.py:CombinedProfilingExperiment`
```

#### 5. Cross-Reference Related Documents

At the end of each document:
```markdown
## Related Documents

- `related_doc1.md`: Description
- `related_doc2.md`: Description
```

### Example Document Template

```markdown
# [Document Title]

## Overview

[High-level description and key principles]

## [Main Section 1]

[Content]

### [Subsection]

[Detailed explanation]

**Example**:
```
[Code or calculation]
```

## [Main Section 2]

[Content]

## Real-World Examples

For detailed examples, see:
- `examples_doc.md`: [Description]

## Related Documents

- `related_doc1.md`: [Description]
- `related_doc2.md`: [Description]
```

### Updating Documentation

When updating documentation:
1. **Maintain consistency**: Follow existing structure and style
2. **Update cross-references**: Update links in related documents
3. **Update index**: Update this README if adding new documents
4. **Version control**: Use clear commit messages describing changes
5. **Review**: Ensure examples are accurate and code references are correct


