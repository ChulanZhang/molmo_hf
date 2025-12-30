# Core Experiments Documentation

This directory contains detailed documentation for all core experiments in the FlexVLM project, organized as independent documents for development.

## Overview

The experiments are organized into two categories:
- **Macro Experiments (E1-E6)**: Main results and system evaluation
- **Micro Experiments (M1-M6)**: Supporting findings and sanity checks

## Dataset Information

### Dataset Sample Counts
**Document**: `dataset_sample_counts.md`

Complete list of sample counts for all 9 datasets used in experiments:
- coco_2014_vqa, coco_caption, tally_qa, doc_qa, okvqa, text_vqa, science_qa_img, st_qa, mmmu

Includes recommendations for `num_samples` configuration based on dataset sizes.

## Macro Experiments

### E1: Stage-Aware Latency Decomposition
**Document**: `e1_stage_aware_latency_decomposition.md`

Quantify which stages dominate end-to-end latency and demonstrate why FLOPs ≠ wall-clock latency.

**Key outputs**:
- Stage time shares (stack plots)
- Scaling curves (prefill vs input tokens, decode vs output tokens)
- Attention vs FFN breakdown
- FLOPs vs latency scatter

### E2: Knob Coupling + Pareto-Front Structure
**Document**: `e2_knob_coupling_pareto.md`

Demonstrate that three knobs are coupled and optimal configurations differ by workload.

**Key outputs**:
- Per-workload Pareto frontiers
- Coupling proof plots
- Illustrative cases

### E3: Stage-Aware Latency Estimator
**Document**: `e3_latency_estimator.md`

Build lightweight latency model and show cross-GPU transfer with small recalibration.

**Key outputs**:
- Prediction error vs calibration samples
- Cross-GPU transfer curves
- Budget enforcement quality

### E4: End-to-End System Evaluation
**Document**: `e4_end_to_end_evaluation.md`

Validate system-level gains vs baselines under latency budgets.

**Key outputs**:
- Latency-quality frontiers (FlexVLM vs baselines)
- Tail latency & SLO compliance
- Controller overhead

### E5: Training Strategy Comparison
**Document**: `e5_training_strategy_comparison.md`

Compare training methods (LUT, Supervised, PPO, DPO, GRPO) for controller learning.

**Key outputs**:
- Learning curves
- Sample efficiency tables
- Final frontiers comparison
- Overhead accounting

### E6: Ablations + Portability
**Document**: `e6_ablations_portability.md`

Isolate what matters and show design generalizes.

**Key outputs**:
- Knob ablation results
- Mask strategy comparison
- Estimator guardrail effectiveness
- Portability summary

## Micro Experiments

### M1: Prefill-Heavy vs Decode-Heavy Contrast
**Document**: `m1_prefill_decode_contrast.md`

Show QA tasks are prefill-sensitive, captioning is decode-sensitive.

### M2: Latency Repeatability
**Document**: `m2_latency_repeatability.md`

Measure variance and determine how many repeats are needed.

### M3: Controller Overhead
**Document**: `m3_controller_overhead.md`

Measure controller runtime overhead and show it's negligible.

### M4: Budget Guardrail Effectiveness
**Document**: `m4_budget_guardrail_effectiveness.md`

Show estimator-based filtering reduces violations without quality loss.

### M5: Depth Mask Policy
**Document**: `m5_depth_mask_policy.md`

Compare prefix vs importance vs random masks.

### M6: Generalization Check
**Document**: `m6_generalization_check.md`

Test on additional benchmarks/models to show robustness.

## Experiment Dependencies

```
E1 (Stage Decomposition)
  ↓
E2 (Knob Coupling) ──→ E3 (Estimator)
  ↓                      ↓
E4 (System Eval) ←───────┘
  ↑
E5 (Training) ──→ E6 (Ablations)
  ↓
M1-M6 (Supporting)
```

## Development Workflow

1. **Read experiment document**: Understand goal, what to run, expected outputs
2. **Implement measurement code**: Based on implementation details
3. **Run experiments**: Collect data according to protocol
4. **Generate plots**: Create visualizations from key outputs
5. **Update document**: Add actual results and findings

## Code Organization

Each experiment should have:
- **Script**: `experiments/core_exp/e{N}_*.py` or `experiments/core_exp/m{N}_*.py`
- **Results**: `results/core_exp/e{N}_*.json`
- **Plots**: `results/core_exp/figures/e{N}_*.png`
- **Documentation**: `docs/core_exp/e{N}_*.md`

## Related Documentation

- **Control Knobs**: `../knobs/` - How to control each knob
- **Batch Size Optimization**: `../mechanisms/batch_size_optimization.md`
- **Model Inference**: `../mechanisms/model_inference_flow.md`
- **COCO Caption Evaluation**: `coco_caption_evaluation.md` - Standard COCO Caption evaluation guide

## Quick Reference

| Experiment | Goal | Key Metric | Dependencies |
|------------|------|------------|--------------|
| E1 | Stage decomposition | Stage latencies | None |
| E2 | Knob coupling | Pareto frontiers | E1 |
| E3 | Latency estimator | Prediction error | E1, E2 |
| E4 | System evaluation | Latency-quality | E2, E3, E5 |
| E5 | Training comparison | Sample efficiency | E2 |
| E6 | Ablations | Component contribution | E2, E3, E5 |
| M1 | Workload contrast | Stage breakdown | E1, E2 |
| M2 | Repeatability | Variance/CoV | None |
| M3 | Overhead | Controller time | E4, E5 |
| M4 | Guardrail | Violation rate | E3, E4 |
| M5 | Mask policy | Quality comparison | E2 |
| M6 | Generalization | Quality retention | E4 |

