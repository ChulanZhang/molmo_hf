# E6: Ablations + Portability Checks

## Goal

Isolate what matters (each knob, mask strategy, estimator, task-type conditioning) and show the design generalizes beyond one narrow setting.

## Key Questions

1. Which knobs contribute most to performance?
2. Does importance-based masking outperform prefix masking?
3. Does the latency estimator improve system performance?
4. Does task-type conditioning help?
5. Does the system generalize to other benchmarks/GPUs/models?

## What to Run

### Ablation 1: Knob Ablations

Test different knob combinations:

1. **1-knob only**:
   - Vision tokens only: Vary vision_tokens, fix top_k=max, active_blocks=all
   - Top-K only: Vary top_k, fix vision_tokens=max, active_blocks=all
   - Depth only: Vary active_blocks, fix vision_tokens=max, top_k=max

2. **2-knob**:
   - Vision tokens + Top-K: Vary both, fix active_blocks=all
   - Vision tokens + Depth: Vary both, fix top_k=max
   - Top-K + Depth: Vary both, fix vision_tokens=max

3. **Full 3-knob** (FlexVLM):
   - Vary all three knobs

**Evaluation**: Compare latency-quality frontiers for each combination

### Ablation 2: Depth Mask Strategy

Compare different block selection strategies:

1. **Prefix blocks** (sequential):
   - Select first N blocks (0 to N-1)
   - Simple, no importance needed

2. **Importance-based** (recommended):
   - Select top-N blocks by importance score
   - Uses accuracy drop method (from transformer_blocks_knob.md)

3. **Random blocks**:
   - Select N random blocks
   - Baseline to show importance matters

**Evaluation**: 
- Quality vs latency at fixed budgets
- Show importance-based > prefix > random

### Ablation 3: Estimator Guardrail

Compare with/without latency estimator:

1. **No estimator**:
   - Controller evaluates all candidate configurations
   - Full profiling for each candidate

2. **With estimator**:
   - Estimator filters candidates before evaluation
   - Only evaluate configurations predicted to satisfy budget

**Evaluation**:
- SLO violation rate
- Quality retention
- Evaluation time (with vs without estimator)

### Ablation 4: Task-Type Conditioning

Compare with/without task-type as input feature:

1. **Without task-type**:
   - Controller doesn't know if request is QA vs Captioning
   - Single policy for all tasks

2. **With task-type**:
   - Controller receives task_type as input
   - Can learn task-specific policies

**Evaluation**:
- Cross-task generalization
- Quality on each task type
- Latency-quality frontiers per task

### Portability Check 1: Different Benchmarks

Test on additional benchmarks:

1. **Primary**: VQA v2, COCO Captions (from E2)
2. **Additional**: 
   - TextVQA (if available)
   - Another VQA dataset
   - Another captioning dataset

**Evaluation**: 
- Quality retention across benchmarks
- Latency characteristics
- "Recalibrate with N points → retain X% of gains"

### Portability Check 2: Different GPUs

Test cross-GPU generalization:

1. **Source GPU**: H100 (training and initial evaluation)
2. **Target GPUs**: 
   - A100
   - L40S
   - (Other available GPUs)

**Protocol**:
- Train controller on source GPU
- Recalibrate latency estimator with small sample (from E3)
- Evaluate on target GPU
- Report: Quality retention, latency characteristics

### Portability Check 3: Different Models (Optional)

If another MoE-VLM is available:
- Test same controller logic on different model
- Report: Generalization capability

## Key Outputs/Plots

### 1. Knob Ablation: Delta-to-Full-System

**Plot**: Bar chart showing performance drop
- X-axis: Ablation variants (1-knob, 2-knob, etc.)
- Y-axis: Delta from full 3-knob system
  - Quality drop (%)
  - Latency increase (%)
  - SLO violation rate increase (%)
- **Insight**: Contribution of each knob

### 2. Mask Strategy Comparison

**Plot**: Quality vs latency at fixed budgets
- X-axis: Latency (ms)
- Y-axis: Quality (accuracy/score)
- Multiple lines: Prefix, Importance-based, Random
- **Insight**: Importance-based > prefix > random

### 3. Estimator Guardrail Effectiveness

**Plot**: SLO violation rate with/without estimator
- X-axis: Budget fraction
- Y-axis: Violation rate (%)
- Multiple lines: No estimator, With estimator
- **Insight**: Estimator reduces violations

### 4. Task-Type Conditioning

**Plot**: Quality per task type
- X-axis: Task type (QA, Captioning)
- Y-axis: Quality (accuracy/score)
- Grouped bars: Without task-type, With task-type
- **Insight**: Task-type conditioning improves performance

### 5. Portability Summary

**Table**: Portability results
- Rows: Portability checks (benchmarks, GPUs, models)
- Columns:
  - Quality retention (%)
  - Latency characteristics
  - Recalibration samples needed
  - Gains retained (%)

## Implementation Details

### Knob Ablation

```python
def run_knob_ablation(ablation_type):
    """
    Run ablation for specific knob combination.
    
    Args:
        ablation_type: '1-knob', '2-knob', '3-knob'
    """
    if ablation_type == '1-knob-vision':
        # Vary only vision_tokens
        configs = [(vt, max_top_k, all_blocks) for vt in vision_token_values]
    elif ablation_type == '2-knob-vision-topk':
        # Vary vision_tokens and top_k
        configs = [(vt, tk, all_blocks) for vt in vision_token_values for tk in top_k_values]
    # ...
    
    # Evaluate and compute frontier
    results = evaluate_configs(configs)
    frontier = compute_pareto_frontier(results)
    return frontier
```

### Mask Strategy Comparison

```python
def compare_mask_strategies(num_active_blocks):
    """
    Compare different block selection strategies.
    """
    strategies = {
        'prefix': select_sequential_blocks(num_active_blocks),
        'importance': select_top_k_blocks_by_importance(num_active_blocks),
        'random': select_random_blocks(num_active_blocks)
    }
    
    results = {}
    for name, block_mask in strategies.items():
        mask_wrapper = BlockMaskWrapper(model, block_mask)
        mask_wrapper.apply()
        results[name] = evaluate_config(model, dataloader)
        mask_wrapper.remove()
    
    return results
```

### Estimator Ablation

```python
def compare_with_without_estimator(controller, estimator, dataloader, budget_ms):
    """
    Compare controller performance with/without estimator guardrail.
    """
    # Without estimator
    controller_no_est = FlexVLMController(budget_ms=budget_ms)
    results_no_est = evaluate_system(controller_no_est, dataloader, [budget_ms])
    
    # With estimator
    controller_with_est = FlexVLMController(
        latency_estimator=estimator,
        budget_ms=budget_ms
    )
    results_with_est = evaluate_system(controller_with_est, dataloader, [budget_ms])
    
    return {
        'without_estimator': results_no_est,
        'with_estimator': results_with_est
    }
```

## Expected Findings

1. **3-knob > 2-knob > 1-knob**: Demonstrates coupling importance
2. **Importance-based > prefix**: Shows importance scores matter
3. **Estimator helps**: Reduces violations and evaluation time
4. **Task-type helps**: Better cross-task performance
5. **Portable**: System generalizes with small recalibration

## Code References

- **Knob control**: `docs/knobs/` documents
- **Importance scores**: `docs/knobs/transformer_blocks_knob.md`
- **Latency estimator**: E3 results
- **Controller**: E5 implementations

## Related Experiments

- **E2**: Provides baseline for ablation comparisons
- **E3**: Provides estimator for guardrail ablation
- **E4**: Uses ablated components for system evaluation
- **E5**: Provides controller implementations for ablation

## Output Files

- `e6_knob_ablation_results.json`: Knob combination results
- `e6_mask_strategy_comparison.json`: Mask strategy results
- `e6_estimator_ablation.json`: Estimator guardrail results
- `e6_task_conditioning.json`: Task-type conditioning results
- `e6_portability_results.json`: Portability check results
- `figures/e6_knob_ablation.png`: Delta-to-full-system bars
- `figures/e6_mask_strategy.png`: Mask strategy comparison
- `figures/e6_portability_summary.png`: Portability table

