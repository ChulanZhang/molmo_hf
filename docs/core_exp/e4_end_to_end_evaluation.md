# E4: End-to-End System Evaluation (FlexVLM vs Baselines)

## Goal

Validate system-level gains in measured latency and quality under explicit latency budgets; show improvements on both median and tail latency.

## Key Questions

1. Does FlexVLM outperform static and single-knob baselines?
2. How much improvement on median vs tail latency?
3. What is the controller overhead?
4. How does performance scale with different latency budgets?

## What to Run

### System Integration

Integrate FlexVLM into a standard single-GPU inference stack:
- Controller: Selects (vision_tokens, top_k, active_blocks) per request
- Latency estimator: Filters configurations (optional, from E3)
- Model execution: Runs inference with selected configuration
- Quality evaluation: Measures task performance

### Baselines to Compare

1. **Static full model**
   - Fixed configuration: max vision tokens, max top_k, all blocks
   - Always uses full model capacity
   - **Expected**: Highest quality, highest latency

2. **Single-knob adaptive baselines**

   **a. Token-only**:
   - Varies: vision_tokens
   - Fixed: top_k=max, active_blocks=all
   - **Expected**: Some improvement, but limited

   **b. Depth-only**:
   - Varies: active_blocks (importance-based)
   - Fixed: vision_tokens=max, top_k=max
   - **Expected**: Some improvement, but limited

   **c. Width-only**:
   - Varies: top_k
   - Fixed: vision_tokens=max, active_blocks=all
   - **Expected**: Some improvement, but limited

3. **Disjoint multi-knob heuristics** (if available)
   - Sequential greedy: Adjust knobs one at a time
   - Example: tokens → depth → width (each step greedily meets budget)
   - **Expected**: Better than single-knob, but suboptimal

4. **FlexVLM (full system)**
   - All three knobs: vision_tokens, top_k, active_blocks
   - Importance-based block selection
   - Latency estimator guardrail (optional)
   - **Expected**: Best latency-quality tradeoff

### Evaluation Protocol

1. **Latency budgets**: Test multiple budget levels
   - Tight: 0.35× baseline latency
   - Medium: 0.50× baseline latency
   - Relaxed: 0.70× baseline latency
   - Very relaxed: 0.85× baseline latency
   - Full: 1.00× baseline latency (no constraint)

2. **Baseline measurement**:
   - Measure full model latency on validation set
   - Compute budget = budget_fraction × baseline_latency

3. **Per-request evaluation**:
   - Controller selects configuration
   - Execute inference once
   - Measure: actual latency, quality score
   - Record: configuration used, controller overhead

4. **Metrics**:
   - **Quality**: Mean accuracy/score across requests
   - **Latency**: Mean, P50, P95, P99
   - **SLO compliance**: Violation rate (latency > budget)
   - **Controller overhead**: CPU/GPU time for decision

## Key Outputs/Plots

### 1. Latency–Quality Frontiers

**Plot**: Scatter plot with Pareto frontiers
- X-axis: Latency (P95, ms)
- Y-axis: Quality (accuracy/score)
- Multiple series: One per method (static, token-only, depth-only, width-only, FlexVLM)
- **Pareto frontiers**: Highlighted for each method
- **Insight**: FlexVLM achieves better frontier

### 2. Tail Latency & SLO Compliance

**Plot 1**: P95/P99 latency vs budget
- X-axis: Budget fraction (0.35, 0.50, 0.70, 0.85, 1.00)
- Y-axis: P95 or P99 latency (ms)
- Multiple lines: One per method
- **Insight**: FlexVLM maintains lower tail latency

**Plot 2**: SLO violation rate vs budget
- X-axis: Budget fraction
- Y-axis: Violation rate (%)
- Multiple lines: One per method
- **Insight**: FlexVLM has lower violation rate

### 3. Controller Overhead

**Plot**: Overhead breakdown
- Bar chart: Controller decision time (CPU/GPU ms)
- Compare to: Latency savings from adaptive selection
- **Insight**: Overhead is negligible vs savings

### 4. Quality vs Budget

**Plot**: Quality retention under budgets
- X-axis: Budget fraction
- Y-axis: Quality (relative to full model, %)
- Multiple lines: One per method
- **Insight**: FlexVLM maintains higher quality at tight budgets

## Implementation Details

### Controller Implementation

```python
class FlexVLMController:
    def __init__(self, latency_estimator=None, budget_ms=None):
        self.estimator = latency_estimator
        self.budget_ms = budget_ms
        # Load Pareto frontiers or policy model
    
    def select_config(self, request_features):
        """
        Select configuration for request.
        
        Args:
            request_features: Dict with image, prompt, task_type, etc.
        
        Returns:
            config: (vision_tokens, top_k, active_blocks)
        """
        # 1. Filter by budget (if estimator available)
        if self.estimator and self.budget_ms:
            candidates = self.estimator.filter_by_budget(
                all_configs, self.budget_ms
            )
        else:
            candidates = all_configs
        
        # 2. Select best config (policy-based or lookup)
        # Implementation depends on training method (E5)
        best_config = self.policy.select(candidates, request_features)
        
        return best_config
```

### Baseline Implementations

**Static full model**:
```python
def static_full_model():
    return {
        'vision_tokens': 1872,  # max
        'top_k': 32,  # max
        'active_blocks': 24  # all
    }
```

**Single-knob adaptive**:
```python
def token_only_adaptive(budget_ms, baseline_latency):
    # Greedily reduce vision tokens until budget met
    target_latency = budget_ms
    # Binary search over vision token values
    # ...
    return {'vision_tokens': optimal, 'top_k': max, 'active_blocks': all}
```

### Evaluation Loop

```python
def evaluate_system(controller, dataloader, budgets):
    results = []
    
    for budget_ms in budgets:
        for batch in dataloader:
            # Controller selects config
            start_controller = time.perf_counter()
            config = controller.select_config(batch)
            controller_time = (time.perf_counter() - start_controller) * 1000
            
            # Apply config
            apply_config(model, config)
            
            # Run inference
            start_inference = time.perf_counter()
            outputs = model.generate(...)
            inference_time = (time.perf_counter() - start_inference) * 1000
            
            # Evaluate quality
            quality = evaluate_quality(outputs, batch['labels'])
            
            results.append({
                'budget_ms': budget_ms,
                'config': config,
                'latency_ms': inference_time,
                'controller_overhead_ms': controller_time,
                'quality': quality,
                'slo_violation': inference_time > budget_ms
            })
    
    return results
```

## Expected Findings

1. **FlexVLM outperforms baselines**: Better latency-quality tradeoff
2. **Multi-knob > single-knob**: Demonstrates coupling importance
3. **Tail latency improvement**: P95/P99 significantly better
4. **Low overhead**: Controller time << inference time savings
5. **Budget compliance**: Lower violation rate than baselines

## Code References

- **Controller**: To be implemented (E5 training)
- **Latency estimator**: E3 results
- **Knob control**: `docs/knobs/` documents
- **Baseline implementations**: Reference implementations needed

## Related Experiments

- **E2**: Provides Pareto frontiers for comparison
- **E3**: Provides latency estimator for guardrail
- **E5**: Provides controller training methods
- **M3**: Detailed controller overhead analysis

## Output Files

- `e4_system_results.json`: Per-request results for all methods
- `e4_summary_stats.json`: Aggregated statistics per method and budget
- `e4_controller_overhead.json`: Overhead measurements
- `figures/e4_latency_quality_frontiers.png`: Frontier comparison
- `figures/e4_tail_latency.png`: P95/P99 comparison
- `figures/e4_slo_compliance.png`: Violation rate comparison
- `figures/e4_controller_overhead.png`: Overhead breakdown

