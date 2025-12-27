# M4: Budget Guardrail Effectiveness

## Goal

Show estimator-based filtering reduces SLO violations without collapsing quality.

## Key Questions

1. Does the estimator guardrail reduce SLO violations?
2. Does filtering reduce quality?
3. What is the false reject / false accept tradeoff?
4. How sensitive is performance to estimator accuracy?

## What to Run

### Comparison Setup

1. **Without guardrail**:
   - Controller evaluates all candidate configurations
   - Selects best based on policy

2. **With guardrail**:
   - Estimator filters candidates before evaluation
   - Only configurations predicted to satisfy budget are evaluated
   - Controller selects from filtered set

### Evaluation

For each budget level:
- Measure: SLO violation rate, Quality, Evaluation time
- Compare: With vs without guardrail

### Sensitivity Analysis

Test with different estimator accuracies:
- Perfect estimator (oracle)
- Trained estimator (from E3)
- Degraded estimator (add noise)

## Key Outputs/Plots

### 1. Violation Rate Comparison

**Plot**: Violation rate with/without guardrail
- X-axis: Budget fraction
- Y-axis: SLO violation rate (%)
- Multiple lines: No guardrail, With guardrail
- **Insight**: Guardrail reduces violations

### 2. Quality Retention

**Plot**: Quality with/without guardrail
- X-axis: Budget fraction
- Y-axis: Quality (accuracy/score)
- Multiple lines: No guardrail, With guardrail
- **Insight**: Quality maintained or improved

### 3. False Reject/False Accept Tradeoff

**Plot**: ROC-style curve
- X-axis: False reject rate
- Y-axis: False accept rate (or 1 - violation rate)
- Different points: Different estimator thresholds
- **Insight**: Tradeoff curve

## Expected Findings

1. **Guardrail reduces violations**: Lower violation rate
2. **Quality maintained**: No significant quality drop
3. **Evaluation time reduced**: Fewer candidates to evaluate
4. **Sensitive to accuracy**: Better estimator = better guardrail

## Code References

- **Latency estimator**: E3 results
- **Controller**: E4/E5 implementations

## Output Files

- `m4_guardrail_analysis.json`: Guardrail effectiveness data
- `figures/m4_violation_rate.png`: Violation comparison
- `figures/m4_quality_retention.png`: Quality comparison

