# M6: Minimal Generalization Check

## Goal

One additional benchmark/task type (or second model) demonstrating the findings are not overfit to a single dataset/model.

## Key Questions

1. Do findings generalize to other benchmarks?
2. Do findings generalize to other models?
3. How much recalibration is needed for generalization?
4. Are the key insights robust?

## What to Run

### Additional Benchmark

Test on one additional benchmark:
- **Option 1**: TextVQA (if available)
- **Option 2**: Another VQA dataset
- **Option 3**: Another captioning dataset

### Additional Model (Optional)

If another MoE-VLM is available:
- Test same controller logic
- Report: Generalization capability

### Evaluation

1. **Reuse controller**: Use controller trained on primary benchmarks
2. **Minimal recalibration**: Recalibrate latency estimator with small sample (10-50 points)
3. **Evaluate**: Quality, latency, SLO compliance
4. **Compare**: To full model and single-knob baselines

## Key Outputs/Plots

### 1. Quality Retention

**Plot**: Quality on new benchmark
- X-axis: Budget fraction
- Y-axis: Quality (accuracy/score)
- Multiple lines: Full model, Single-knob, FlexVLM
- **Insight**: Generalization capability

### 2. Recalibration Curve

**Plot**: Performance vs recalibration samples
- X-axis: Number of recalibration samples
- Y-axis: Quality retention (%)
- **Insight**: How many samples needed for good generalization

## Expected Findings

1. **Generalizes well**: Quality retention > 90% with small recalibration
2. **Key insights hold**: Stage decomposition, knob coupling, etc. still valid
3. **Small recalibration sufficient**: 25-50 samples enough

## Code References

- **Controller**: E5 implementations
- **Recalibration**: E3 methods

## Output Files

- `m6_generalization_results.json`: Generalization results
- `figures/m6_quality_retention.png`: Quality on new benchmark

