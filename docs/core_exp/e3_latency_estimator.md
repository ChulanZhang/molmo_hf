# E3: Stage-Aware Latency Estimator + Small-Sample Cross-GPU Recalibration

## Goal

Build a lightweight latency model that predicts wall-clock latency from knob settings by stage, then demonstrate it transfers across GPUs with small recalibration.

## Key Questions

1. Can we predict latency from knob settings without full profiling?
2. How many calibration samples are needed for cross-GPU transfer?
3. Can the estimator effectively filter budget-violating configurations?
4. What is the tradeoff between estimator accuracy and calibration cost?

## What to Run

### Stage-Aware Latency Estimator

Build a model that predicts latency by stage:

**Input features**:
- Vision tokens count
- Prompt length (text tokens)
- MoE top-K value
- Number of active transformer blocks
- Task type (optional: QA vs Captioning)

**Output predictions**:
- T_vision_encoder (ms)
- T_projector (ms)
- T_LLM_prefill (ms)
- T_LLM_decode_per_token (ms)
- T_total (ms) = sum of stages

### Training on Source GPU

1. **Collect training data** on source GPU (e.g., H100):
   - Sample configurations from knob grid (sparse sampling)
   - Measure actual latencies using `measure_inference_latency()`
   - Collect: ~100-200 configurations for training

2. **Model architecture**:
   - Option 1: Linear regression per stage
   - Option 2: Lightweight neural network (2-3 layers)
   - Option 3: Stage-wise polynomial models

3. **Training**:
   - Train separate models for each stage
   - Or train unified model with stage outputs
   - Validation split: 20-30%

### Cross-GPU Recalibration

**Transfer to target GPU** (e.g., A100) with small calibration:

1. **Calibration samples**: N ∈ {10, 25, 50, 100}
   - Uniformly sample across knob space
   - Measure actual latencies on target GPU

2. **Recalibration methods**:

   **Method A: Per-stage affine scaling**
   ```python
   T_stage_target = a_stage * T_stage_source + b_stage
   ```
   - Fit (a_stage, b_stage) per stage using calibration samples
   - Simple, interpretable

   **Method B: Single global scaling**
   ```python
   T_total_target = a * T_total_source + b
   ```
   - Fit (a, b) on total latency
   - Simpler but less accurate

   **Method C: Learned recalibration**
   - Fine-tune source model on calibration samples
   - More flexible but requires more samples

3. **Evaluation**:
   - Test on held-out configurations (50-100 configs)
   - Measure: MAE, MAPE, P95 absolute error

### Operational Use: Budget Guardrail

Use estimator to filter configurations:

1. **Before policy evaluation**:
   - Predict latency for all candidate configurations
   - Filter out configurations predicted to violate budget
   - Evaluate policy only on remaining configurations

2. **During scheduling**:
   - Use estimator as guardrail
   - Reject configurations predicted to exceed budget
   - Fallback to full measurement if uncertain

## Key Outputs/Plots

### 1. Prediction Error vs Calibration Samples

**Plot**: Calibration sample size vs prediction error
- X-axis: Number of calibration samples (N)
- Y-axis: Prediction error (MAE, MAPE, or P95 error)
- Multiple lines:
  - No calibration (direct transfer)
  - With recalibration (Method A, B, C)
- **Insight**: How fast error drops with small calibration

### 2. Cross-GPU Transfer Curve

**Plot**: Error reduction with calibration
- X-axis: Calibration samples
- Y-axis: MAPE or MAE (ms)
- Show: Error reduction from no-calibration to calibrated
- **Insight**: Diminishing returns of more calibration samples

### 3. Prediction Scatter Plots

**Plot**: Predicted vs actual latency
- X-axis: Predicted latency (ms)
- Y-axis: Actual latency (ms)
- Color: Different stages or configurations
- Diagonal line: Perfect prediction
- **Insight**: Model accuracy and bias

### 4. Budget Enforcement Quality

**Plot**: Violation rate with/without estimator
- X-axis: Budget fraction (0.5, 0.7, 0.85, 1.0)
- Y-axis: SLO violation rate
- Multiple lines:
  - No guardrail (baseline)
  - With estimator guardrail
- **Insight**: Estimator effectiveness in preventing violations

### 5. False Reject/False Accept Tradeoff

**Plot**: ROC-style curve (optional)
- X-axis: False reject rate (rejecting valid configs)
- Y-axis: False accept rate (accepting invalid configs)
- Different points: Different estimator thresholds
- **Insight**: Tradeoff between safety and efficiency

## Implementation Details

### Stage-Aware Model

```python
class StageAwareLatencyEstimator:
    def __init__(self):
        self.vision_model = None
        self.projector_model = None
        self.prefill_model = None
        self.decode_model = None
    
    def predict(self, vision_tokens, prompt_tokens, top_k, active_blocks, task_type=None):
        """Predict latency for each stage."""
        features = self._extract_features(vision_tokens, prompt_tokens, top_k, active_blocks, task_type)
        
        t_vision = self.vision_model.predict(features)
        t_projector = self.projector_model.predict(features)
        t_prefill = self.prefill_model.predict(features)
        t_decode_per_token = self.decode_model.predict(features)
        
        return {
            'T_vision_encoder': t_vision,
            'T_projector': t_projector,
            'T_LLM_prefill': t_prefill,
            'T_LLM_decode_per_token': t_decode_per_token,
            'T_total': t_vision + t_projector + t_prefill + t_decode_per_token * num_output_tokens
        }
```

### Recalibration

```python
def recalibrate_affine(source_model, calibration_data, target_gpu_data):
    """
    Recalibrate using per-stage affine scaling.
    
    Args:
        source_model: Trained model on source GPU
        calibration_data: (config, actual_latency) pairs on target GPU
        target_gpu_data: Full target GPU measurements
    
    Returns:
        recalibrated_model: Model with affine scaling parameters
    """
    # Predict on calibration samples using source model
    source_predictions = [source_model.predict(cfg) for cfg, _ in calibration_data]
    target_actuals = [lat for _, lat in calibration_data]
    
    # Fit affine scaling per stage
    scaling_params = {}
    for stage in ['vision', 'projector', 'prefill', 'decode']:
        source_vals = [p[stage] for p in source_predictions]
        target_vals = [a[stage] for a in target_actuals]
        a, b = fit_affine(source_vals, target_vals)
        scaling_params[stage] = (a, b)
    
    return RecalibratedModel(source_model, scaling_params)
```

### Budget Guardrail

```python
def filter_by_budget(estimator, candidate_configs, budget_ms, threshold=1.1):
    """
    Filter configurations predicted to violate budget.
    
    Args:
        estimator: Latency estimator
        candidate_configs: List of (vision_tokens, top_k, active_blocks) tuples
        budget_ms: Latency budget in milliseconds
        threshold: Safety margin (1.1 = 10% margin)
    
    Returns:
        valid_configs: Configurations predicted to satisfy budget
    """
    valid_configs = []
    
    for config in candidate_configs:
        predicted_latency = estimator.predict_total(*config)
        
        # Add safety margin
        if predicted_latency * threshold <= budget_ms:
            valid_configs.append(config)
    
    return valid_configs
```

## Expected Findings

1. **Stage-aware models are accurate**: Better than single total-latency model
2. **Small calibration sufficient**: 25-50 samples enough for good transfer
3. **Affine scaling effective**: Simple recalibration works well
4. **Guardrail reduces violations**: Estimator effectively filters bad configs

## Code References

- **Latency measurement**: `experiments/base_experiment.py::measure_inference_latency()`
- **Stage decomposition**: E1 results
- **Knob control**: `docs/knobs/` documents

## Related Experiments

- **E1**: Provides stage decomposition data for training
- **E2**: Provides knob grid and Pareto frontiers
- **E4**: Uses estimator in end-to-end system evaluation
- **M4**: Detailed analysis of guardrail effectiveness

## Output Files

- `e3_estimator_model.pkl`: Trained estimator model
- `e3_calibration_results.json`: Calibration experiment results
- `e3_transfer_curves.json`: Cross-GPU transfer data
- `e3_guardrail_analysis.json`: Budget enforcement results
- `figures/e3_prediction_error_vs_samples.png`: Calibration curve
- `figures/e3_predicted_vs_actual.png`: Scatter plot
- `figures/e3_guardrail_effectiveness.png`: Violation rate comparison

