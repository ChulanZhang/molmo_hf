# Positioned Decode Latency Training Strategy

## Problem Statement

**Challenge**: Decode per-token latency increases with token position due to KV cache growth:
- Position 1: ~25 ms/token (fastest)
- Position 5: ~35 ms/token
- Position 10: ~41 ms/token
- Position 21+: ~45 ms/token (slowest)

**Current Data Limitation**: 
- Existing data only has **average** decode per-token latency: `T_LLM_decode / output_tokens`
- This average doesn't capture the progressive slowdown
- New profiling experiments with positioned decode latency (`T_decode_per_step`) are running but won't be ready for 2 days

**Question**: How to train a positioned decode latency model using only average latency data?

## Solution: Total Decode Latency as Training Target

### Key Insight

Instead of training on average per-token latency, we train on **total decode latency**:

```
Training Target: T_LLM_decode (total decode latency)
Training Method: Predict positioned latencies and sum them
```

### Training Strategy

1. **Input**: Configuration + `output_tokens` (known during training)
2. **Model Prediction**: Predict decode latency at each position `[1, 2, ..., output_tokens]`
3. **Training Target**: Sum of predicted latencies should equal `T_LLM_decode`
4. **Loss Function**: 
   ```
   loss = 2.0 * loss_prefill + 1.0 * loss_decode_total
   ```
   where `loss_decode_total = MSE(sum(predicted_latencies), T_LLM_decode)`

### Why This Works

1. **Model Learns Position Dependency**: By predicting at all positions and summing, the model must learn that later positions are slower
2. **No Positioned Data Required**: We only need total decode latency, which we have
3. **Future Compatibility**: When positioned data arrives, we can add per-position loss as additional supervision

### Implementation Details

#### Dataset (`LatencyEstimatorDataset`)

```python
{
    'vision_tokens': (B,),
    'text_tokens': (B,),
    'tier_idx': (B,),
    'top_k': (B,),
    'num_active_blocks': (B,),
    'output_tokens': (B,),  # Used during training only
    'T_prefill_total': (B,),
    'T_LLM_decode': (B,),  # Total decode latency (target)
    'T_decode_per_token_avg': (B,),  # Average (for reference)
    'positioned_decode_latencies': List[List[float]],  # Optional: if available
    'has_positioned_data': List[bool],  # Optional: flag
}
```

#### Training Step (`LatencyEstimatorTrainer.train_step`)

1. Predict decode latency at all positions `[1, 2, ..., max(output_tokens)]`
2. For each sample, sum predicted latencies up to its `output_tokens`
3. Compute loss: `MSE(sum(predicted), T_LLM_decode)`
4. Optional: If positioned data available, add per-position loss

#### Model Architecture

- **Config Encoder**: Encodes configuration features (vision_tokens, text_tokens, tier, top_k, blocks)
- **Position Encoder**: Encodes token position (using `log(position + 1)` normalization)
- **Decode Head**: Concatenates config + position encodings, predicts latency at that position

### Training Process

```python
# For each sample with output_tokens = N:
positions = [1, 2, 3, ..., N]
predicted_latencies = model.predict_decode_at_positions(config, positions)  # (N,)
predicted_total = sum(predicted_latencies)  # Scalar
loss = MSE(predicted_total, T_LLM_decode)
```

### Advantages

1. **Uses Existing Data**: No need to wait for positioned data
2. **Learns Position Dependency**: Model must learn progressive slowdown
3. **Future-Proof**: Can easily incorporate positioned data when available
4. **Theoretically Sound**: Total latency is the ground truth we care about

### When Positioned Data Arrives

When `T_decode_per_step` data is available:

1. **Hybrid Training**: 
   - Primary: Total decode latency loss (as before)
   - Secondary: Per-position loss (if positioned data available)
   - Loss: `2.0 * loss_prefill + 1.0 * loss_decode_total + 0.5 * loss_decode_positioned`

2. **Better Supervision**: Per-position loss provides direct supervision on the progressive slowdown pattern

### Inference Usage

During inference (when `output_tokens` is unknown):

1. **Estimate Average Position**: Use expected average position (e.g., 5-10 tokens)
2. **Predict at Multiple Positions**: Predict at positions `[1, 5, 10, 20]` to understand latency curve
3. **Conservative Estimate**: Use maximum predicted latency for budget checking
4. **Integration**: If expected output length distribution is known, integrate over positions

## Example Usage

### Training

```bash
python experiments/controller/train_latency_estimator.py \
    --results_dir results/core_exp_h100/4run_2000samples \
    --use_all_datasets \
    --output_dir checkpoints/latency_estimator \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 1e-3 \
    --device cuda \
    --seed 3407
```

### Inference

```python
# Predict at specific position
latency_at_pos_5 = model(
    vision_tokens=vision_tokens,
    text_tokens=text_tokens,
    tier_idx=tier_idx,
    top_k=top_k,
    num_active_blocks=num_active_blocks,
    token_position=torch.tensor([5.0]),
)['T_decode_per_token']

# Predict at multiple positions
positions = torch.tensor([1.0, 5.0, 10.0, 20.0])
latencies = model.predict_decode_at_positions(
    vision_tokens=vision_tokens,
    text_tokens=text_tokens,
    tier_idx=tier_idx,
    top_k=top_k,
    num_active_blocks=num_active_blocks,
    positions=positions,
)  # Returns latencies at each position
```

## Expected Behavior

After training, the model should learn:
- **Early positions (1-3)**: Lower latency (~25-30 ms/token)
- **Mid positions (5-10)**: Medium latency (~35-40 ms/token)
- **Late positions (20+)**: Higher latency (~45 ms/token)

This progressive slowdown pattern should emerge naturally from the total latency training objective.

## Validation

To verify the model learned position dependency:

1. **Visualization**: Plot predicted latency vs. position for different configurations
2. **Comparison**: Compare predicted positioned latencies with actual `T_decode_per_step` data (when available)
3. **Total Latency Accuracy**: Verify that sum of predicted positioned latencies matches `T_LLM_decode`

## Related Documents

- **[LATENCY_ESTIMATOR_DESIGN.md](LATENCY_ESTIMATOR_DESIGN.md)**: Overall design
- **[DECODE_LATENCY_VS_OUTPUT_TOKENS.md](../analysis/DECODE_LATENCY_VS_OUTPUT_TOKENS.md)**: Analysis of KV cache impact
- **[LATENCY_ESTIMATOR_IMPROVEMENT.md](LATENCY_ESTIMATOR_IMPROVEMENT.md)**: Improvement plan

---

**Last Updated**: 2026-01-08  
**Status**: Implemented and ready for training



