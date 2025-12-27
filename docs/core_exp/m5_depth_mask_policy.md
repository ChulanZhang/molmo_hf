# M5: Depth Mask Policy Sanity Checks

## Goal

Prefix vs importance vs random masks; connect results to "depth is not just a scalar".

## Key Questions

1. Does importance-based masking outperform prefix masking?
2. Does random masking perform worse?
3. How much does block selection matter?
4. Does this support "depth is not just a scalar"?

## What to Run

### Mask Strategies

1. **Prefix blocks**: First N blocks (0 to N-1)
2. **Importance-based**: Top-N blocks by importance score
3. **Random blocks**: N randomly selected blocks

### Evaluation

For each strategy and each number of active blocks:
- Measure: Quality, Latency
- Compare: Strategies at same number of active blocks

## Key Outputs/Plots

### 1. Quality vs Active Blocks

**Plot**: Quality for different mask strategies
- X-axis: Number of active blocks
- Y-axis: Quality (accuracy/score)
- Multiple lines: Prefix, Importance-based, Random
- **Insight**: Importance-based > prefix > random

### 2. Latency vs Active Blocks

**Plot**: Latency for different strategies
- X-axis: Number of active blocks
- Y-axis: Latency (ms)
- Multiple lines: Prefix, Importance-based, Random
- **Insight**: Similar latency (same number of blocks)

### 3. Quality-Latency Tradeoff

**Plot**: Quality vs latency
- X-axis: Latency (ms)
- Y-axis: Quality (accuracy/score)
- Multiple series: Prefix, Importance-based, Random
- **Insight**: Importance-based achieves better frontier

## Expected Findings

1. **Importance-based best**: Highest quality at same latency
2. **Prefix better than random**: Sequential blocks more important
3. **Depth is not scalar**: Which blocks matter, not just how many
4. **Significant difference**: Importance-based can be 5-10% better quality

## Code References

- **Block masking**: `docs/knobs/transformer_blocks_knob.md`
- **Importance scores**: Accuracy drop method

## Output Files

- `m5_mask_strategy_comparison.json`: Strategy comparison results
- `figures/m5_quality_vs_blocks.png`: Quality comparison
- `figures/m5_quality_latency_tradeoff.png`: Frontier comparison

