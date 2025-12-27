# E2: Knob Coupling + Pareto-Front Structure Across Workloads

## Goal

Demonstrate that the three knobs are coupled and that "best" configurations differ by workload/content/budget—making single-knob policies insufficient.

## Key Questions

1. How do the three knobs interact? Are they independent or coupled?
2. Do optimal configurations differ across workloads (QA vs captioning)?
3. Does fixing one knob change the Pareto frontier over the other two?
4. Can single-knob policies achieve the same performance as multi-knob?

## What to Run

### Knob Grid Definition

Define a comprehensive knob grid:

1. **Input sequence length** (via image resolution / vision tokens)
   - Values: Target vision tokens = [288, 432, 576, 720, 1008, 1440, 1872]
   - Corresponds to: [1, 2, 3, 4, 6, 9, 12] crops
   - Implementation: Resize images to achieve target vision tokens

2. **Model width** (via MoE top-K)
   - Values: top_k = [1, 2, 4, 8, 16, 32]
   - Implementation: Dynamic top-K adjustment

3. **Model depth** (via transformer block masks)
   - Values: Active blocks = [8, 10, 12, 14, 16, 18, 20, 24] (for 24-block model)
   - Strategy: Importance score-based selection (top-K by importance)
   - Implementation: BlockMaskWrapper with importance scores

### Configuration Space

Total configurations: 7 × 6 × 8 = 336 configurations per workload

**Note**: Use sparse sampling for initial exploration, then dense sampling around interesting regions.

### Workloads

Test across at least two workload types:

1. **QA-like tasks** (prefill-sensitive)
   - Dataset: VQA v2 validation
   - Characteristics: Short answers, prefill-dominated
   - Quality metric: VQA accuracy (soft score)

2. **Captioning/long-form tasks** (decode-sensitive)
   - Dataset: COCO Captions validation
   - Characteristics: Long outputs, decode-dominated
   - Quality metric: CIDEr, BLEU-4, ROUGE-L

### Measurements per Configuration

For each (vision_tokens, top_k, active_blocks) configuration:
- **Task quality**: Accuracy/score on validation set
- **Latency**: 
  - Mean latency (ms)
  - Tail latency: P95, P99 (ms)
  - Per-stage breakdown (from E1)

### Coupling Analysis

To demonstrate coupling:

1. **Fix vision_tokens**, compute Pareto frontier over (top_k, active_blocks)
2. **Fix top_k**, compute Pareto frontier over (vision_tokens, active_blocks)
3. **Fix active_blocks**, compute Pareto frontier over (vision_tokens, top_k)
4. Compare frontiers: Show they change materially when fixed knob changes

## Key Outputs/Plots

### 1. Per-Workload Latency–Quality Pareto Frontiers

**Plot**: Scatter plot with Pareto frontier overlay
- X-axis: Latency (P95, ms)
- Y-axis: Quality (accuracy/score)
- Color/Shape: Different knob combinations
- **Pareto frontier**: Highlighted optimal configurations
- **Separate plots**: One per workload (QA vs Captioning)

**Insight**: Optimal configurations differ between workloads

### 2. Coupling Proof Plots

**Plot 1**: Pareto frontiers with fixed vision_tokens
- Multiple subplots: One per fixed vision_tokens value
- Each subplot: Pareto frontier over (top_k, active_blocks)
- **Insight**: Frontier shape changes when vision_tokens changes

**Plot 2**: Pareto frontiers with fixed top_k
- Multiple subplots: One per fixed top_k value
- Each subplot: Pareto frontier over (vision_tokens, active_blocks)
- **Insight**: Frontier shape changes when top_k changes

**Plot 3**: Pareto frontiers with fixed active_blocks
- Multiple subplots: One per fixed active_blocks value
- Each subplot: Pareto frontier over (vision_tokens, top_k)
- **Insight**: Frontier shape changes when active_blocks changes

### 3. Illustrative Cases

**Case studies**: Show specific examples where:
- Same knob setting is Pareto-optimal for one workload but suboptimal for another
- Single-knob policy fails compared to multi-knob
- Coupling creates non-obvious optimal configurations

**Example visualization**:
- Side-by-side comparison: Same configuration on different workloads
- Annotate: "Optimal for QA" vs "Suboptimal for Captioning"

## Implementation Details

### Sparse Sampling Strategy

For initial exploration, use stratified sampling:
- Sample uniformly across knob ranges
- Focus on boundary regions (low/high values)
- Sample around expected optimal regions

### Dense Sampling

After initial exploration:
- Identify interesting regions
- Dense sampling around Pareto frontier
- Validate frontier shape

### Importance Score Calculation

For transformer block selection:
1. Compute importance scores using accuracy drop method (from `transformer_blocks_knob.md`)
2. Select top-K blocks by importance
3. Apply BlockMaskWrapper

### Pareto Frontier Computation

```python
def compute_pareto_frontier(configs, latency_key='latency_p95', quality_key='quality'):
    """
    Compute Pareto frontier from configuration results.
    
    Args:
        configs: List of dicts with latency and quality metrics
    
    Returns:
        pareto_configs: List of Pareto-optimal configurations
    """
    # Sort by latency (ascending)
    sorted_configs = sorted(configs, key=lambda x: x[latency_key])
    
    pareto_configs = []
    best_quality = -float('inf')
    
    for config in sorted_configs:
        if config[quality_key] > best_quality:
            pareto_configs.append(config)
            best_quality = config[quality_key]
    
    return pareto_configs
```

## Expected Findings

1. **Knobs are coupled**: Optimal (top_k, active_blocks) depends on vision_tokens
2. **Workload-dependent**: QA optimal configs differ from Captioning optimal configs
3. **Single-knob insufficient**: Multi-knob policies outperform single-knob
4. **Non-linear interactions**: Optimal configs not at knob extremes

## Code References

- **Vision tokens control**: `docs/knobs/vision_tokens_knob.md`
- **MoE top-K control**: `docs/knobs/moe_topk_knob.md`
- **Transformer blocks control**: `docs/knobs/transformer_blocks_knob.md`
- **Pareto computation**: `experiments/profiling/utils/compare_pareto_frontiers.py`

## Related Experiments

- **E1**: Provides stage decomposition for understanding knob effects
- **E4**: Uses Pareto frontiers for system evaluation
- **E6**: Ablates individual knobs to isolate effects

## Output Files

- `e2_knob_grid_results.json`: All configuration results
- `e2_pareto_frontiers.json`: Computed Pareto frontiers per workload
- `e2_coupling_analysis.json`: Coupling analysis results
- `figures/e2_pareto_qa.png`: QA workload Pareto frontier
- `figures/e2_pareto_captioning.png`: Captioning workload Pareto frontier
- `figures/e2_coupling_proof.png`: Coupling proof plots
- `figures/e2_illustrative_cases.png`: Case study visualizations

