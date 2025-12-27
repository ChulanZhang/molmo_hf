# E5: Training Strategy Comparison (Controller Learning)

## Goal

Establish why our training choice is justified empirically (stability, sample-efficiency, overhead) in this discrete combinatorial space.

## Key Questions

1. Which training method is most sample-efficient?
2. Which method is most stable across seeds?
3. What is the overhead cost of each method?
4. How do final frontiers compare across methods?

## What to Run

### Training Methods to Compare

1. **Training-free LUT / Heuristic** (baseline)
   - Pre-computed lookup table from profiling
   - Simple heuristic rules
   - **Overhead**: Minimal (table lookup)
   - **Sample efficiency**: N/A (no training)

2. **Supervised Learning** (if oracle available)
   - Target: Oracle optimal configuration (from E2 Pareto frontiers)
   - Input: Request features (image, prompt, task_type)
   - Output: Optimal configuration
   - **Overhead**: Training time only
   - **Sample efficiency**: Depends on model complexity

3. **PPO-style Online RL** (critic-based)
   - Actor-critic architecture
   - Online policy updates
   - **Overhead**: Critic model training, value estimation
   - **Sample efficiency**: May require many samples

4. **DPO-style Preference Learning** (pairwise)
   - Learn from preference pairs
   - No explicit reward model
   - **Overhead**: Pair generation, preference labeling
   - **Sample efficiency**: Depends on pair quality

5. **GRPO** (critic-free group-relative)
   - Group-relative policy optimization
   - No critic model
   - **Overhead**: Group generation, relative ranking
   - **Sample efficiency**: Potentially high (no critic)

### Common Setup

All methods use:
- **Same observation space**: Request features (image, prompt, task_type, budget)
- **Same action space**: (vision_tokens, top_k, active_blocks)
- **Same latency measurement**: From E1/E2 profiling
- **Same reward definition**: Quality - λ × max(0, latency - budget)
- **Same evaluation budgets**: [0.35, 0.50, 0.70, 0.85, 1.00] × baseline

### Training Protocol

1. **Profiling budget**: Same for all methods (or report explicitly)
   - Collect profiling data: ~200-500 configurations
   - Measure latency and quality for each

2. **Training data collection**:
   - **LUT**: Use all profiling data
   - **Supervised**: Use profiling data with oracle labels
   - **PPO/DPO/GRPO**: Collect online samples during training

3. **Training iterations**:
   - Track: utility, quality, latency, SLO violation rate
   - Evaluate on held-out validation set periodically
   - Stop when convergence or max iterations

4. **Multiple seeds**: Run each method with 3-5 different seeds
   - Report: Mean and std across seeds
   - Assess: Stability

### Metrics to Track

**During training**:
- `env_steps`: Number of executed configurations (real profiling calls)
- `time_to_threshold`: Wall-clock training time
- `utility_curve`: Average reward over steps
- `stability`: Reward variance across steps
- `num_failed_updates`: Number of failed training updates

**Final evaluation**:
- `quality_mean`: Average quality on validation set
- `latency_p95`: P95 latency
- `slo_violation_rate`: Fraction violating budget
- `controller_overhead_ms`: Decision time per request

## Key Outputs/Plots

### 1. Learning Curves

**Plot 1**: Utility vs training steps
- X-axis: Training steps / env_steps
- Y-axis: Average reward / utility
- Multiple lines: One per method
- Error bars: Std across seeds
- **Insight**: Sample efficiency and convergence speed

**Plot 2**: SLO violation rate vs training steps
- X-axis: Training steps
- Y-axis: SLO violation rate (%)
- Multiple lines: One per method
- **Insight**: How fast methods learn to satisfy budgets

**Plot 3**: Quality vs training steps
- X-axis: Training steps
- Y-axis: Quality (accuracy/score)
- Multiple lines: One per method
- **Insight**: Quality improvement over training

### 2. Sample Efficiency Table

**Table**: Steps to reach target performance
- Rows: Methods
- Columns: Target thresholds (e.g., SLO violation ≤ 5%, quality ≥ 90% of full model)
- Values: Number of env_steps required
- **Insight**: Which method is most sample-efficient

### 3. Final Frontiers Comparison

**Plot**: Latency-quality frontiers
- X-axis: Latency (P95, ms)
- Y-axis: Quality (accuracy/score)
- Multiple series: One per method (after training)
- **Insight**: Final performance comparison

### 4. Stability Analysis

**Plot**: Reward variance vs training steps
- X-axis: Training steps
- Y-axis: Reward std (across seeds or within method)
- Multiple lines: One per method
- **Insight**: Training stability

### 5. Overhead Accounting

**Table**: Overhead breakdown
- Rows: Methods
- Columns: 
  - Critic training time (PPO)
  - Pair generation time (DPO)
  - Group generation time (GRPO)
  - Total training overhead
- **Insight**: Cost of each method

## Implementation Details

### LUT Baseline

```python
class LUTController:
    def __init__(self, profiling_data, budget_ms):
        # Build lookup table from profiling data
        self.lut = self._build_lut(profiling_data, budget_ms)
    
    def select_config(self, request_features):
        # Simple lookup or nearest neighbor
        task_type = request_features['task_type']
        return self.lut.get((task_type, self.budget_ms), default_config)
```

### Supervised Learning

```python
class SupervisedController:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # (vision_tokens_idx, top_k_idx, active_blocks_idx)
        )
    
    def train(self, training_data):
        # training_data: (features, oracle_config) pairs
        # Train with cross-entropy or MSE loss
        pass
```

### PPO Implementation

```python
class PPOController:
    def __init__(self):
        self.actor = PolicyNetwork()
        self.critic = ValueNetwork()
    
    def train_step(self, batch):
        # PPO update with actor-critic
        # Requires: states, actions, rewards, values
        pass
```

### DPO Implementation

```python
class DPOController:
    def __init__(self):
        self.policy = PolicyNetwork()
    
    def train_step(self, preference_pairs):
        # DPO update from preference pairs
        # Requires: (config_A, config_B, preference) pairs
        pass
```

### GRPO Implementation

```python
class GRPOController:
    def __init__(self):
        self.policy = PolicyNetwork()
    
    def train_step(self, groups):
        # GRPO update from groups
        # Requires: Groups of configs with relative rankings
        pass
```

## Expected Findings

1. **GRPO is sample-efficient**: Fewer samples than PPO/DPO
2. **GRPO is stable**: Lower variance than PPO
3. **GRPO has low overhead**: No critic training
4. **LUT is fast but limited**: Good baseline but can't adapt
5. **Supervised works if oracle available**: But requires oracle labels

## Code References

- **Training frameworks**: To be implemented
- **E2 results**: For oracle configurations (supervised learning)
- **E3 estimator**: For filtering and guardrails

## Related Experiments

- **E2**: Provides Pareto frontiers and oracle configurations
- **E4**: Uses trained controllers for system evaluation
- **E6**: Ablates training components

## Output Files

- `e5_training_curves.json`: Learning curves for all methods
- `e5_sample_efficiency.json`: Steps to reach thresholds
- `e5_final_frontiers.json`: Final performance comparison
- `e5_overhead_analysis.json`: Overhead measurements
- `e5_stability_analysis.json`: Variance analysis
- `figures/e5_learning_curves.png`: Utility over steps
- `figures/e5_sample_efficiency.png`: Steps to threshold
- `figures/e5_final_frontiers.png`: Frontier comparison
- `figures/e5_overhead_comparison.png`: Overhead breakdown

