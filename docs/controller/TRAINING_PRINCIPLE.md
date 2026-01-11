# Controller 训练原理详解

## 1. 整体架构

### 两阶段控制器

```
Input (Image + Prompt + Latency Budget)
    ↓
[Stage 1: Knob1 Predictor]
    ↓
Knob1: Vision Token Tier (low/medium/high)
    ↓
[Vision Encoder with Tier]
    ↓
[Stage 2: Knob2 & Knob3 Predictor] (after first transformer block)
    ↓
Knob2: MoE Top-K (4,5,6,7,8)
Knob3: Number of Blocks (12,13,14,15,16)
    ↓
[LLM Forward with Knobs]
    ↓
Output (Generated Text)
    ↓
[Compute Reward: Accuracy + Latency Penalty]
```

## 2. 训练方法：GRPO (Group Relative Policy Optimization)

### 核心思想

GRPO是一种**相对排序**的强化学习方法，不需要绝对奖励值，只需要知道**哪个配置更好**。

### 为什么用GRPO？

1. **奖励信号稀疏**：准确率可能很低（0-1之间），直接优化困难
2. **相对比较更容易**：即使所有配置的准确率都很低，我们仍然可以比较哪个更好
3. **稳定性**：不需要估计baseline，减少方差

### GRPO工作流程

#### Step 1: 采样动作（Actions）

对于每个样本，控制器预测：
- **Stage1**: `P(knob1 | language_feat, budget_feat)` → 采样 tier
- **Stage2**: `P(knob2, knob3 | vision_feat, language_feat, budget_feat)` → 采样 top_k 和 num_blocks

```python
# Stage1: 预测tier
knob1_logits = knob1_predictor(lang_feat, budget_feat)  # (B, 3)
knob1_probs = softmax(knob1_logits)
knob1_action = multinomial(knob1_probs)  # 采样: 0=low, 1=medium, 2=high

# Stage2: 预测top_k和num_blocks
knob2_logits, knob3_logits = knob2_knob3_predictor(vision_feat, lang_feat, budget_feat)
knob2_action = multinomial(softmax(knob2_logits))  # 采样: 0-4 → 4,5,6,7,8
knob3_action = multinomial(softmax(knob3_logits))  # 采样: 0-4 → 12,13,14,15,16
```

#### Step 2: 执行模型并计算奖励

对每个样本，使用采样的配置运行模型：

```python
for each sample in batch:
    # 应用knobs
    tier = knob1_values[knob1_action]
    top_k = knob2_values[knob2_action]
    num_blocks = knob3_values[knob3_action]
    
    # 运行模型
    output = model.generate(input, tier=tier, top_k=top_k, num_blocks=num_blocks)
    
    # 计算奖励
    accuracy = compute_accuracy(output, ground_truth)
    latency = estimate_latency(tier, top_k, num_blocks)
    reward = reward_fn(accuracy, latency, budget)
```

**奖励函数**：
```python
reward = alpha * accuracy                    # 准确率奖励
        - beta * latency                     # 延迟惩罚
        - gamma * max(0, latency - budget)  # 预算违反惩罚
        + delta * efficiency_bonus          # 效率奖励
        - epsilon * complexity_penalty       # 复杂度惩罚
```

#### Step 3: 分组（Grouping）

将batch中的样本按 `(sample_id, latency_budget)` 分组：

```python
groups = form_groups(batch)  # 相同sample_id和budget的样本在一组
# 例如: group = [sample_1_config_A, sample_1_config_B, sample_1_config_C, ...]
```

**为什么分组？**
- 相同样本的不同配置可以公平比较
- 相同budget下的配置可以比较效率

#### Step 4: 计算相对排序损失

对每个group内的样本，按reward排序，然后计算pairwise损失：

```python
for group in groups:
    # 按reward降序排序
    sorted_indices = argsort(group_rewards, descending=True)
    sorted_log_probs = group_log_probs[sorted_indices]
    sorted_rewards = group_rewards[sorted_indices]
    
    # 对每对(i, j)，其中i的reward > j的reward
    for i in range(group_size):
        for j in range(i+1, group_size):
            log_prob_diff = sorted_log_probs[i] - sorted_log_probs[j]
            reward_diff = sorted_rewards[i] - sorted_rewards[j]
            
            # 损失：鼓励高reward的配置有更高的log_prob
            loss = -log(sigmoid(log_prob_diff * sign(reward_diff)))
```

**损失函数解释**：
- 如果 `reward[i] > reward[j]`，我们希望 `log_prob[i] > log_prob[j]`
- `log(sigmoid(x))` 确保梯度方向正确
- 使用相对比较，不需要绝对reward值

#### Step 5: 反向传播

```python
loss.backward()
optimizer.step()
```

## 3. 为什么Accuracy是0？

可能的原因：

1. **Metadata格式问题**：
   - `metadata` 中可能没有 `answers` 字段
   - 需要检查数据加载时metadata的格式

2. **模型生成问题**：
   - 生成的文本可能格式不对
   - VQA score计算可能失败

3. **训练初期正常**：
   - 模型刚开始训练，生成质量很差
   - 需要更多训练才能提升

**调试方法**：
- 检查metadata中是否有answers
- 打印生成的文本和ground truth
- 检查VQA score计算是否正常

## 4. 为什么Latency很高？

1. **Latency Estimator可能不准确**：
   - Estimator是在profiling数据上训练的
   - 可能对某些配置估计不准确

2. **配置选择问题**：
   - Controller可能选择了高latency的配置（high tier + 16 blocks + top_k=8）
   - 训练初期，controller是随机选择

3. **实际测量vs估计**：
   - Training用estimator（快速）
   - Validation用实际测量（准确但慢）

## 5. 训练速度优化

### 当前瓶颈

1. **每个样本都要运行模型**：
   - 这是online training的代价
   - 无法避免，因为需要实际accuracy和latency

2. **Batch size小**：
   - 当前默认32，可以增加到64-128（H100有80GB）
   - 增加batch size可以：
     - 减少forward pass次数
     - 提高GPU利用率
     - 加速训练

3. **Sequential execution**：
   - 当前是逐个样本执行（for loop）
   - 可以尝试batch execution（但需要处理不同配置）

### 优化建议

1. **增加batch size**：
   ```python
   batch_size = 128  # 从32增加到128
   ```

2. **增加数据量**：
   ```python
   num_samples = 5000  # 从100增加到5000
   ```

3. **使用gradient accumulation**：
   - 如果batch size受限于内存，可以用gradient accumulation模拟更大的batch

4. **并行化**：
   - 如果有多GPU，可以用数据并行
   - 但controller模型很小，收益有限

## 6. 训练流程总结

```
Epoch Loop:
    For each batch:
        1. Extract features (language, budget)
        2. Stage1: Predict and sample knob1 (tier)
        3. Process images with tier
        4. Stage2: Predict and sample knob2, knob3
        5. For each sample in batch:
           - Execute model with sampled knobs
           - Compute accuracy and latency
           - Compute reward
        6. Group samples by (sample_id, budget)
        7. Compute GRPO loss (relative ranking)
        8. Backward and update
```

## 7. 关键超参数

- **batch_size**: 32 → 64-128（H100可以更大）
- **group_size**: 5（每组至少2个样本才能比较）
- **learning_rate**: 1e-4（可以尝试1e-3或5e-5）
- **num_samples**: 100 → 5000（更多数据）
- **use_latency_estimator**: True（training用estimator，validation可选实际测量）

