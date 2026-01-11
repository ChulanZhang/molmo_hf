# GRPO实现修复总结

## 修复的问题

### 1. 核心问题
**原实现**：每个sample只采样一次，然后按`(sample_id, latency_budget)`分组。如果每个sample的budget都不同，就不会形成groups。

**正确实现**：对每个sample采样`group_size`次（例如5次），得到`group_size`个不同的configs，然后按`sample_id`分组进行相对排序。

### 2. 具体修复

#### 2.1 采样逻辑
- **原实现**：`batch_size`个samples，每个采样1次 → `batch_size`个configs
- **新实现**：`batch_size`个samples，每个采样`group_size`次 → `batch_size * group_size`个configs

#### 2.2 执行逻辑
- **原实现**：执行`batch_size`次模型forward
- **新实现**：执行`batch_size * group_size`次模型forward

#### 2.3 分组逻辑
- **原实现**：按`(sample_id, latency_budget)`分组
- **新实现**：按`sample_id`分组（每个sample有`group_size`个configs）

#### 2.4 损失计算
- **原实现**：如果无法形成groups，fallback到standard policy gradient
- **新实现**：每个sample都有`group_size`个configs，总是可以形成groups并进行相对排序

### 3. 损失函数

```python
# 对每个sample的group_size个configs：
# 1. 按reward降序排序
sorted_indices = argsort(group_rewards, descending=True)
sorted_log_probs = group_log_probs[sorted_indices]

# 2. 对每对(i, j)，其中i的reward > j的reward
for i in range(group_size):
    for j in range(i+1, group_size):
        # reward[i] > reward[j] (因为已排序)
        # 希望 log_prob[i] > log_prob[j]
        log_prob_diff = sorted_log_probs[i] - sorted_log_probs[j]  # 应该是正数
        loss += -log(sigmoid(log_prob_diff))
```

### 4. 性能影响

- **计算量**：从`batch_size`次增加到`batch_size * group_size`次（例如5倍）
- **内存**：需要存储`batch_size * group_size`个logits、actions、rewards
- **训练时间**：大约增加`group_size`倍

### 5. 理论正确性

✅ **符合GRPO理论**：
- 对同一状态（sample）采样多个动作（configs）
- 在组内进行相对比较
- 使用pairwise损失进行相对排序

### 6. 需要确认的问题

1. **损失函数形式**：当前使用`-log(sigmoid(log_prob_diff))`，其中`log_prob_diff = sorted_log_probs[i] - sorted_log_probs[j]`（i的reward > j）。这是正确的吗？还是应该使用`-log(sigmoid(log_prob_diff * sign(reward_diff)))`？

2. **batch_size调整**：由于计算量增加，可能需要减少batch_size。当前batch_size=1（per device），如果group_size=5，实际执行5次。这是可以接受的吗？

3. **验证逻辑**：`validate`方法是否也需要同样的修复？

