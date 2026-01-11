# GRPO 正确实现方案

## 核心问题

当前实现：每个sample只采样一次，然后按`(sample_id, latency_budget)`分组。如果每个sample的budget都不同，就不会形成groups。

## 正确的GRPO实现

### 理论依据

GRPO的核心思想是：**对同一个状态（sample）采样多个动作（configs），在组内进行相对比较**。

### 实现步骤

1. **对每个sample，采样`group_size`次**：
   - 对sample i，重复采样`group_size`次（例如5次）
   - 每次采样得到不同的config：`(tier, insertion_pos, top_k, num_blocks)`
   - 这样就有`batch_size * group_size`个configs

2. **执行所有configs**：
   - 对每个config，运行模型，得到`(accuracy, latency)`
   - 计算reward：`reward = f(accuracy, latency, budget)`
   - 总共得到`batch_size * group_size`个rewards

3. **按sample分组，进行相对排序**：
   - 对每个sample，有`group_size`个configs和rewards
   - 对这`group_size`个configs按reward排序
   - 计算pairwise损失：鼓励高reward的config有更高的log_prob

### 损失函数

```python
for each sample in batch:
    # sample有group_size个configs
    group_rewards = [reward_1, reward_2, ..., reward_group_size]
    group_log_probs = [log_prob_1, log_prob_2, ..., log_prob_group_size]
    
    # 按reward排序
    sorted_indices = argsort(group_rewards, descending=True)
    sorted_log_probs = group_log_probs[sorted_indices]
    
    # Pairwise损失
    for i in range(group_size):
        for j in range(i+1, group_size):
            # reward[i] > reward[j] (因为已排序)
            # 希望 log_prob[i] > log_prob[j]
            loss += -log(sigmoid(sorted_log_probs[i] - sorted_log_probs[j]))
```

### 关键变化

1. **采样次数**：从`batch_size`次增加到`batch_size * group_size`次
2. **执行次数**：从`batch_size`次增加到`batch_size * group_size`次
3. **分组方式**：从按`(sample_id, budget)`分组改为按`sample_id`分组
4. **损失计算**：对每个sample的`group_size`个configs进行相对排序

### 性能影响

- **计算量增加**：从`batch_size`次模型执行增加到`batch_size * group_size`次
- **内存增加**：需要存储`batch_size * group_size`个logits、actions、rewards
- **训练时间**：大约增加`group_size`倍（例如5倍）

### 优化建议

1. **减少batch_size**：如果计算资源有限，可以减少batch_size
2. **并行化**：如果有多GPU，可以并行执行不同samples的configs
3. **梯度累积**：如果内存不足，可以使用梯度累积

