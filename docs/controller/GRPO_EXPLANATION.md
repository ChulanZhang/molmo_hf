# GRPO (Group Relative Policy Optimization) 详解

## 核心思想

GRPO是一种**相对排序**的强化学习方法，不需要绝对奖励值，只需要知道**哪个配置更好**。

## 正确的GRPO实现

### 对于每个样本（Sample）：

1. **生成多个配置**：
   - 对同一个sample，controller采样`group_size`次（例如5次）
   - 每次采样得到不同的config组合：`(tier, insertion_pos, top_k, num_blocks)`
   - 这样就有`group_size`个不同的configs

2. **执行所有配置**：
   - 对每个config，运行模型，得到`(accuracy, latency)`
   - 计算reward：`reward = f(accuracy, latency, budget)`

3. **相对排序**：
   - 对这`group_size`个configs按reward排序
   - 计算pairwise损失：鼓励高reward的config有更高的log_prob

### 损失函数：

```python
for i in range(group_size):
    for j in range(i+1, group_size):
        if reward[i] > reward[j]:
            # 希望 log_prob[i] > log_prob[j]
            loss += -log(sigmoid(log_prob[i] - log_prob[j]))
```

## 为什么这样设计？

1. **公平比较**：同一个sample的不同configs可以公平比较（相同的输入、相同的ground truth）
2. **相对学习**：不需要知道绝对reward值，只需要知道哪个更好
3. **稳定性**：减少方差，提高训练稳定性

## 当前实现的问题

当前实现按`(sample_id, latency_budget)`分组，但：
- 如果每个sample的budget都是随机生成的，每个`(sample_id, budget)`组合都是唯一的
- 这样就不会形成groups（需要至少2个样本）
- 所以会fallback到standard policy gradient

## 正确的实现方式

应该改为：
- 对每个sample，采样`group_size`次，得到`group_size`个configs
- 执行这些configs，得到`group_size`个rewards
- 然后对这些rewards进行相对排序

