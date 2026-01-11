# GRPO训练中5个配置的生成机制

## 概述

在GRPO（Group Relative Policy Optimization）训练过程中，对于每个原始样本，会生成`group_size`个不同的配置（默认`group_size=5`）。这5个配置用于进行相对排名，从而计算GRPO损失。

## 配置生成流程

### 1. 输入准备

对于每个原始样本（batch_size=1），我们有以下输入：
- `input_ids`: 语言token IDs
- `images`: 图像数据
- `prompt`: 文本提示
- `latency_budget`: 延迟预算（标量）

### 2. Stage1 Controller（Knob1预测）

**输入特征：**
- `lang_feat`: 语言特征（从`wte_layer`提取，维度：`d_model=2048`）
- `budget_feat`: 预算特征（从`LatencyBudgetEncoder`提取，维度：`d_model=2048`）

**输出：**
- `tier_logits`: (3,) - 3个选项的概率分布：`["low", "medium", "high"]`
- `insertion_logits`: (5,) - 5个选项的概率分布：插入位置1-5（即Stage2控制器插入在第1-5个transformer block之后）

**采样过程：**
```python
# 对于每个配置（共5个），都独立采样
tier_probs = F.softmax(tier_logits, dim=-1)  # (5, 3)
insertion_probs = F.softmax(insertion_logits, dim=-1)  # (5, 5)

# 每个配置独立采样
tier_actions = torch.multinomial(tier_probs, 1)  # (5,)
insertion_actions = torch.multinomial(insertion_probs, 1)  # (5,)
```

**结果：**
- 5个配置的`tier`可能不同（例如：`["low", "medium", "high", "medium", "low"]`）
- 5个配置的`insertion_position`可能不同（例如：`[1, 2, 3, 1, 4]`）

### 3. 图像处理（基于Stage1的tier）

对于每个配置，根据其`tier`值处理图像：
- `low`: 最多3个crops
- `medium`: 最多6个crops
- `high`: 最多12个crops

### 4. Stage2 Controller（Knob2 & Knob3预测）

**前置步骤：提取latency token**

对于每个配置，需要：
1. 运行模型的前`insertion_position`个transformer blocks
2. 提取最后一个token（latency token）的hidden state
3. 这个token已经与vision tokens和language tokens进行了attention交互

**输入特征：**
- `latency_token`: (1, d_model=2048) - 从插入位置提取的token
- `insertion_position`: (1,) - 插入位置（1-5）

**输出：**
- `knob2_logits`: (5,) - 5个选项的概率分布：`[4, 5, 6, 7, 8]`（MoE top-K）
- `knob3_logits`: (动态长度) - 动态选项的概率分布，取决于`insertion_position`

**Knob3选项的动态计算：**
```python
def get_knob3_options(self, insertion_position: int) -> List[int]:
    """
    根据插入位置动态计算Knob3选项。
    
    例如，如果insertion_position=1（在第1个block之后插入）：
    - 总blocks = 16
    - 已用blocks = 1（第0个block）
    - 剩余blocks = 15
    - 可选blocks数 = 11, 12, 13, 14, 15（从15个剩余blocks中选择）
    - 总active blocks = 1 + [11, 12, 13, 14, 15] = [12, 13, 14, 15, 16]
    
    如果insertion_position=3（在第3个block之后插入）：
    - 已用blocks = 3
    - 剩余blocks = 13
    - 可选blocks数 = 9, 10, 11, 12, 13
    - 总active blocks = 3 + [9, 10, 11, 12, 13] = [12, 13, 14, 15, 16]
    """
    remaining_blocks = self.total_blocks - insertion_position
    # 从剩余blocks中选择，使得总active blocks在[12, 13, 14, 15, 16]范围内
    # ...
```

**采样过程：**
```python
# 对于每个配置，独立采样
knob2_probs = F.softmax(knob2_logits, dim=-1)  # (5,)
knob3_probs = F.softmax(knob3_logits, dim=-1)  # (动态长度)

knob2_action = torch.multinomial(knob2_probs, 1)  # (1,)
knob3_action = torch.multinomial(knob3_probs, 1)  # (1,)
```

**结果：**
- 5个配置的`top_k`可能不同（例如：`[4, 5, 6, 7, 8]`）
- 5个配置的`num_active_blocks`可能不同（例如：`[12, 13, 14, 15, 16]`）

### 5. 模型执行

对于每个配置，使用其特定的参数执行模型：
```python
result = self._execute_model(
    tier=tiers[i],                    # 例如："medium"
    insertion_position=insertion_positions[i],  # 例如：2
    top_k=top_k_values[i],            # 例如：6
    num_active_blocks=num_active_blocks_values[i],  # 例如：14
    ...
)
```

### 6. 奖励计算

对于每个配置，计算奖励：
```python
reward = self.reward_fn(
    accuracy=accuracy,           # 模型输出的准确率
    latency=prefill_latency,     # 主要使用prefill latency
    latency_budget=latency_budget,
    config={
        'max_crops': tier_to_max_crops[tier],
        'top_k': top_k,
        'num_active_blocks': num_active_blocks,
    },
)
```

### 7. GRPO损失计算

将5个配置按照奖励排序，计算相对排名损失：
```python
# 按奖励降序排序
sorted_indices = torch.argsort(rewards, descending=True)
sorted_log_probs = log_probs[sorted_indices]

# 对于每对配置(i, j)，其中i的奖励 > j的奖励
# 损失 = -log(sigmoid(log_prob[i] - log_prob[j]))
```

## 关键点总结

1. **所有knob都可能不同**：
   - Knob1（tier和insertion_position）：5个配置可能不同
   - Knob2（top_k）：5个配置可能不同
   - Knob3（num_active_blocks）：5个配置可能不同

2. **采样是独立的**：
   - 每个配置都从各自的概率分布中独立采样
   - 这意味着5个配置的多样性取决于controller的当前预测分布

3. **动态性**：
   - Knob3的选项取决于`insertion_position`
   - 如果5个配置的`insertion_position`不同，它们的Knob3选项空间也可能不同

4. **训练目标**：
   - GRPO通过相对排名来学习：奖励高的配置应该被赋予更高的概率
   - 这鼓励controller学习到更好的配置选择策略

## 示例

假设对于某个样本，生成的5个配置如下：

| 配置 | tier | insertion_position | top_k | num_active_blocks | reward |
|------|------|-------------------|-------|------------------|--------|
| 1    | low  | 1                 | 4     | 12               | 0.5    |
| 2    | medium| 2               | 6     | 14               | 0.8    |
| 3    | high | 1                 | 8     | 16               | 0.3    |
| 4    | medium| 3               | 5     | 15               | 0.9    |
| 5    | low  | 2                 | 7     | 13               | 0.6    |

GRPO损失会鼓励：
- 配置4（reward=0.9）的概率 > 配置2（reward=0.8）的概率
- 配置2的概率 > 配置5（reward=0.6）的概率
- 配置5的概率 > 配置1（reward=0.5）的概率
- 配置1的概率 > 配置3（reward=0.3）的概率

这样，controller会逐渐学习到：对于这个样本，`(medium, insertion=3, top_k=5, blocks=15)`是一个更好的配置。

