# Reward Function设计原理详解

## 当前实现

```python
reward = accuracy_reward 
       - latency_penalty 
       - budget_violation_penalty  # 硬约束
       - complexity_penalty 
       + efficiency_bonus
```

## 各项详细解释

### 1. Accuracy Reward (accuracy_reward)

```python
relative_accuracy = accuracy  # 或相对于baseline的相对准确率
accuracy_reward = alpha * relative_accuracy
```

**原理**：
- **目标**：最大化模型输出质量
- **alpha = 1.0**：准确率的权重
- **作用**：准确率越高，reward越高
- **范围**：通常accuracy在[0, 1]之间，所以reward贡献在[0, 1.0]之间

**为什么需要**：
- 这是主要优化目标：在满足latency约束的前提下最大化accuracy

---

### 2. Latency Penalty (latency_penalty)

```python
latency_ratio = latency / latency_budget
latency_penalty = beta * torch.sigmoid(10.0 * (latency_ratio - 1.0))
```

**原理**：
- **目标**：鼓励latency接近budget（不要过度保守）
- **beta = 0.5**：latency penalty的权重
- **Sigmoid函数**：平滑的惩罚，当latency接近budget时惩罚较小，超过budget时惩罚增大
- **10.0**：sigmoid的陡峭程度，值越大，惩罚曲线越陡

**行为**：
- latency << budget: penalty ≈ 0（不惩罚）
- latency ≈ budget: penalty ≈ 0.5 * 0.5 = 0.25（中等惩罚）
- latency >> budget: penalty ≈ 0.5（最大惩罚）

**为什么需要**：
- 防止过度保守：如果只考虑budget violation，模型可能会选择非常保守的配置（latency远小于budget）
- 鼓励充分利用budget：在满足约束的前提下，使用更多资源（更高accuracy）

---

### 3. Budget Violation Penalty (budget_violation_penalty) - **硬约束**

```python
budget_violation = max(0, latency - latency_budget)
budget_violation_penalty = gamma * (budget_violation / latency_budget) ** 2
```

**原理**：
- **目标**：**硬约束** - 严格惩罚超过budget的配置
- **gamma = 10.0**：非常大的权重，确保这是硬约束
- **平方项**：超过budget越多，惩罚越大（二次增长）
- **归一化**：除以budget，使得惩罚与budget大小无关

**行为**：
- latency <= budget: penalty = 0（不惩罚）
- latency > budget: penalty = 10.0 * (violation/budget)^2（严重惩罚）

**示例**：
- 如果budget=200ms，latency=220ms（超过10%）
  - violation = 20ms
  - penalty = 10.0 * (20/200)^2 = 10.0 * 0.01 = 0.1
- 如果budget=200ms，latency=300ms（超过50%）
  - violation = 100ms
  - penalty = 10.0 * (100/200)^2 = 10.0 * 0.25 = 2.5

**为什么需要**：
- **硬约束**：这是必须满足的条件
- **大权重**：确保即使accuracy很高，超过budget的配置也会被严重惩罚
- **二次增长**：防止模型"稍微超过一点budget"来换取accuracy

---

### 4. Complexity Penalty (complexity_penalty)

```python
max_crops_norm = max_crops / 12.0
top_k_norm = top_k / 32.0
blocks_norm = num_active_blocks / 16.0
complexity = (max_crops_norm + top_k_norm + blocks_norm) / 3.0
complexity_penalty = epsilon * complexity
```

**原理**：
- **目标**：轻微鼓励使用更简单的配置（即使accuracy相同）
- **epsilon = 0.05**：很小的权重
- **归一化**：将各个knob的值归一化到[0, 1]
- **平均**：三个knob的平均复杂度

**行为**：
- 简单配置（low tier, small top_k, few blocks）: penalty ≈ 0.05 * 0.3 = 0.015
- 复杂配置（high tier, large top_k, many blocks）: penalty ≈ 0.05 * 1.0 = 0.05

**为什么需要**：
- **Tie-breaking**：当两个配置有相同的accuracy和latency时，选择更简单的
- **轻微偏好**：不会显著影响主要优化目标，但有助于选择更高效的配置

---

### 5. Efficiency Bonus (efficiency_bonus)

```python
efficiency_bonus = delta * (1.0 - latency / latency_budget)  # if latency <= budget
efficiency_bonus = 0  # if latency > budget
```

**原理**：
- **目标**：奖励在满足budget的前提下，latency更小的配置
- **delta = 0.1**：较小的权重
- **线性奖励**：latency越小（相对于budget），bonus越大

**行为**：
- latency = budget: bonus = 0
- latency = 0.5 * budget: bonus = 0.1 * 0.5 = 0.05
- latency = 0.1 * budget: bonus = 0.1 * 0.9 = 0.09

**为什么需要**：
- **鼓励效率**：在满足accuracy和budget的前提下，选择更高效的配置
- **与latency_penalty配合**：latency_penalty防止过度保守，efficiency_bonus鼓励效率

---

## 整体设计理念

### 优先级排序

1. **Budget Violation (硬约束)** - 最高优先级
   - 超过budget的配置必须被严重惩罚
   - gamma = 10.0 确保这一点

2. **Accuracy (主要目标)** - 第二优先级
   - 在满足budget的前提下，最大化accuracy
   - alpha = 1.0

3. **Latency Efficiency (辅助目标)** - 第三优先级
   - latency_penalty: 防止过度保守
   - efficiency_bonus: 鼓励效率
   - complexity_penalty: 轻微偏好简单配置

### Reward范围估计

假设：
- accuracy: [0.5, 0.9]
- latency_ratio: [0.5, 1.2]（可能稍微超过budget）
- budget_violation: [0, 0.2]（最多超过20%）
- complexity: [0.3, 1.0]
- efficiency: [0, 0.5]

Reward范围：
- Best case: 0.9 - 0.0 - 0.0 - 0.015 + 0.05 = **0.935**
- Worst case (within budget): 0.5 - 0.5 - 0.0 - 0.05 + 0.0 = **-0.05**
- Worst case (over budget): 0.5 - 0.5 - 2.5 - 0.05 + 0.0 = **-2.55**

**关键观察**：
- Budget violation penalty (最大可达2.5) 远大于其他项
- 这确保了超过budget的配置会被严重惩罚

---

## 简化版本（如果只需要accuracy和latency constraint）

如果你只想保留accuracy和latency constraint，可以简化为：

```python
def simplified_reward(accuracy, latency, latency_budget):
    # Accuracy reward
    accuracy_reward = accuracy
    
    # Budget violation penalty (hard constraint)
    budget_violation = max(0, latency - latency_budget)
    budget_violation_penalty = 10.0 * (budget_violation / latency_budget) ** 2
    
    reward = accuracy_reward - budget_violation_penalty
    return reward
```

**优点**：
- 更简单，更容易理解
- 专注于核心目标：accuracy + budget constraint

**缺点**：
- 可能过度保守（选择latency远小于budget的配置）
- 可能选择不必要的复杂配置（如果accuracy相同）

---

## 建议

**当前实现已经很好**，但如果你想要更简单的版本，可以：
1. 保留：accuracy_reward, budget_violation_penalty（必需）
2. 可选保留：latency_penalty（防止过度保守）
3. 可以移除：complexity_penalty, efficiency_bonus（如果觉得太复杂）

**推荐配置**：
```python
RewardFunction(
    alpha=1.0,      # accuracy weight (必需)
    beta=0.5,       # latency penalty (可选，防止过度保守)
    gamma=10.0,     # budget violation penalty (必需，硬约束)
    delta=0.0,      # efficiency bonus (可选，可以设为0)
    epsilon=0.0,    # complexity penalty (可选，可以设为0)
)
```

