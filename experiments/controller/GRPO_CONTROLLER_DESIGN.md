# GRPO Controller 设计与执行文档

## 1. 项目概述

### 1.1 目标
训练一个智能控制器，根据输入图像和语言特征以及延迟预算，动态选择最优的模型配置（max_crops, top_k, num_active_blocks），在满足延迟约束的前提下最大化准确率。

### 1.2 核心挑战
- **多目标优化**：同时优化准确率和延迟
- **约束满足**：必须满足延迟预算
- **动态决策**：需要根据输入特征实时决策
- **低开销**：控制器本身不能引入过多计算开销

## 2. GRPO原理详解

### 2.1 GRPO核心思想

Group Relative Policy Optimization (GRPO) 是一种基于相对比较的策略优化方法，其核心思想是：

1. **组内比较**：将多个轨迹（trajectories）分组，通过组内轨迹的相对表现来估计优势函数
2. **无价值函数**：不需要学习独立的价值函数（V-function），简化训练过程
3. **稳定优化**：通过相对比较而非绝对奖励，提高训练稳定性

### 2.2 GRPO数学原理

#### 传统策略梯度方法
传统方法（如PPO）使用优势函数：
```
A(s,a) = Q(s,a) - V(s)
```
需要学习价值函数V(s)来估计优势。

#### GRPO方法
GRPO通过组内比较直接估计相对优势：
```
A_relative(s_i, a_i) = R_i - R_group_mean
```
其中：
- `R_i` 是轨迹i的累积奖励
- `R_group_mean` 是组内平均奖励

策略更新：
```
∇θ J(θ) = E[∇θ log π_θ(a|s) · A_relative(s,a)]
```

### 2.3 GRPO的优势

1. **简化训练**：不需要价值网络，减少一半的模型参数
2. **提高稳定性**：相对比较减少方差，训练更稳定
3. **适合离线学习**：可以利用已有的exp5/exp6数据
4. **组内归一化**：自动处理不同延迟预算下的奖励尺度问题

## 3. 系统架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Input Processing                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Image        │  │ Language     │  │ Latency      │ │
│  │ Features     │  │ Features     │  │ Budget       │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Controller (GRPO Policy)                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Feature Encoder (Lightweight)                   │  │
│  │  - Image feature pooling                         │  │
│  │  - Language feature extraction                    │  │
│  │  - Latency budget encoding                       │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Policy Network                                  │  │
│  │  - Input: [image_feat, lang_feat, budget]       │  │
│  │  - Output: Action distribution                   │  │
│  │    (max_crops, top_k, num_active_blocks)         │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Model Execution                             │
│  - Apply selected configuration                         │
│  - Run inference                                         │
│  - Measure latency and accuracy                         │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Reward Computation                          │
│  R = α·accuracy - β·latency_penalty - γ·budget_violation│
└─────────────────────────────────────────────────────────┘
```

### 3.2 Controller位置设计

**推荐方案：Pre-inference Controller**

将Controller放在模型推理之前，作为独立的决策模块：

```
User Input → Controller → Configuration Selection → Model Inference → Output
```

**优点**：
- 低耦合：Controller独立于主模型
- 易训练：可以单独训练和更新
- 易部署：可以灵活替换不同的controller策略

**位置细节**：
1. **特征提取点**：在vision backbone和language encoder之后
   - 图像特征：使用vision backbone的CLS token或pooled features
   - 语言特征：使用language encoder的[CLS] token或mean pooling
   
2. **决策时机**：在模型forward之前
   - 根据特征和预算，选择配置
   - 设置model.config.max_crops, top_k, num_active_blocks
   - 应用block mask

### 3.3 特征提取设计

#### 3.3.1 图像特征提取

**方案1：使用Vision Backbone的CLS Token（推荐）**
```python
# 在vision backbone forward之后
image_features, cls_embed = vision_backbone(images, image_masks)
# cls_embed: (batch_size, num_crops, d_model)
# 使用mean pooling或attention pooling
image_feat = cls_embed.mean(dim=1)  # (batch_size, d_model)
# 或者使用轻量级attention
image_feat = lightweight_attention_pool(cls_embed)  # (batch_size, d_model)
```

**方案2：使用Patch Features的统计信息**
```python
# 提取patch features的统计特征
image_features = vision_backbone(images, image_masks)[0]  # (B, T, N, D)
# 提取统计特征
mean_feat = image_features.mean(dim=(1, 2))  # (B, D)
std_feat = image_features.std(dim=(1, 2))    # (B, D)
max_feat = image_features.max(dim=1)[0].max(dim=1)[0]  # (B, D)
image_feat = torch.cat([mean_feat, std_feat, max_feat], dim=-1)  # (B, 3*D)
```

**推荐**：方案1，使用CLS token，更简洁且信息丰富。

#### 3.3.2 语言特征提取

**方案：使用Language Encoder的Embedding**
```python
# 在tokenizer和embedding之后
input_ids = tokenizer(text)
input_embeds = model.transformer.wte(input_ids)  # (B, L, D)

# 提取语言特征
# 方法1：使用[CLS] token（如果有）
lang_feat = input_embeds[:, 0]  # (B, D)

# 方法2：使用mean pooling
lang_feat = input_embeds.mean(dim=1)  # (B, D)

# 方法3：使用attention pooling（轻量级）
lang_feat = lightweight_attention_pool(input_embeds)  # (B, D)
```

**推荐**：方法2（mean pooling），简单高效。

#### 3.3.3 延迟预算编码

```python
# 延迟预算可以是：
# 1. 绝对时间（秒）：budget = 0.5  # 500ms
# 2. 相对比例：budget = 0.8  # 80% of max latency
# 3. 类别编码：budget = 0/1/2  # low/medium/high

# 编码方式
budget_embed = nn.Linear(1, d_budget)(budget.unsqueeze(-1))  # (B, d_budget)
# 或者使用sinusoidal encoding
budget_embed = sinusoidal_encoding(budget)  # (B, d_budget)
```

### 3.4 Controller网络架构

```python
class GRPOController(nn.Module):
    def __init__(
        self,
        image_feat_dim: int,      # 图像特征维度
        lang_feat_dim: int,       # 语言特征维度
        budget_dim: int = 32,     # 预算编码维度
        hidden_dim: int = 256,   # 隐藏层维度
        num_actions: int = None,  # 动作空间大小（可选）
    ):
        super().__init__()
        
        # 特征投影层（可选，用于维度对齐）
        self.image_proj = nn.Linear(image_feat_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_feat_dim, hidden_dim)
        self.budget_proj = nn.Linear(budget_dim, hidden_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        
        # 策略网络
        # 输出三个独立的分布：max_crops, top_k, num_active_blocks
        self.max_crops_head = nn.Linear(hidden_dim, num_max_crops)
        self.top_k_head = nn.Linear(hidden_dim, num_top_k)
        self.blocks_head = nn.Linear(hidden_dim, num_blocks)
        
    def forward(self, image_feat, lang_feat, budget):
        # 投影特征
        img_proj = self.image_proj(image_feat)
        lang_proj = self.lang_proj(lang_feat)
        budget_proj = self.budget_proj(budget)
        
        # 融合
        fused = torch.cat([img_proj, lang_proj, budget_proj], dim=-1)
        hidden = self.fusion(fused)
        
        # 输出动作分布
        max_crops_logits = self.max_crops_head(hidden)
        top_k_logits = self.top_k_head(hidden)
        blocks_logits = self.blocks_head(hidden)
        
        return {
            'max_crops': max_crops_logits,
            'top_k': top_k_logits,
            'num_active_blocks': blocks_logits,
        }
```

### 3.5 动作空间设计

**离散动作空间**（推荐）：
```python
# 动作空间定义
MAX_CROPS_OPTIONS = [2, 4, 6, 8, 10, 12]  # 6个选项
TOP_K_OPTIONS = [4, 8, 12, 16, 20, 24, 28, 32]  # 8个选项
NUM_BLOCKS_OPTIONS = [8, 9, 10, 11, 12, 13, 14, 15, 16]  # 9个选项

# 总动作空间大小：6 * 8 * 9 = 432种组合
# 但实际训练时，可以只考虑exp5/exp6中测试过的组合
```

**连续动作空间**（可选）：
```python
# 输出连续值，然后映射到离散选项
max_crops_value = torch.sigmoid(max_crops_logits) * (max_crops_max - max_crops_min) + max_crops_min
# 然后round到最近的选项
```

**推荐**：离散动作空间，更稳定且易于解释。

## 4. Reward Function设计

### 4.1 基础Reward设计

```python
def compute_reward(accuracy, latency, latency_budget, config):
    """
    计算reward
    
    Args:
        accuracy: 准确率 (0-1)
        latency: 实际延迟 (秒)
        latency_budget: 延迟预算 (秒)
        config: 选择的配置 (max_crops, top_k, num_active_blocks)
    
    Returns:
        reward: 标量奖励值
    """
    # 基础准确率奖励
    accuracy_reward = alpha * accuracy
    
    # 延迟惩罚（线性）
    latency_penalty = beta * max(0, latency - latency_budget)
    
    # 预算违反惩罚（硬约束）
    if latency > latency_budget:
        budget_violation_penalty = gamma * (latency - latency_budget) ** 2
    else:
        budget_violation_penalty = 0.0
    
    # 效率奖励（在预算内时，延迟越小越好）
    if latency <= latency_budget:
        efficiency_bonus = delta * (latency_budget - latency) / latency_budget
    else:
        efficiency_bonus = 0.0
    
    reward = accuracy_reward - latency_penalty - budget_violation_penalty + efficiency_bonus
    
    return reward
```

### 4.2 改进的Reward设计

考虑更多因素：

```python
def compute_reward_v2(accuracy, latency, latency_budget, config, baseline_accuracy=None):
    """
    改进的reward设计
    
    特点：
    1. 使用相对准确率（相对于baseline）
    2. 延迟惩罚使用sigmoid函数，更平滑
    3. 考虑配置的复杂度（鼓励选择简单配置）
    """
    # 相对准确率
    if baseline_accuracy is not None:
        relative_accuracy = (accuracy - baseline_accuracy) / (1.0 - baseline_accuracy + 1e-6)
    else:
        relative_accuracy = accuracy
    
    accuracy_reward = alpha * relative_accuracy
    
    # 平滑的延迟惩罚
    latency_ratio = latency / (latency_budget + 1e-6)
    latency_penalty = beta * torch.sigmoid(10 * (latency_ratio - 1.0))
    
    # 预算违反惩罚（硬约束）
    if latency > latency_budget:
        budget_violation_penalty = gamma * (latency / latency_budget - 1.0) ** 2
    else:
        budget_violation_penalty = 0.0
    
    # 配置复杂度惩罚（可选，鼓励选择简单配置）
    complexity = (config['max_crops'] / 12.0 + 
                  config['top_k'] / 32.0 + 
                  config['num_active_blocks'] / 16.0) / 3.0
    complexity_penalty = epsilon * complexity
    
    # 效率奖励
    if latency <= latency_budget:
        efficiency_bonus = delta * (1.0 - latency / latency_budget)
    else:
        efficiency_bonus = 0.0
    
    reward = (accuracy_reward - 
              latency_penalty - 
              budget_violation_penalty - 
              complexity_penalty + 
              efficiency_bonus)
    
    return reward
```

### 4.3 Reward参数设置

```python
# 默认参数（需要根据实际情况调整）
REWARD_PARAMS = {
    'alpha': 1.0,      # 准确率权重（主要目标）
    'beta': 0.5,       # 延迟惩罚权重
    'gamma': 10.0,     # 预算违反惩罚权重（硬约束，应该很大）
    'delta': 0.1,      # 效率奖励权重
    'epsilon': 0.05,   # 复杂度惩罚权重（可选）
}
```

### 4.4 Reward Model（可选）

如果需要更复杂的reward，可以训练一个reward model：

```python
class RewardModel(nn.Module):
    """
    学习reward function的模型
    输入：accuracy, latency, latency_budget, config
    输出：reward预测值
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),  # accuracy, latency, budget, max_crops, top_k, blocks
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, accuracy, latency, budget, max_crops, top_k, blocks):
        x = torch.stack([accuracy, latency, budget, 
                        max_crops/12.0, top_k/32.0, blocks/16.0], dim=-1)
        return self.net(x)
```

**使用场景**：
- 当reward function难以手工设计时
- 当需要从人类反馈中学习时（RLHF）
- 当reward需要适应不同数据集时

## 5. GRPO训练流程

### 5.1 数据准备

从exp5和exp6数据中提取训练数据：

```python
def prepare_training_data(exp5_results_dir, exp6_results_dir):
    """
    从exp5和exp6结果中提取训练数据
    
    返回格式：
    {
        'image_feat': ...,      # 需要从原始数据中提取
        'lang_feat': ...,       # 需要从原始数据中提取
        'latency_budget': ...,  # 从exp6中获取
        'config': {
            'max_crops': ...,
            'top_k': ...,
            'num_active_blocks': ...,
        },
        'accuracy': ...,
        'latency': ...,
        'reward': ...,
    }
    """
    # 1. 加载exp5和exp6的JSON结果
    # 2. 提取每个样本的配置、准确率、延迟
    # 3. 对于每个样本，需要重新提取特征（或保存特征）
    # 4. 计算reward
    # 5. 组织成训练数据格式
    pass
```

### 5.2 GRPO训练算法

```python
def grpo_train_step(policy, batch, group_size=8):
    """
    GRPO训练步骤
    
    Args:
        policy: 策略网络
        batch: 批次数据
        group_size: 组大小（用于组内比较）
    
    Returns:
        loss: 训练损失
    """
    # 1. 前向传播，获取动作分布
    action_logits = policy(batch['image_feat'], 
                          batch['lang_feat'], 
                          batch['budget'])
    
    # 2. 采样动作（或使用历史动作）
    actions = sample_actions(action_logits)
    
    # 3. 计算reward（使用实际执行结果）
    rewards = compute_reward(
        accuracy=batch['accuracy'],
        latency=batch['latency'],
        latency_budget=batch['budget'],
        config=actions,
    )
    
    # 4. 将数据分组
    groups = group_trajectories(batch, group_size)
    
    # 5. 计算组内相对优势
    advantages = []
    for group in groups:
        group_rewards = rewards[group]
        group_mean = group_rewards.mean()
        group_advantages = group_rewards - group_mean
        advantages.append(group_advantages)
    advantages = torch.cat(advantages)
    
    # 6. 计算策略损失
    log_probs = compute_log_probs(action_logits, actions)
    loss = -(log_probs * advantages).mean()
    
    return loss
```

### 5.3 完整训练流程

```python
def train_grpo_controller(
    policy,
    train_loader,
    val_loader,
    num_epochs=100,
    lr=1e-4,
    group_size=8,
):
    """
    完整的GRPO训练流程
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        # 训练阶段
        policy.train()
        for batch in train_loader:
            loss = grpo_train_step(policy, batch, group_size)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        policy.eval()
        val_metrics = evaluate_controller(policy, val_loader)
        
        # 保存checkpoint
        if epoch % 10 == 0:
            save_checkpoint(policy, epoch, val_metrics)
```

## 6. Overhead控制

### 6.1 Controller计算开销

**目标**：Controller的计算开销应该 < 5% 的总推理时间

**优化策略**：

1. **轻量级网络**：
   - 使用小的hidden dimension（128-256）
   - 使用1-2层MLP
   - 使用深度可分离卷积（如果处理图像）

2. **特征缓存**：
   - 图像特征和语言特征可以在主模型forward时提取
   - Controller只需要做轻量级的投影和融合

3. **量化**：
   - 使用INT8量化
   - 使用TensorRT或ONNX Runtime优化

4. **批处理**：
   - Controller支持batch inference
   - 一次处理多个样本

### 6.2 特征提取开销

**优化方案**：

1. **复用主模型特征**：
   ```python
   # 在主模型forward时，同时提取controller需要的特征
   class ModelWithController:
       def forward(self, ...):
           # 主模型forward
           outputs = self.model(...)
           
           # 同时提取controller特征（几乎无额外开销）
           image_feat = self._extract_image_feat(...)
           lang_feat = self._extract_lang_feat(...)
           
           return outputs, image_feat, lang_feat
   ```

2. **特征降维**：
   - 使用PCA或线性投影降维
   - 从2048维降到256维

3. **异步提取**：
   - Controller特征提取可以异步进行
   - 使用CUDA streams并行

### 6.3 内存开销

**优化方案**：

1. **共享权重**：
   - Controller的特征投影层可以与主模型共享部分权重

2. **梯度检查点**：
   - 训练时使用gradient checkpointing

3. **混合精度**：
   - 使用FP16或BF16

## 7. 与其他方法的比较

### 7.1 vs 监督学习（Supervised Learning）

**监督学习**：
- 需要大量标注数据（每个样本的最佳配置）
- 难以处理延迟预算的动态变化
- 无法探索新的配置组合

**GRPO**：
- 可以从exp5/exp6的离线数据中学习
- 可以适应不同的延迟预算
- 可以通过探索发现更好的配置

### 7.2 vs PPO（Proximal Policy Optimization）

**PPO**：
- 需要价值网络（V-function），参数多
- 训练不稳定，需要clip
- 需要在线交互或大量模拟

**GRPO**：
- 不需要价值网络，参数少
- 通过组内比较更稳定
- 适合离线学习

### 7.3 vs DPO（Direct Policy Optimization）

**DPO**：
- 直接优化策略，简单
- 但可能更新幅度过大
- 需要成对的偏好数据

**GRPO**：
- 通过相对比较控制更新幅度
- 更稳定
- 可以从单样本数据中学习

## 8. 实施计划

### 8.1 阶段1：数据准备（1-2周）

1. 解析exp5和exp6的JSON结果
2. 提取每个样本的配置、准确率、延迟
3. 重新运行模型，提取图像和语言特征（或从缓存中加载）
4. 组织成训练数据格式
5. 数据增强和平衡

### 8.2 阶段2：模型实现（1周）

1. 实现Controller网络架构
2. 实现特征提取模块
3. 实现reward function
4. 实现GRPO训练算法

### 8.3 阶段3：训练（2-3周）

1. 超参数调优
2. 训练Controller
3. 验证和评估

### 8.4 阶段4：集成和优化（1-2周）

1. 集成到主模型
2. 优化overhead
3. 端到端测试

### 8.5 阶段5：评估和部署（1周）

1. 在多个数据集上评估
2. 性能分析
3. 部署准备

## 9. 评估指标

### 9.1 主要指标

1. **准确率提升**：相比固定配置或baseline的准确率提升
2. **延迟满足率**：满足延迟预算的样本比例
3. **平均延迟**：所有样本的平均延迟
4. **配置选择分布**：不同配置被选择的频率

### 9.2 辅助指标

1. **Controller开销**：Controller本身的计算时间
2. **训练稳定性**：训练过程中的loss和reward变化
3. **泛化能力**：在不同数据集上的表现

## 10. 风险和缓解措施

### 10.1 风险

1. **数据不足**：exp5/exp6的数据可能不足以覆盖所有情况
2. **特征提取开销**：可能引入过多计算开销
3. **训练不稳定**：GRPO可能在某些情况下不稳定
4. **过拟合**：可能过拟合到训练数据

### 10.2 缓解措施

1. **数据增强**：使用数据增强技术
2. **特征缓存**：缓存特征，减少重复计算
3. **正则化**：使用dropout、weight decay
4. **交叉验证**：使用交叉验证评估泛化能力

## 11. 未来改进方向

1. **在线学习**：支持在线更新Controller
2. **多任务学习**：同时优化多个数据集
3. **元学习**：快速适应新的延迟预算
4. **可解释性**：分析Controller的决策过程

