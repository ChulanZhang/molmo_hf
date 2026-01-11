# Latency Estimator 设计与结构

> **文档目的**: 详细说明Latency Estimator的架构、设计原理和使用方法  
> **最后更新**: 2026-01-01  
> **版本**: 2.0 (重构版)

## 📋 目录

1. [设计目标与用途](#设计目标与用途)
2. [模型架构](#模型架构)
3. [输入特征](#输入特征)
4. [输出预测](#输出预测)
5. [训练方法](#训练方法)
6. [使用场景](#使用场景)
7. [代码实现](#代码实现)

---

## 🎯 设计目标与用途

### 为什么需要Latency Estimator？

在Controller的训练和推理中，我们需要：

1. **预测每个configuration的latency**: 对于给定的configuration（tier, top_k, num_active_blocks），预测其latency特征
2. **反推满足budget的configuration set**: 给定latency budget，找出所有满足budget的configurations
3. **选择最优configuration**: 在满足budget的configurations中，选择accuracy最高的

### 核心设计原则

1. **不依赖未知信息**: 推理时不知道会生成多少个token，所以**不使用output_tokens作为输入**
2. **预测可组合的latency**: 分别预测prefill latency和decode per-token latency，可以根据实际output_tokens计算总latency
3. **轻量级**: 参数量小（~100K），推理快（<0.1ms）
4. **准确性**: 预测误差 <5ms（prefill），<1ms（decode per-token）

### 使用逻辑

```
给定latency budget → 对于每个configuration:
  1. 预测 T_prefill_total 和 T_decode_per_token
  2. 计算满足budget的最大output_tokens:
     max_output_tokens = (budget - T_prefill_total) / T_decode_per_token
  3. 如果 max_output_tokens > 0，这个configuration是可行的
  4. 在可行configurations中选择accuracy最高的
```

---

## 🏗️ 模型架构

### 整体结构

```
Input Features (5维) → Feature Encoder (MLP) → 2个预测头 → 输出
```

### 详细架构

```python
LatencyEstimator(
    hidden_dim=256,        # 隐藏层维度
    num_layers=2,          # MLP层数（不包括输出层）
)
```

**参数量计算**:
- Input → Hidden: `5 × 256 = 1,280`
- Hidden → Hidden: `256 × 256 = 65,536` (每层)
- Hidden → Output: `256 × 1 = 256` (每个头)
- LayerNorm: `256 × 2 = 512` (每层)
- **总参数量**: ~100K-200K（取决于num_layers）

### 网络结构

```
Input (B, 5)
    ↓
Linear(5 → 256) + LayerNorm + ReLU
    ↓
[可选] Linear(256 → 256) + LayerNorm + ReLU  (如果num_layers > 1)
    ↓
Shared Encoder Output (B, 256)
    ↓
    ├─→ Linear(256 → 1) + ReLU → T_prefill_total
    └─→ Linear(256 → 1) + ReLU → T_decode_per_token
```

**关键设计**:
- **共享编码器**: 两个预测头共享同一个特征编码器，减少参数量
- **ReLU激活**: 确保输出非负（latency不能为负）
- **LayerNorm**: 稳定训练，加速收敛

---

## 📥 输入特征

### 特征列表（5维）

| 特征 | 类型 | 范围/取值 | 说明 |
|------|------|----------|------|
| `vision_tokens` | int | 100-1000 | Vision token数量（取决于tier和图像大小） |
| `text_tokens` | int | 20-200 | 文本token数量（prompt长度） |
| `tier_idx` | int | 0, 1, 2 | Tier索引（0=low, 1=medium, 2=high） |
| `top_k` | int | 4, 6, 8, 10, 12 | MoE top-K值 |
| `num_active_blocks` | int | 8, 10, 12, 14, 16 | 激活的transformer block数量 |

**重要**: **不使用output_tokens作为输入**，因为推理时不知道会生成多少个token。

### 特征编码

```python
# 构建特征向量
features = torch.stack([
    vision_tokens.float(),      # (B,)
    text_tokens.float(),        # (B,)
    tier_idx.float(),           # (B,)
    top_k.float(),              # (B,)
    num_active_blocks.float(),  # (B,)
], dim=-1)  # (B, 5)
```

---

## 📤 输出预测

### 预测目标（2个）

| 输出 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `T_prefill_total` | float | ms | 总prefill latency = Vision encoder + Projector + LLM prefill |
| `T_decode_per_token` | float | ms/token | 每个输出token的decode latency |

### 输出计算流程

```python
# 1. 预测latency
T_prefill_total = prefill_head(encoded)      # (B,)
T_decode_per_token = decode_head(encoded)    # (B,)

# 2. 使用时的总latency计算（在外部）
# T_total = T_prefill_total + T_decode_per_token * output_tokens
```

**设计考虑**:
- **阶段分解**: 分别预测prefill和decode per-token，可以根据实际output_tokens计算总latency
- **不预测T_total**: 因为output_tokens未知，无法在estimator内部计算T_total
- **可组合性**: 预测的latency可以灵活组合，适应不同的output_tokens

### 使用示例

```python
# 预测latency
latencies = estimator(
    vision_tokens=vision_tokens,
    text_tokens=text_tokens,
    tier_idx=tier_idx,
    top_k=top_k,
    num_active_blocks=num_active_blocks,
)

# 检查是否满足budget（假设expected_output_tokens）
T_prefill = latencies['T_prefill_total']
T_decode_per_token = latencies['T_decode_per_token']
T_total = T_prefill + T_decode_per_token * expected_output_tokens

if T_total <= latency_budget:
    # Configuration可行
    pass
```

---

## 🎓 训练方法

### 训练数据

**数据来源**: Core experiment结果（JSON文件）

**数据格式**:
```json
{
  "per_sample_results": [
    {
      "actual_vision_tokens": 384,
      "actual_text_tokens": 53,
      "output_tokens": 11,
      "tier": "low",
      "top_k": 4,
      "num_active_blocks": 12,
      "T_vision_total": 11.93,
      "T_LLM_prefill": 99.25,
      "T_LLM_decode": 38.85,
      "T_decode_per_token": 19.43
    }
  ]
}
```

**数据预处理**:
- `T_prefill_total = T_vision_total + T_LLM_prefill`
- `T_decode_per_token = T_LLM_decode / output_tokens`（如果JSON中没有，则计算）

### 损失函数

**多任务损失**:
```python
loss = loss_prefill + loss_decode
```

其中每个loss都是MSE loss:
- `loss_prefill = MSE(pred_T_prefill_total, target_T_prefill_total)`
- `loss_decode = MSE(pred_T_decode_per_token, target_T_decode_per_token)`

**设计考虑**:
- **只训练两个目标**: 不训练T_total，因为output_tokens在训练时已知，但在推理时未知
- **等权重**: 两个loss等权重，因为都是重要的latency组件

### 训练指标

**主要指标**:
- **MAE (Mean Absolute Error)**: 平均绝对误差
  - `MAE_prefill < 5ms`
  - `MAE_decode_per_token < 1ms`
- **Relative Error**: 相对误差
  - `rel_error_prefill < 5%`
  - `rel_error_decode < 10%`

### 训练配置

```python
# 默认配置
batch_size = 64
num_epochs = 50
lr = 1e-3
weight_decay = 1e-5
optimizer = Adam
train_split = 0.8  # 80%训练，20%验证
```

---

## 🚀 使用场景

### 1. 反推满足budget的configuration set

```python
# 给定latency budget和expected_output_tokens
latency_budget = 200.0  # ms
expected_output_tokens = 10  # 假设值

# 枚举所有可能的configurations
configs = [
    {'tier': 'low', 'top_k': 4, 'num_active_blocks': 8},
    {'tier': 'low', 'top_k': 6, 'num_active_blocks': 10},
    # ...
]

feasible_configs = []
for config in configs:
    # 预测latency
    latencies = estimator.predict_from_config({
        'vision_tokens': vision_tokens,
        'text_tokens': text_tokens,
        'tier': config['tier'],
        'top_k': config['top_k'],
        'num_active_blocks': config['num_active_blocks'],
    })
    
    # 计算总latency
    T_total = latencies['T_prefill_total'] + latencies['T_decode_per_token'] * expected_output_tokens
    
    # 检查是否满足budget
    if T_total <= latency_budget:
        feasible_configs.append(config)
```

### 2. Controller训练加速（GRPO）

在GRPO训练中，使用estimator预测latency，避免实际运行模型：

```python
# 传统方法（慢）
latency = run_model(config)  # batch_size=1, 很慢

# 使用estimator（快）
latencies = estimator.predict(config)
T_total = latencies['T_prefill_total'] + latencies['T_decode_per_token'] * expected_output_tokens
```

**优势**:
- 支持batch预测（batch_size > 1）
- 推理速度快（<0.1ms vs 100ms+）
- 训练速度提升10-100倍

### 3. 配置搜索

在controller训练前，可以使用estimator快速评估不同配置：

```python
configs = [
    {'tier': 'low', 'top_k': 4, 'num_active_blocks': 8},
    {'tier': 'medium', 'top_k': 6, 'num_active_blocks': 10},
    # ...
]

for config in configs:
    latencies = estimator.predict_from_config(config)
    print(f"Config {config}: prefill={latencies['T_prefill_total']:.2f}ms, decode_per_token={latencies['T_decode_per_token']:.3f}ms/token")
```

---

## 💻 代码实现

### 模型定义

**文件**: `experiments/controller/latency_estimator.py`

**核心类**:
- `LatencyEstimator`: 模型定义
- `LatencyEstimatorTrainer`: 训练器

### 使用示例

**训练**:
```python
from experiments.controller.latency_estimator import LatencyEstimator, LatencyEstimatorTrainer

# 创建模型
model = LatencyEstimator(hidden_dim=256, num_layers=2)

# 创建训练器
trainer = LatencyEstimatorTrainer(model, device='cuda', lr=1e-3)

# 训练
for epoch in range(num_epochs):
    for batch in train_loader:
        metrics = trainer.train_step(batch)
```

**推理**:
```python
# 方法1: 直接forward
latencies = model(
    vision_tokens=torch.tensor([384]),
    text_tokens=torch.tensor([53]),
    tier_idx=torch.tensor([0]),  # low
    top_k=torch.tensor([4]),
    num_active_blocks=torch.tensor([12]),
)

# 方法2: 使用predict_from_config
config = {
    'vision_tokens': torch.tensor([384]),
    'text_tokens': torch.tensor([53]),
    'tier': ['low'],  # 或 torch.tensor([0])
    'top_k': torch.tensor([4]),
    'num_active_blocks': torch.tensor([12]),
}
latencies = model.predict_from_config(config)

# 计算总latency（需要output_tokens）
output_tokens = 10
T_total = latencies['T_prefill_total'] + latencies['T_decode_per_token'] * output_tokens
```

**检查budget可行性**:
```python
# 使用check_budget_feasibility方法
feasible = model.check_budget_feasibility(
    vision_tokens=vision_tokens,
    text_tokens=text_tokens,
    tier_idx=tier_idx,
    top_k=top_k,
    num_active_blocks=num_active_blocks,
    latency_budget=latency_budget,
    expected_output_tokens=expected_output_tokens,
)
```

### 训练脚本

**文件**: `experiments/controller/train_latency_estimator.py`

**使用**:
```bash
# 使用所有可用数据集（推荐）
python experiments/controller/train_latency_estimator.py \
    --results_dir results/core_exp_h100/4run_2000samples \
    --use_all_datasets \
    --output_dir checkpoints/latency_estimator \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 1e-3 \
    --device cuda \
    --seed 3407

# 或者不指定dataset_names，会自动检测所有数据集
python experiments/controller/train_latency_estimator.py \
    --results_dir results/core_exp_h100/4run_2000samples \
    --output_dir checkpoints/latency_estimator \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 1e-3 \
    --device cuda \
    --seed 3407
```

**可用数据集**（在`4run_2000samples`目录下）:
- `coco_2014_vqa`
- `coco_caption`
- `doc_qa`
- `mmmu`
- `okvqa`
- `science_qa_img`
- `st_qa`
- `tally_qa`
- `text_vqa`

总共9个数据集，每个数据集有27个JSON文件（不同配置组合）。

---

## 📊 性能指标

### 预期性能

**参数量**: ~100K-200K parameters

**推理速度**: <0.1ms per sample (GPU)

**预测准确度**:
- **MAE_prefill**: <5ms
- **MAE_decode_per_token**: <1ms
- **Relative Error**: <5% (prefill), <10% (decode)

### 实际性能（待训练后验证）

训练完成后，会在验证集上评估：
- 各阶段的MAE和RMSE
- 不同配置下的误差分布
- 相对误差分析

---

## 🔧 设计细节

### 1. 为什么只预测prefill和decode per-token？

**原因**:
- 推理时不知道output_tokens，无法预测T_total
- 分别预测prefill和decode per-token，可以根据实际output_tokens灵活计算T_total
- 更符合实际使用场景

### 2. 为什么不用output_tokens作为输入？

**原因**:
- 推理时不知道会生成多少个token
- 如果使用output_tokens作为输入，训练和推理的数据分布不一致
- 会导致模型在推理时无法使用

### 3. 为什么使用共享编码器？

**原因**:
- 减少参数量（2个独立编码器 vs 1个共享编码器）
- 多任务学习，共享特征表示
- 训练更稳定

### 4. 如何计算T_prefill_total？

**计算方式**:
- `T_prefill_total = T_vision_total + T_LLM_prefill`
- 在训练数据预处理时计算
- 这样预测一个值即可，不需要分别预测vision和prefill

---

## 📚 相关文档

- **[DESIGN.md](DESIGN.md)**: Controller整体设计
- **[EXPERIMENTS.md](EXPERIMENTS.md)**: 实验说明（Exp 1）
- **[ANALYSIS.md](ANALYSIS.md)**: 技术分析

---

**最后更新**: 2026-01-01  
**维护者**: Controller Team
