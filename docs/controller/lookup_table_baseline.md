# Lookup Table Baseline Controller

## 概述

Lookup Table Baseline Controller 是一个基于 offline profiling 结果的简单 baseline controller。它不需要训练，直接从 core_exp 的 profiling 结果构建一个 lookup table，根据给定的 latency budget 查找最优的 (tier, top_k, num_active_blocks) 配置。

## 设计理念

与基于 GRPO 训练的 controller 不同，lookup table baseline：

1. **不需要训练**：直接使用 profiling 数据
2. **简单高效**：O(1) 查找时间，无模型推理开销
3. **基于统计**：对每个配置聚合多个样本的 accuracy 和 latency
4. **预算约束**：给定 latency budget，找到满足预算且 accuracy 最高的配置

## 使用方法

### 1. 从 Profiling 结果构建 Lookup Table

```bash
# 从 core_exp_h100 结果构建 lookup table
python experiments/controller/lookup_table_baseline.py \
    --results_dir ./results/core_exp_h100 \
    --output_file ./checkpoints/controller/lookup_table_baseline.json \
    --aggregation_method mean \
    --tolerance 0.05
```

参数说明：
- `--results_dir`: core_exp profiling 结果目录
- `--output_file`: 保存 lookup table 的路径
- `--aggregation_method`: 聚合方法 (`mean`, `median`, `max_accuracy`)
- `--tolerance`: Latency budget 匹配的容差 (0.05 = 5%)
- `--datasets`: 可选，指定要使用的数据集（默认使用所有可用数据集）

### 2. 使用 Lookup Table Controller

#### 方法 1: 直接使用

```python
from experiments.controller.lookup_table_baseline import LookupTableBaselineController

# 从文件加载
controller = LookupTableBaselineController.load("./checkpoints/controller/lookup_table_baseline.json")

# 预测配置
config = controller.predict(latency_budget=200.0)
print(f"Tier: {config['tier']}")
print(f"Top-K: {config['top_k']}")
print(f"Num Active Blocks: {config['num_active_blocks']}")
print(f"Expected Accuracy: {config['accuracy']:.4f}")
print(f"Expected Latency: {config['latency']:.2f}ms")
```

#### 方法 2: 使用 Wrapper（兼容现有接口）

```python
from experiments.controller.lookup_table_wrapper import create_lookup_table_controller

# 创建 controller
controller = create_lookup_table_controller(
    lookup_table_path="./checkpoints/controller/lookup_table_baseline.json"
)

# 预测所有 knobs
config = controller.predict_all(latency_budget=200.0)

# 或分别预测 Stage1 和 Stage2
stage1_config = controller.predict_stage1(latency_budget=200.0)  # {'tier': 'medium'}
stage2_config = controller.predict_stage2(latency_budget=200.0)  # {'top_k': 8, 'num_active_blocks': 16}
```

### 3. 测试 Controller

```bash
# 测试 lookup table controller
python experiments/controller/lookup_table_wrapper.py \
    --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
    --test_budgets 150.0 200.0 250.0 300.0 350.0
```

## 工作原理

### 1. 构建配置表

对于每个 (tier, top_k, num_active_blocks) 组合：
- 从 profiling 结果中收集所有匹配的样本
- 使用聚合方法（mean/median/max_accuracy）计算平均 accuracy 和 latency
- 存储配置信息：`(tier, top_k, num_active_blocks) -> (accuracy, latency, num_samples)`

### 2. 查找最优配置

给定 latency budget：
1. 找到所有满足 `latency <= budget * (1 + tolerance)` 的配置
2. 在这些配置中选择 accuracy 最高的
3. 如果没有满足预算的配置，返回 latency 最接近的配置

### 3. 聚合方法

- **mean**: 使用所有样本的平均值（推荐，最稳定）
- **median**: 使用中位数（对异常值更鲁棒）
- **max_accuracy**: 使用 accuracy 最高的样本的 latency（可能过于乐观）

## 输出格式

`predict()` 方法返回的配置字典：

```python
{
    'tier': 'medium',              # str: "low", "medium", "high"
    'top_k': 8,                     # int: MoE top-K value
    'num_active_blocks': 16,        # int: Number of active transformer blocks
    'accuracy': 0.8234,             # float: Expected accuracy
    'latency': 195.23,              # float: Expected latency (ms)
    'num_samples': 150,            # int: Number of samples used for aggregation
    'accuracy_std': 0.0123,         # float: Standard deviation of accuracy
    'latency_std': 5.67,            # float: Standard deviation of latency
}
```

## 与 GRPO Controller 的对比

| 特性 | Lookup Table Baseline | GRPO Controller |
|------|----------------------|-----------------|
| 训练需求 | ❌ 不需要 | ✅ 需要 |
| 推理开销 | O(1) 查找 | 需要运行神经网络 |
| 输入特征 | 仅需 latency budget | Vision + Language + Budget |
| 个性化 | ❌ 全局最优 | ✅ 可针对样本个性化 |
| 准确性 | 基于统计平均 | 可学习复杂模式 |

## 适用场景

Lookup Table Baseline 适合：
- **快速 baseline**：不需要训练即可获得结果
- **简单场景**：latency budget 是唯一约束
- **资源受限**：无法运行神经网络 controller
- **对比实验**：作为 GRPO controller 的 baseline

GRPO Controller 适合：
- **个性化决策**：需要根据图像和文本内容调整
- **复杂模式**：需要学习 accuracy-latency 的复杂关系
- **在线学习**：可以通过强化学习持续优化

## 文件结构

```
experiments/controller/
├── lookup_table_baseline.py      # 核心实现
├── lookup_table_wrapper.py      # 兼容性 wrapper
└── core_exp_data_loader.py      # 数据加载器

checkpoints/controller/
└── lookup_table_baseline.json    # 保存的 lookup table
```

## 示例：完整工作流

```python
# 1. 构建 lookup table
from experiments.controller.lookup_table_baseline import build_lookup_table_from_core_exp

controller = build_lookup_table_from_core_exp(
    results_dir="./results/core_exp_h100",
    dataset_names=["coco_2014_vqa", "text_vqa"],
    output_file="./checkpoints/controller/lookup_table_baseline.json",
    aggregation_method="mean",
    tolerance=0.05,
)

# 2. 使用 controller
config = controller.predict(latency_budget=200.0)
print(f"Best config for 200ms budget: {config}")

# 3. 查看所有候选配置
all_configs = controller.predict(latency_budget=200.0, return_all_candidates=True)
print(f"Found {len(all_configs)} valid configurations")

# 4. 获取统计信息
stats = controller.get_statistics()
print(f"Latency range: {stats['latency_range']}")
print(f"Accuracy range: {stats['accuracy_range']}")
```

## 注意事项

1. **数据质量**：Lookup table 的质量取决于 profiling 结果的覆盖度和准确性
2. **配置覆盖**：确保 profiling 结果包含足够的 (tier, top_k, num_active_blocks) 组合
3. **容差设置**：`tolerance` 参数影响配置选择的灵活性，建议 0.05-0.1
4. **聚合方法**：`mean` 方法最稳定，适合大多数场景
5. **跨数据集**：可以合并多个数据集的 profiling 结果，提高配置覆盖度

## 未来改进

- [ ] 支持按数据集选择最优配置
- [ ] 支持动态调整 tolerance
- [ ] 添加配置缓存机制
- [ ] 支持增量更新 lookup table
- [ ] 添加配置验证和错误处理

