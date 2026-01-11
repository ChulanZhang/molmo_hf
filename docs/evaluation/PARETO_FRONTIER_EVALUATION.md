# Pareto Frontier 评估指南

## 概述

本指南介绍如何使用 lookup table baseline controller 进行 Pareto frontier 评估，并生成可视化图表。

## 快速开始

### 步骤 1: 运行评估

运行批量评估，测试多个数据集和 latency budget：

```bash
python experiments/controller/evaluate_pareto_frontier.py \
    --model_path checkpoints/molmo \
    --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
    --datasets text_vqa okvqa coco_2014_vqa \
    --latency_budgets 170 200 230 260 290 320 350 380 \
    --num_samples 1000 \
    --output_path ./logs_eval/pareto_frontier/
```

### 步骤 2: 绘制 Pareto Frontier 图

评估完成后，生成可视化图表：

```bash
python experiments/controller/plot_pareto_frontier.py \
    --pareto_data ./logs_eval/pareto_frontier/pareto_data.json \
    --output_dir ./plots/pareto_frontier/
```

## 详细说明

### 评估脚本 (`evaluate_pareto_frontier.py`)

#### 功能

1. **批量评估**: 在多个数据集和多个 latency budget 上运行评估
2. **结果收集**: 自动收集所有评估结果
3. **Pareto Frontier 计算**: 计算每个数据集的 Pareto frontier
4. **数据保存**: 保存为 JSON 格式，便于后续分析和可视化

#### 参数说明

- `--model_path`: 模型 checkpoint 路径（必需）
- `--lookup_table_path`: Lookup table JSON 文件路径（必需）
- `--datasets`: 要评估的数据集列表（默认: text_vqa, okvqa, coco_2014_vqa）
- `--latency_budgets`: Latency budget 列表，单位 ms（默认: 170, 200, 230, 260, 290, 320, 350, 380）
- `--num_samples`: 每个数据集评估的样本数（默认: 1000）
- `--max_new_tokens`: 最大生成 token 数（默认: 128）
- `--device`: 使用的设备（默认: cuda）
- `--output_path`: 输出目录（默认: ./logs_eval/pareto_frontier/）
- `--save_predictions`: 是否保存每个样本的预测结果
- `--skip_evaluation`: 跳过评估，只收集已有结果并计算 Pareto frontier

#### 输出文件

评估完成后，会生成以下文件：

```
logs_eval/pareto_frontier/
├── results/
│   ├── text_vqa/
│   │   ├── budget_170/
│   │   │   ├── text_vqa_validation_budget_170_results.json
│   │   │   └── text_vqa_validation_budget_170_predictions.jsonl
│   │   ├── budget_200/
│   │   └── ...
│   ├── okvqa/
│   └── coco_2014_vqa/
└── pareto_data.json  # 汇总的 Pareto 数据
```

`pareto_data.json` 包含：
- `all_points`: 所有评估点（每个数据集的所有 budget 点）
- `pareto_frontiers`: Pareto frontier 点（每个数据集）
- `summary`: 统计摘要

### 可视化脚本 (`plot_pareto_frontier.py`)

#### 功能

1. **单数据集图**: 为每个数据集生成单独的 Pareto frontier 图
2. **对比图**: 生成所有数据集的对比图
3. **标注**: 在 Pareto 点上标注配置信息（tier, top_k, num_active_blocks）

#### 参数说明

- `--pareto_data`: Pareto 数据 JSON 文件路径（必需）
- `--output_dir`: 输出目录（默认: ./plots/pareto_frontier/）
- `--figsize`: 图片尺寸，格式: width height（默认: 10 7）
- `--dpi`: 图片分辨率（默认: 300）

#### 输出文件

```
plots/pareto_frontier/
├── text_vqa_pareto_frontier.png
├── okvqa_pareto_frontier.png
├── coco_2014_vqa_pareto_frontier.png
└── all_datasets_comparison.png  # 所有数据集对比图
```

## 使用示例

### 示例 1: 完整评估流程

```bash
# 1. 运行评估（可能需要几小时）
python experiments/controller/evaluate_pareto_frontier.py \
    --model_path checkpoints/molmo \
    --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
    --datasets text_vqa okvqa coco_2014_vqa \
    --latency_budgets 170 200 230 260 290 320 350 380 \
    --num_samples 1000 \
    --output_path ./logs_eval/pareto_frontier/

# 2. 生成图表
python experiments/controller/plot_pareto_frontier.py \
    --pareto_data ./logs_eval/pareto_frontier/pareto_data.json \
    --output_dir ./plots/pareto_frontier/
```

### 示例 2: 只评估特定数据集

```bash
python experiments/controller/evaluate_pareto_frontier.py \
    --model_path checkpoints/molmo \
    --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
    --datasets text_vqa \
    --latency_budgets 170 200 230 260 290 320 350 380 \
    --num_samples 500 \
    --output_path ./logs_eval/pareto_frontier_text_vqa/
```

### 示例 3: 使用已有结果重新计算 Pareto Frontier

如果评估已经完成，只想重新计算 Pareto frontier 或生成图表：

```bash
# 只收集结果并计算 Pareto frontier（不重新评估）
python experiments/controller/evaluate_pareto_frontier.py \
    --model_path checkpoints/molmo \
    --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
    --datasets text_vqa okvqa coco_2014_vqa \
    --latency_budgets 170 200 230 260 290 320 350 380 \
    --skip_evaluation \
    --output_path ./logs_eval/pareto_frontier/
```

### 示例 4: 自定义 Latency Budget 范围

```bash
python experiments/controller/evaluate_pareto_frontier.py \
    --model_path checkpoints/molmo \
    --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
    --datasets text_vqa \
    --latency_budgets 150 180 210 240 270 300 330 360 \
    --num_samples 1000 \
    --output_path ./logs_eval/pareto_frontier_custom/
```

## Pareto Frontier 说明

### 什么是 Pareto Frontier？

在 accuracy-latency trade-off 中，Pareto frontier 是指：
- **没有其他点同时具有更高 accuracy 和更低 latency** 的点集合
- 或者说：**没有其他点支配它**的点集合

一个点支配另一个点，当且仅当：
- 前者的 accuracy ≥ 后者的 accuracy **且**
- 前者的 latency ≤ 后者的 latency
- 至少有一个严格不等式

### Pareto Frontier 的意义

- **最优配置**: Pareto frontier 上的点代表在给定 latency 下能达到的最高 accuracy
- **Trade-off 曲线**: 展示了 accuracy 和 latency 之间的权衡关系
- **配置选择**: 可以根据 latency budget 选择 Pareto frontier 上对应的配置

## 结果分析

### 查看 Pareto 数据

```python
import json

with open('./logs_eval/pareto_frontier/pareto_data.json', 'r') as f:
    data = json.load(f)

# 查看某个数据集的 Pareto frontier
dataset = 'text_vqa'
pareto_points = data['pareto_frontiers'][dataset]

for point in pareto_points:
    print(f"Budget: {point['budget']:.0f}ms, "
          f"Latency: {point['latency']:.1f}ms, "
          f"Accuracy: {point['accuracy']:.4f}, "
          f"Config: {point['config']}")
```

### 分析 Pareto Frontier 特征

1. **斜率**: 陡峭的斜率表示小的 latency 增加能带来大的 accuracy 提升
2. **范围**: Latency 和 accuracy 的范围显示了系统的灵活性
3. **点数**: Pareto frontier 上的点数表示有多少个不同的最优配置

## 故障排除

### 问题 1: 评估失败

**症状**: 某些数据集或 budget 的评估失败

**解决方案**:
- 检查模型和 lookup table 路径是否正确
- 检查 GPU 内存是否足够
- 减少 `--num_samples` 进行测试
- 查看错误日志

### 问题 2: 没有 Pareto 点

**症状**: Pareto frontier 为空或点数很少

**可能原因**:
- 评估点太少
- Latency budget 范围不合适
- 所有点都在 Pareto frontier 上（正常情况）

**解决方案**:
- 增加评估的 budget 点数量
- 检查数据是否正确加载

### 问题 3: 图表无法生成

**症状**: `plot_pareto_frontier.py` 报错

**解决方案**:
- 确保安装了 matplotlib: `pip install matplotlib`
- 检查 Pareto 数据文件是否存在
- 检查数据文件格式是否正确

## 性能优化

### 加速评估

1. **减少样本数**: 使用较小的 `--num_samples` 进行快速测试
2. **并行评估**: 可以手动并行运行多个评估任务（不同数据集或 budget）
3. **跳过已有结果**: 使用 `--skip_evaluation` 只处理新数据

### 内存优化

- 使用 `batch_size=1`（默认）
- 减少 `--num_samples`
- 不使用 `--save_predictions`（如果不需要）

## 扩展

### 对比多个 Controller

可以运行多个评估（不同的 controller），然后对比它们的 Pareto frontiers：

```bash
# 评估 lookup table baseline
python experiments/controller/evaluate_pareto_frontier.py \
    --model_path checkpoints/molmo \
    --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
    --output_path ./logs_eval/pareto_frontier_lookup_table/

# 评估 GRPO controller（需要相应的评估脚本）
# python experiments/controller/evaluate_pareto_frontier_grpo.py \
#     --output_path ./logs_eval/pareto_frontier_grpo/

# 然后可以修改 plot_pareto_frontier.py 来对比多个结果
```

### 自定义可视化

可以修改 `plot_pareto_frontier.py` 来：
- 添加更多数据集
- 改变颜色和样式
- 添加额外的统计信息
- 生成不同格式的图表

## 参考

- [Lookup Table Baseline Evaluation](./LOOKUP_TABLE_BASELINE_EVALUATION.md)
- [Evaluation Guide](./EVALUATION_GUIDE.md)
- [AdaLLaVA Datasets](./ADALLAVA_DATASETS.md)

