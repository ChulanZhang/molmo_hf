# 合并多数据集 Importance Scores

## 概述

这个脚本用于合并多个数据集的 importance scores，生成一个跨数据集的统一 ranking。

## 使用方法

### 1. 生成多数据集合并的 Importance Scores

```bash
python3 experiments/profiling/knob3_layers/merge_multi_dataset_importance_scores.py \
    --comparison_dir results/profiling/exp3_importance_comparison \
    --output_file results/layer_importance_scores_multi_dataset.json \
    --method average
```

### 2. 合并方法选项

- `average`: 简单平均（推荐，默认）
- `weighted_avg`: 加权平均（可以指定每个数据集的权重）
- `median`: 中位数（对异常值更鲁棒）

### 3. 示例

#### 使用简单平均（推荐）

```bash
python3 experiments/profiling/knob3_layers/merge_multi_dataset_importance_scores.py \
    --comparison_dir results/profiling/exp3_importance_comparison \
    --output_file results/layer_importance_scores_multi_dataset.json \
    --method average
```

#### 使用加权平均

```bash
python3 experiments/profiling/knob3_layers/merge_multi_dataset_importance_scores.py \
    --comparison_dir results/profiling/exp3_importance_comparison \
    --output_file results/layer_importance_scores_multi_dataset.json \
    --method weighted_avg \
    --dataset_weights 0.2 0.2 0.15 0.15 0.1 0.1 0.05 0.05
```

#### 使用中位数（对异常值更鲁棒）

```bash
python3 experiments/profiling/knob3_layers/merge_multi_dataset_importance_scores.py \
    --comparison_dir results/profiling/exp3_importance_comparison \
    --output_file results/layer_importance_scores_multi_dataset.json \
    --method median
```

## 输出文件

脚本会生成两个文件：

1. **完整版本** (`layer_importance_scores_multi_dataset.json`):
   - 包含所有元数据
   - 包含每个数据集的详细 scores
   - 包含 ranking 信息

2. **简化版本** (`layer_importance_scores_multi_dataset_simple.json`):
   - 只包含合并后的 scores（字符串键）
   - 用于其他脚本直接加载

## 自动使用

`run_multi_datasets_h100.py` 和 `run_multi_datasets_a100.py` 已经更新，会自动：

1. 优先使用 `results/layer_importance_scores_multi_dataset_simple.json`
2. 如果不存在，回退到 `results/layer_importance_scores.json`（单数据集版本）

## 工作流程

1. **单数据集合并** (train + validation):
   - 对每个数据集，使用 weighted average 合并 train 和 validation scores
   - 默认权重：train=0.6, validation=0.4

2. **跨数据集合并**:
   - 使用指定的方法（average/weighted_avg/median）合并所有数据集的 scores
   - 生成最终的跨数据集 importance scores

## 重要性说明

- **Block 0** 和 **Block 15**（最后一个）始终会被保留
- 中间 blocks (1-14) 根据 importance scores 进行选择
- 更高的 score = 更重要的 block = 优先保留
- 更低的 score = 不重要的 block = 可以优先 prune

## 验证

运行脚本后，检查输出：
- 查看 ranking（least → most important）
- 确认 top 5 least important blocks（可以优先 prune）
- 确认 top 5 most important blocks（必须保留）

