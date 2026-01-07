# EXP3 脚本更新说明

## 更新内容

已更新 `run_multi_datasets_h100.py` 和 `run_multi_datasets_a100.py`，使其按照 EXP3 beam search 推荐结果来移除 blocks。

## 更改详情

### 1. 创建了 EXP3 推荐的 Importance Scores 文件

**文件**: `results/layer_importance_scores_exp3_recommended.json`

这个文件确保当设置 `num_active_blocks` 时，移除的 blocks 严格按照 EXP3 推荐顺序：

| num_active_blocks | 移除的 Blocks | 准确率下降 | 说明 |
|------------------|--------------|-----------|------|
| 15 | Block 4 | 3.09% | 移除 1 个 block |
| 14 | Block 4, 13 | 5.41% | 移除 2 个 blocks（推荐） |
| 13 | Block 4, 10, 13 | 7.53% | 移除 3 个 blocks |
| 12 | Block 2, 4, 10, 13 | 18.36% | 移除 4 个 blocks |
| 16 | 无 | 0% | Baseline（不移除） |

### 2. 更新了实验脚本

两个脚本都已更新：

1. **`experiments/core_exp/run_multi_datasets_h100.py`**
2. **`experiments/core_exp/run_multi_datasets_a100.py`**

**主要更改**：
- 优先使用 `layer_importance_scores_exp3_recommended.json`
- 添加了详细的注释说明每个 `num_active_blocks` 对应的配置
- 保留了 fallback 机制（如果 EXP3 文件不存在，使用其他文件）

### 3. 当前配置

两个脚本的默认配置：

```python
num_active_blocks_list = [12, 14, 16]
```

这对应：
- **12 blocks**: 移除 Block 2, 4, 10, 13（准确率下降 18.36%）
- **14 blocks**: 移除 Block 4, 13（准确率下降 5.41%）- **推荐配置**
- **16 blocks**: 不移除（baseline）

## 使用方法

### 运行实验

```bash
# H100
python experiments/core_exp/run_multi_datasets_h100.py

# A100
python experiments/core_exp/run_multi_datasets_a100.py
```

### 自定义配置

如果需要测试其他配置，可以修改脚本中的 `num_active_blocks_list`：

```python
# 保守策略：只移除 2 个 blocks
num_active_blocks_list = [14, 16]  # 移除 Block 4, 13 和 baseline

# 激进策略：测试所有配置
num_active_blocks_list = [12, 13, 14, 15, 16]  # 移除 4, 3, 2, 1, 0 个 blocks
```

## 验证

运行实验后，可以检查日志确认实际移除的 blocks 是否符合预期。在 `acc_lat_profiling.py` 的日志中会显示：

```
Importance-based selection: keeping blocks [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15]
```

这表示移除了 Block 4 和 13（当 `num_active_blocks = 14` 时）。

## 文件位置

- **Importance Scores**: `results/layer_importance_scores_exp3_recommended.json`
- **生成脚本**: `experiments/profiling/knob3_layers/create_exp3_importance_scores.py`
- **实验脚本**: 
  - `experiments/core_exp/run_multi_datasets_h100.py`
  - `experiments/core_exp/run_multi_datasets_a100.py`

## 注意事项

1. **Block 0 和 15 总是保留**：代码逻辑确保第一个和最后一个 block 不会被移除
2. **Importance Scores 优先级**：脚本会按以下顺序查找文件：
   - `layer_importance_scores_exp3_recommended.json`（EXP3 推荐）
   - `layer_importance_scores_multi_dataset_simple.json`（多数据集合并）
   - `layer_importance_scores.json`（单数据集）
3. **验证移除顺序**：如果发现实际移除的 blocks 与预期不符，检查 importance_scores 文件中的分数顺序

