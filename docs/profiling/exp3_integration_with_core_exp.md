# EXP3 结果与 Core Exp 集成指南

## 快速答案：去掉 1-4 个 Block 的最佳选择

| 去掉数量 | 去掉的 Blocks | 准确率下降 | 对应 num_active_blocks |
|---------|--------------|-----------|---------------------|
| **1 个** | **[4]** | 3.09% | **15** |
| **2 个** | **[4, 13]** | 5.41% | **14** |
| **3 个** | **[4, 10, 13]** | 7.53% | **13** |
| **4 个** | **[2, 4, 10, 13]** | 18.36% | **12** |

## 在 run_multi_datasets_h100.py 中使用

### 方法 1：直接修改 num_active_blocks_list（推荐）

修改 `run_multi_datasets_h100.py` 中的配置：

```python
# 保守策略：去掉 2 个 blocks（准确率下降 5.41%）
num_active_blocks_list = [14]  # 对应去掉 Block 4, 13

# 激进策略：去掉 4 个 blocks（准确率下降 18.36%）
num_active_blocks_list = [12]  # 对应去掉 Block 2, 4, 10, 13

# 或者测试多个配置
num_active_blocks_list = [12, 14, 16]  # 去掉 4, 2, 0 个 blocks
```

**注意**：`num_active_blocks` 参数会通过 `importance_scores_file` 来选择最重要的 blocks。确保你的 `importance_scores_file` 中 Block 4, 13 等的重要性分数较低。

### 方法 2：创建对应的 importance_scores 文件

如果你想精确控制去掉哪些 blocks，可以创建一个新的 importance_scores 文件，将要去掉的 blocks 的重要性分数设为 0 或很小的值。

```python
# 例如，要去掉 Block 4, 13，创建如下配置：
importance_scores_remove_4_13 = {
    "0": 0.56,   # 高重要性，保留
    "1": 0.08,
    "2": 0.06,
    "3": 0.03,
    "4": 0.0,    # 设为 0，会被优先移除
    "5": 0.08,
    # ... 其他 blocks
    "13": 0.0,   # 设为 0，会被优先移除
    # ...
}
```

### 当前配置说明

查看 `run_multi_datasets_h100.py` 第 223 行：

```python
num_active_blocks_list = [12, 14, 16]  # Number of active transformer blocks
```

这个配置会测试：
- `12 blocks`：去掉 4 个最重要的 blocks（根据 importance_scores）
- `14 blocks`：去掉 2 个最重要的 blocks
- `16 blocks`：保留所有 blocks（baseline）

**与 EXP3 推荐的对应关系**：
- `12 blocks` ≈ 去掉 Block 2, 4, 10, 13（如果 importance_scores 正确）
- `14 blocks` ≈ 去掉 Block 4, 13（如果 importance_scores 正确）

## 验证 Importance Scores

当前使用的文件：`./results/layer_importance_scores_multi_dataset_simple.json`

检查该文件中 Block 4, 13 等的重要性分数是否确实较低：

```bash
cat results/layer_importance_scores_multi_dataset_simple.json | jq
```

如果 Block 4, 13 的重要性分数不是最低的，可能需要：
1. 使用 EXP3 的结果重新生成 importance_scores
2. 或者直接修改代码，指定要移除的 blocks

## 推荐的实验配置

### 保守配置（推荐用于生产）
```python
num_active_blocks_list = [14]  # 去掉 Block 4, 13，准确率下降 5.41%
```

### 平衡配置
```python
num_active_blocks_list = [15, 14]  # 去掉 1 个和 2 个 blocks
```

### 激进配置（测试用）
```python
num_active_blocks_list = [12]  # 去掉 Block 2, 4, 10, 13，准确率下降 18.36%
```

## 完整配置示例

```python
# 在 run_multi_datasets_h100.py 中修改：

# 使用 EXP3 推荐的配置
num_active_blocks_list = [15, 14, 13, 12]  # 分别去掉 1, 2, 3, 4 个 blocks

# 对应的准确率下降预期：
# 15 blocks: ~3.09% drop (去掉 Block 4)
# 14 blocks: ~5.41% drop (去掉 Block 4, 13)
# 13 blocks: ~7.53% drop (去掉 Block 4, 10, 13)
# 12 blocks: ~18.36% drop (去掉 Block 2, 4, 10, 13)
```

## 注意事项

1. **Importance Scores 必须匹配**：`num_active_blocks` 会根据 importance_scores 选择最重要的 blocks。如果 importance_scores 与 EXP3 推荐不一致，实际移除的 blocks 可能不同。

2. **验证实际移除的 Blocks**：运行实验后，检查日志或结果文件，确认实际移除的 blocks 是否符合预期。

3. **任务类型差异**：不同任务类型对 block 移除的敏感性不同，VQA 任务比 Captioning 更敏感。

## 快速参考

- **EXP3 结果目录**：`results/profiling/exp3_beam_search_multi_dataset/analysis/`
- **推荐配置 JSON**：`pruning_recommendations.json`
- **Block 配置 JSON**：`block_configs_for_core_exp.json`
- **快速总结**：`docs/profiling/exp3_quick_summary.md`

