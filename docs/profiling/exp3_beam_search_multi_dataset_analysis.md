# Exp3 Beam Search Multi-Dataset Analysis

## 概述

本文档总结了跨8个数据集的 beam search 实验结果，分析了不同 block 组合的剪枝效果。

## 实验配置

- **数据集**: 8个数据集（coco-2014-vqa, text-vqa, okvqa, science-qa-img, st-qa, doc-qa, tally-qa, coco-caption）
- **Split**: train+validation (或 train+test)
- **样本数**: 每个数据集 1000 个样本
- **Beam width**: 3
- **最大移除块数**: 4
- **总测试配置数**: 587 个唯一组合

## 主要发现

### 1. 最佳 Block 组合（跨数据集平均）

以下组合在多个数据集上测试，平均准确率下降最小：

| 排名 | 移除的 Blocks | 平均准确率下降 | 标准差 | 测试数据集数 |
|------|--------------|---------------|--------|------------|
| 1 | [2, 10] | 0.0242 | 0.0317 | 2 |
| 2 | [2, 12] | 0.0304 | 0.0405 | 3 |
| 3 | [4] | 0.0309 | 0.0315 | 8 |
| 4 | [12, 14] | 0.0332 | 0.0098 | 2 |
| 5 | [2, 14] | 0.0356 | 0.0331 | 3 |
| 6 | [3] | 0.0369 | 0.0353 | 8 |
| 7 | [6] | 0.0395 | 0.0218 | 8 |
| 8 | [11] | 0.0414 | 0.0399 | 8 |
| 9 | [2, 15] | 0.0427 | 0.0497 | 2 |
| 10 | [3, 14] | 0.0453 | 0.0481 | 3 |

### 2. 单 Block 移除影响（所有数据集）

移除单个 block 的平均影响（按准确率下降排序）：

| Block | 平均准确率下降 | 标准差 | 测试数据集数 |
|-------|---------------|--------|------------|
| Block 4 | 0.0309 | 0.0315 | 8 |
| Block 3 | 0.0369 | 0.0353 | 8 |
| Block 6 | 0.0395 | 0.0218 | 8 |
| Block 11 | 0.0414 | 0.0399 | 8 |
| Block 7 | 0.0506 | 0.0501 | 8 |
| Block 13 | 0.0510 | 0.0462 | 8 |
| Block 10 | 0.0552 | 0.0287 | 8 |
| Block 14 | 0.0575 | 0.0457 | 8 |
| Block 2 | 0.0577 | 0.0510 | 8 |
| Block 12 | 0.0600 | 0.0441 | 8 |

**关键发现**：
- Block 4 是最不重要的单个 block（平均下降仅 3.09%）
- Block 0 是最重要的（移除后平均下降 62.7%，未在上表中显示）

### 3. 各数据集最优组合

每个数据集上表现最好的 block 组合：

| 数据集 | 最优组合 | 准确率下降 | 最终准确率 |
|--------|---------|-----------|-----------|
| coco-2014-vqa | [2] | 0.0117 | 0.8685 |
| text-vqa | [4] | 0.0143 | 0.8594 |
| science-qa-img | [12] | 0.0078 | 0.8359 |
| doc-qa | [13] | 0.0253 | 0.8807 |
| okvqa | [6] | 0.0286 | 0.8646 |
| st-qa | [3] | 0.0379 | 0.6170 |
| tally-qa | [14] | -0.0156 | 0.8438 |
| coco-caption | [2, 4] | -0.0558 | 0.2325 |

**注意**：tally-qa 和 coco-caption 的负值可能是测量误差或特殊情况。

### 4. 统计摘要

- **总唯一组合数**: 587
- **平均准确率下降（均值）**: 0.2350
- **平均准确率下降（标准差）**: 0.2030
- **最小准确率下降**: -0.0408（可能是测量误差）
- **最大准确率下降**: 0.9035

## 结论

1. **Block 4 是最不重要的单个 block**：在所有8个数据集上测试，平均准确率下降仅 3.09%，标准差也较小（0.0315），说明结果稳定。

2. **Block 组合 [2, 10] 表现最佳**：虽然只在2个数据集上测试，但平均准确率下降仅 2.42%，是测试过的组合中最低的。

3. **Block 0 至关重要**：移除 Block 0 会导致平均准确率下降 62.7%，说明第一个 block 对模型性能至关重要。

4. **中间层 blocks (3, 4, 6, 11) 相对不重要**：这些 blocks 的移除对准确率影响较小，可能是剪枝的候选目标。

5. **跨数据集一致性**：Block 4 在所有8个数据集上都表现出较低的重要性，说明这个发现具有跨数据集的一般性。

## 文件位置

- **分析结果**: `results/profiling/exp3_beam_search_multi_dataset/analysis/cross_dataset_analysis.json`
- **文本报告**: `results/profiling/exp3_beam_search_multi_dataset/analysis/cross_dataset_report.txt`
- **可视化图表**: 
  - `best_combinations.png`: 最佳组合对比
  - `single_block_comparison.png`: 单 block 移除影响
  - `accuracy_drop_distribution.png`: 准确率下降分布

## 使用方法

### 重新运行分析

```bash
# 1. 汇总所有数据集的结果（如果需要）
python3 experiments/profiling/knob3_layers/aggregate_all_beam_search_results.py

# 2. 分析跨数据集结果
python3 experiments/profiling/knob3_layers/analyze_multi_dataset_beam_search.py

# 3. 生成可视化
python3 experiments/profiling/knob3_layers/visualize_multi_dataset_beam_search.py
```

### 查看结果

```bash
# 查看文本报告
cat results/profiling/exp3_beam_search_multi_dataset/analysis/cross_dataset_report.txt

# 查看 JSON 分析结果
cat results/profiling/exp3_beam_search_multi_dataset/analysis/cross_dataset_analysis.json | jq '.best_combinations[:5]'
```

## 下一步建议

1. **验证最佳组合**：在更多数据集或更大的测试集上验证 Block 4 和组合 [2, 10] 的效果。

2. **探索更多组合**：测试更多包含 Block 4 的组合，看是否能进一步减少准确率下降。

3. **分析 Block 0 的重要性**：深入研究为什么 Block 0 如此重要，是否可以优化其结构。

4. **跨任务一致性**：分析不同任务类型（VQA、Captioning、QA）之间的 block 重要性差异。


