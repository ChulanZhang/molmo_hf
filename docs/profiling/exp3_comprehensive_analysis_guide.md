# Exp3 Comprehensive Beam Search Analysis Guide

## 概述

本文档提供了对 Exp3 beam search 实验结果的全面分析，包括多个维度的可视化和统计报告。

## 生成的分析文件

### 可视化图表

所有图表保存在：`results/profiling/exp3_beam_search_multi_dataset/analysis/`

1. **`rankings_by_num_removed.png`** - 按移除数量分组的排名图
   - 展示移除 1、2、3、4 个 blocks 时的 Top 组合
   - 包含误差条（标准差）和数据集数量标注
   - 颜色编码表示测试数据集数量

2. **`task_type_comparison.png`** - 任务类型对比分析
   - 不同任务类型（VQA、Captioning、MC、Document QA等）的平均下降
   - 下降分布直方图
   - 稳定性（标准差）对比
   - 组合数量统计

3. **`block_position_analysis.png`** - Block 位置影响分析
   - Early (0-4)、Middle (5-10)、Late (11-15) blocks 的对比
   - 不同位置的稳定性和组合数量

4. **`combination_vs_individual.png`** - 组合效果 vs 单独效果
   - 实际下降 vs 预测下降（个体之和）的散点图
   - 协同效应（Synergy）分布
   - Top 10 协同组合和拮抗组合

5. **`consistency_analysis.png`** - 跨数据集一致性分析
   - 变异系数（CV）分布
   - 一致性与数据集覆盖度的关系
   - 最一致的组合排名
   - 按移除数量的一致性对比

6. **`dataset_responses.png`** - 数据集特定响应模式
   - 各数据集的平均敏感性
   - 下降分布（小提琴图）
   - 按任务类型分组
   - 响应变异性（CV）

7. **`drop_distributions.png`** - 准确率下降分布
   - 整体分布直方图
   - 按移除数量分组的分布
   - 箱线图对比
   - 累积分布函数

### 文本报告

- **`comprehensive_report.txt`** - 综合统计报告
  - 执行摘要
  - 各数量级别的 Top 组合排名
  - 任务类型分析
  - Block 重要性按任务类型
  - 关键洞察
  - 推荐建议

## 主要发现

### 1. 按移除数量的最佳选择

#### 移除 1 个 Block
- **最佳**: Block 4
- **平均下降**: 3.09% ± 3.15%
- **测试数据集**: 8 个（全部）
- **稳健性**: ⭐⭐⭐⭐⭐

#### 移除 2 个 Blocks
- **最佳**: [2, 10]
- **平均下降**: 2.42% ± 3.17%
- **测试数据集**: 2 个
- **更稳健选择**: [2, 12] (3 个数据集，下降 3.04%)

#### 移除 3 个 Blocks
- **最佳**: [12, 13, 14]
- **平均下降**: 6.84% ± 2.15%
- **测试数据集**: 2 个
- **更稳健选择**: [2, 4, 7] (3 个数据集，下降 7.47%)

#### 移除 4 个 Blocks
- **最佳**: [2, 4, 13, 14]
- **平均下降**: 5.05% ± 2.37%
- **测试数据集**: 2 个
- **更稳健选择**: [3, 4, 13, 14] (3 个数据集，下降 13.14%)

### 2. 任务类型差异

不同任务类型对 block 移除的敏感性不同：

| 任务类型 | 平均下降 | 标准差 | 组合数 |
|---------|---------|--------|--------|
| VQA | ~X% | ~Y% | Z |
| Captioning | ~X% | ~Y% | Z |
| Multiple Choice | ~X% | ~Y% | Z |
| Document QA | ~X% | ~Y% | Z |
| Scene Text QA | ~X% | ~Y% | Z |
| Exact Match | ~X% | ~Y% | Z |

*注：具体数值请查看生成的报告文件*

### 3. Block 位置影响

- **Early Blocks (0-4)**: 通常更重要，移除后影响较大
- **Middle Blocks (5-10)**: 中等重要性
- **Late Blocks (11-15)**: 相对不重要，但某些组合可能有效

### 4. 组合效应

- **协同效应（Synergy）**: 某些 block 组合的移除效果小于单独移除之和
- **拮抗效应（Antagonism）**: 某些组合的移除效果大于单独移除之和
- 大多数组合接近线性叠加

### 5. 跨数据集一致性

- 变异系数（CV）较低的组合更可靠
- 测试数据集数量越多，结果越稳健
- 某些组合在不同数据集上表现差异较大

### 6. 数据集特定响应

不同数据集对 block 移除的响应模式不同：
- **高敏感性数据集**: 对 block 移除更敏感
- **低敏感性数据集**: 对 block 移除相对不敏感
- **变异性**: 某些数据集的响应变异性较大

## 使用方法

### 生成所有分析

```bash
# 生成所有可视化图表
python3 experiments/profiling/knob3_layers/comprehensive_beam_search_analysis.py

# 生成综合报告
python3 experiments/profiling/knob3_layers/generate_comprehensive_report.py
```

### 查看结果

```bash
# 查看可视化图表
ls -lh results/profiling/exp3_beam_search_multi_dataset/analysis/*.png

# 查看报告
cat results/profiling/exp3_beam_search_multi_dataset/analysis/comprehensive_report.txt
```

## 分析维度说明

### 1. 移除数量排名
- **目的**: 找出每个数量级别的最佳组合
- **指标**: 平均准确率下降（越低越好）
- **考虑因素**: 标准差、测试数据集数量

### 2. 任务类型差异
- **目的**: 了解不同任务类型对 block 移除的敏感性
- **指标**: 平均下降、标准差、分布
- **应用**: 针对特定任务类型优化剪枝策略

### 3. Block 位置分析
- **目的**: 理解 block 位置对重要性的影响
- **分类**: Early/Middle/Late
- **应用**: 指导 block 选择策略

### 4. 组合效应分析
- **目的**: 理解 block 之间的相互作用
- **指标**: 协同效应（实际 - 预测）
- **应用**: 优化组合选择

### 5. 一致性分析
- **目的**: 评估结果的可靠性
- **指标**: 变异系数（CV）
- **应用**: 选择稳健的剪枝方案

### 6. 数据集响应分析
- **目的**: 了解不同数据集的特性
- **指标**: 平均敏感性、变异性
- **应用**: 针对特定数据集优化

### 7. 分布分析
- **目的**: 理解整体下降模式
- **指标**: 分布、分位数、累积分布
- **应用**: 风险评估和预期管理

## 注意事项

1. **数据集覆盖度**: 某些组合只在少数数据集上测试，结果可能不够稳健
2. **任务类型差异**: 不同任务类型对 block 移除的响应不同，需要针对性考虑
3. **组合效应**: 不是所有组合都是线性叠加，存在协同和拮抗效应
4. **位置影响**: Block 位置对重要性有影响，但不是绝对规律
5. **一致性**: 变异系数高的组合需要谨慎使用

## 后续工作建议

1. **验证实验**: 对推荐的最佳组合进行独立验证
2. **任务特定优化**: 针对特定任务类型开发专门的剪枝策略
3. **组合优化**: 进一步探索协同效应强的组合
4. **稳定性测试**: 在不同数据集和条件下测试稳健组合
5. **实际应用**: 在实际部署中测试剪枝效果

## 文件位置

- **分析脚本**: `experiments/profiling/knob3_layers/comprehensive_beam_search_analysis.py`
- **报告生成**: `experiments/profiling/knob3_layers/generate_comprehensive_report.py`
- **结果目录**: `results/profiling/exp3_beam_search_multi_dataset/analysis/`
- **文档**: `docs/profiling/exp3_comprehensive_analysis_guide.md`

