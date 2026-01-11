# Exp3 Comprehensive Analysis

> **最后更新**: 2026-01-10  
> **版本**: 2.0 (合并了分析指南)

## 概述

本文档提供了对 Exp3 beam search 实验结果的全面分析，包括多个维度的可视化和统计报告，以及如何使用分析工具。

## 生成的所有分析文件

### 📊 可视化图表（9个）

所有图表保存在：`results/profiling/exp3_beam_search_multi_dataset/analysis/`

1. **`rankings_by_num_removed.png`** - 按移除数量分组的排名图
   - 展示移除 1、2、3、4 个 blocks 时的 Top 15 组合
   - 包含误差条（标准差）和数据集数量标注
   - 颜色编码表示测试数据集数量

2. **`task_type_comparison.png`** - 任务类型对比分析
   - 不同任务类型的平均下降、分布、稳定性和组合数量

3. **`block_position_analysis.png`** - Block 位置影响分析
   - Early (0-4)、Middle (5-10)、Late (11-15) blocks 的对比

4. **`combination_vs_individual.png`** - 组合效果 vs 单独效果
   - 实际下降 vs 预测下降的散点图
   - 协同效应（Synergy）分布
   - Top 10 协同组合和拮抗组合

5. **`consistency_analysis.png`** - 跨数据集一致性分析
   - 变异系数（CV）分布
   - 一致性与数据集覆盖度的关系
   - 最一致的组合排名

6. **`dataset_responses.png`** - 数据集特定响应模式
   - 各数据集的平均敏感性
   - 下降分布（小提琴图）
   - 按任务类型分组

7. **`drop_distributions.png`** - 准确率下降分布
   - 整体分布、按移除数量分组、箱线图、累积分布

8. **`dataset_pairwise_comparison.png`** - 数据集两两对比
   - 相关性分布（同任务类型 vs 不同任务类型）
   - 相关性矩阵
   - 任务类型对之间的平均差异
   - Top 5 最相关和最不相关的数据集对

9. **`task_type_detailed_comparison.png`** - 详细任务类型对比
   - 按移除数量分组的平均下降、稳定性、组合数量、中位数

### 📄 文本报告和统计数据

- **`comprehensive_report.txt`** - 综合统计报告
- **`pruning_recommendations.json`** - 剪枝推荐（JSON格式）
- **`pairwise_dataset_stats.json`** - 数据集两两对比统计

## 核心发现

### 1. 按移除数量的最佳选择

#### 移除 1 个 Block
- **最佳**: Block 4
- **平均下降**: 3.09% ± 3.15%
- **测试数据集**: 8 个（全部）
- **稳健性**: ⭐⭐⭐⭐⭐

**Top 5**:
1. Block 4: 3.09% (8 datasets)
2. Block 3: 3.69% (8 datasets)
3. Block 6: 3.95% (8 datasets)
4. Block 11: 4.14% (8 datasets)
5. Block 7: 5.06% (8 datasets)

#### 移除 2 个 Blocks
- **最佳**: [2, 10]
- **平均下降**: 2.42% ± 3.17%
- **测试数据集**: 2 个
- **更稳健选择**: [2, 12] (3 个数据集，下降 3.04%)

**Top 5**:
1. [2, 10]: 2.42% (2 datasets)
2. [2, 12]: 3.04% (3 datasets)
3. [12, 14]: 3.32% (2 datasets)
4. [2, 14]: 3.56% (3 datasets)
5. [2, 15]: 4.27% (2 datasets)

#### 移除 3 个 Blocks
- **最佳**: [12, 13, 14]
- **平均下降**: 6.84% ± 2.15%
- **测试数据集**: 2 个
- **更稳健选择**: [2, 4, 7] (3 个数据集，下降 7.47%)

**Top 5**:
1. [12, 13, 14]: 6.84% (2 datasets)
2. [11, 13, 14]: 7.42% (2 datasets)
3. [2, 4, 7]: 7.47% (3 datasets)
4. [4, 10, 13]: 7.53% (2 datasets)
5. [4, 11, 13]: 7.81% (3 datasets)

#### 移除 4 个 Blocks
- **最佳**: [2, 4, 13, 14]
- **平均下降**: 5.05% ± 2.37%
- **测试数据集**: 2 个
- **更稳健选择**: [3, 4, 13, 14] (3 个数据集，下降 13.14%)

**Top 5**:
1. [2, 4, 13, 14]: 5.05% (2 datasets)
2. [2, 4, 7, 12]: 6.62% (2 datasets)
3. [3, 4, 7, 13]: 9.73% (2 datasets)
4. [3, 4, 13, 14]: 13.14% (3 datasets)
5. [6, 11, 13, 14]: 15.86% (2 datasets)

### 2. 任务类型差异分析

根据分析结果，不同任务类型对 block 移除的敏感性不同：

| 任务类型 | 平均下降 | 标准差 | 组合数 | 最小 | 最大 | 中位数 |
|---------|---------|--------|--------|------|------|--------|
| **Captioning** | 12.59% | 11.79% | 97 | 2.42% | 64.91% | 9.02% |
| **Document QA** | 14.93% | 11.82% | 86 | 3.09% | 64.83% | 12.33% |
| **Exact Match** | 13.89% | 12.84% | 89 | 3.09% | 74.02% | 10.22% |
| **Multiple Choice** | 13.40% | 12.44% | 70 | 3.04% | 64.83% | 9.74% |
| **Scene Text QA** | 16.29% | 13.97% | 96 | 3.09% | 74.02% | 12.36% |
| **VQA** | 15.58% | 15.69% | 124 | 2.42% | 85.29% | 11.44% |

**关键发现**:
- **Captioning** 任务对 block 移除最不敏感（平均下降最小）
- **Scene Text QA** 和 **VQA** 任务最敏感
- **VQA** 任务的变异性最大（标准差 15.69%）

### 3. Block 位置影响

- **Early Blocks (0-4)**: 平均下降较高，说明早期 blocks 通常更重要
- **Middle Blocks (5-10)**: 中等重要性
- **Late Blocks (11-15)**: 相对不重要，但某些组合可能有效

**注意**: Block 0 绝对不能移除（平均下降 62.7%）

### 4. 组合效应分析

- **协同效应（Synergy）**: 某些 block 组合的移除效果小于单独移除之和
- **拮抗效应（Antagonism）**: 某些组合的移除效果大于单独移除之和
- 大多数组合接近线性叠加

**Top 协同组合**: 实际下降显著小于预测下降的组合
**Top 拮抗组合**: 实际下降显著大于预测下降的组合

### 5. 跨数据集一致性

- **变异系数（CV）**: 衡量结果的一致性，越低越可靠
- **数据集覆盖度**: 测试数据集数量越多，结果越稳健
- **最一致的组合**: CV 最低的组合，适合作为通用剪枝方案

### 6. 数据集特定响应

不同数据集对 block 移除的响应模式不同：

- **高敏感性数据集**: 对 block 移除更敏感
- **低敏感性数据集**: 对 block 移除相对不敏感
- **变异性**: 某些数据集的响应变异性较大

### 7. 数据集两两对比

- **同任务类型的数据集**: 通常相关性较高
- **不同任务类型的数据集**: 相关性较低
- **最相关的数据集对**: 对 block 移除的响应模式最相似
- **最不相关的数据集对**: 响应模式差异最大

## 重要洞察

### 1. Block 4 是最不重要的单个 Block
- 在所有 8 个数据集上测试
- 平均下降仅 3.09%，标准差也较小（3.15%）
- 结果稳定可靠

### 2. 任务类型影响显著
- Captioning 任务对 block 移除最不敏感
- VQA 和 Scene Text QA 任务最敏感
- 需要针对不同任务类型制定不同的剪枝策略

### 3. 组合效应存在但有限
- 大多数组合接近线性叠加
- 少数组合存在显著的协同或拮抗效应
- 需要仔细评估组合效果

### 4. 数据集覆盖度很重要
- 只在少数数据集上测试的组合结果不够稳健
- 推荐使用在至少 3 个数据集上测试的组合

### 5. 位置不是绝对规律
- 虽然早期 blocks 通常更重要，但不是绝对规律
- 某些晚期 blocks 的组合可能很有效

## 推荐策略

### 保守策略（推荐用于生产环境）

1. **只移除 Block 4**
   - 准确率下降: 3.09%
   - 测试数据集: 8 个（全部）
   - 稳健性: ⭐⭐⭐⭐⭐

2. **移除 Blocks [4, 13]**
   - 准确率下降: 5.41%
   - 测试数据集: 7 个
   - 增量下降: 2.32%

### 平衡策略

1. **移除 Blocks [2, 4, 7]**
   - 准确率下降: 7.47%
   - 测试数据集: 3 个
   - 移除 3 个 blocks 中最稳健的选择

### 激进策略（追求更高压缩率）

1. **移除 Blocks [2, 4, 13, 14]**
   - 准确率下降: 5.05%
   - 但只在 2 个数据集上测试，需要谨慎

2. **移除 Blocks [3, 4, 13, 14]**
   - 准确率下降: 13.14%
   - 测试数据集: 3 个
   - 更稳健的 4-block 移除选择

## 使用方法

### 生成所有分析

```bash
# 生成所有可视化图表
python3 experiments/profiling/knob3_layers/comprehensive_beam_search_analysis.py

# 生成数据集差异分析
python3 experiments/profiling/knob3_layers/analyze_dataset_differences.py

# 生成综合报告
python3 experiments/profiling/knob3_layers/generate_comprehensive_report.py
```

### 查看结果

```bash
# 查看所有可视化图表
ls -lh results/profiling/exp3_beam_search_multi_dataset/analysis/*.png

# 查看报告
cat results/profiling/exp3_beam_search_multi_dataset/analysis/comprehensive_report.txt

# 查看推荐
cat results/profiling/exp3_beam_search_multi_dataset/analysis/pruning_recommendations.json | jq
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

## 关于 Train/Validation 差异

**注意**: 当前的 beam search 实验使用的是 `train+validation` 组合数据，没有单独的 train 和 validation 结果。这是因为：

1. **增加样本多样性**: 同时使用训练集和验证集
2. **提高统计显著性**: 更多样本提供更可靠的重要性估计
3. **减少样本数需求**: 由于使用了两个 split，可以减少每个数据集的样本数

如果需要分析 train 和 validation 的差异，可以：
1. 查看 `compare_train_val_importance.py` 脚本的结果（如果有）
2. 重新运行实验，分别使用 train 和 validation split
3. 使用数据集间的差异分析来间接了解不同数据分布的影响

## 文件位置

- **分析脚本**: 
  - `experiments/profiling/knob3_layers/comprehensive_beam_search_analysis.py`
  - `experiments/profiling/knob3_layers/analyze_dataset_differences.py`
  - `experiments/profiling/knob3_layers/generate_comprehensive_report.py`
- **结果目录**: `results/profiling/exp3_beam_search_multi_dataset/analysis/`
- **文档**: 
  - `docs/profiling/exp3_comprehensive_analysis_guide.md`
  - `docs/profiling/exp3_comprehensive_analysis_summary.md` (本文档)

## 后续工作建议

1. **验证实验**: 对推荐的最佳组合进行独立验证
2. **任务特定优化**: 针对特定任务类型开发专门的剪枝策略
3. **组合优化**: 进一步探索协同效应强的组合
4. **稳定性测试**: 在不同数据集和条件下测试稳健组合
5. **实际应用**: 在实际部署中测试剪枝效果
6. **Train/Validation 分析**: 如果需要，可以分别运行 train 和 validation 的实验

