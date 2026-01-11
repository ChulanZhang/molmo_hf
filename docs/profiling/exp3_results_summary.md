# EXP3 实验结果总结

> **最后更新**: 2026-01-10  
> **版本**: 2.0 (合并了快速总结)

## 实验概述

EXP3 实验旨在分析 Molmo VLM 模型中 Transformer 块的重要性，为模型剪枝提供指导。实验包含两个主要部分：

1. **重要性分数一致性分析**：比较训练集和验证集上的重要性分数，验证其一致性
2. **Beam Search 剪枝探索**：使用 beam search 算法探索块组合的剪枝策略

## 快速总结：去掉 1-4 个 Block 的最佳选择

基于 8 个数据集的 beam search 实验结果：

### 去掉 1 个 Block
**推荐：去掉 Block 4**
- 准确率下降：**3.09%** ± 3.15%
- 测试数据集：8 个（全部）
- 稳健性：⭐⭐⭐⭐⭐

### 去掉 2 个 Blocks
**推荐：去掉 Block 4, 13**
- 准确率下降：**5.41%** ± 4.43%
- 测试数据集：7 个
- 增量下降：+2.32%

### 去掉 3 个 Blocks
**推荐：去掉 Block 4, 10, 13**
- 准确率下降：**7.53%** ± 8.73%
- 测试数据集：2 个
- 增量下降：+2.12%

### 去掉 4 个 Blocks
**推荐：去掉 Block 2, 4, 10, 13**
- 准确率下降：**18.36%** ± 18.90%
- 测试数据集：2 个
- 增量下降：+10.83% ⚠️（较大）

### 关键发现

1. **Block 4 是最不重要的单个 block**：在所有数据集上验证，下降仅 3.09%
2. **Block 0 绝对不能移除**：移除后准确率下降 62.7%
3. **逐步移除策略**：4 → 4,13 → 4,10,13 → 2,4,10,13
4. **任务类型差异**：
   - Captioning 最不敏感（平均下降 12.59%）
   - VQA 和 Scene Text QA 最敏感（15-16%）

### 与 run_multi_datasets_h100.py 集成

`run_multi_datasets_h100.py` 使用 `num_active_blocks_list` 参数来控制激活的 block 数量。

**当前配置**：`num_active_blocks_list = [12, 14, 16]`

**对应关系**：
- `num_active_blocks = 12` → 移除 4 个 blocks
- `num_active_blocks = 14` → 移除 2 个 blocks  
- `num_active_blocks = 16` → 不移除（baseline）

**推荐的 block 组合**：
- 移除 2 个：去掉 Block 4, 13 → `active_blocks = [0,1,2,3,5,6,7,8,9,10,11,12,14,15]`
- 移除 4 个：去掉 Block 2, 4, 10, 13 → `active_blocks = [0,1,3,5,6,7,8,9,11,12,14,15]`

---

## 1. 训练集 vs 验证集重要性分数一致性分析

### 实验方法

- **敏感性分析**：逐个移除每个 transformer 块，测量准确率下降
- **重要性分数**：`importance_score[block_i] = accuracy_baseline - accuracy_without_block_i`
- **一致性评估**：使用 Spearman 秩相关系数比较训练集和验证集的重要性分数排序

### 实验结果

**可视化图表**：
- 相关性条形图：`results/profiling/exp3_visualizations/correlation_bar_chart.png`
- 散点图对比：`results/profiling/exp3_visualizations/scatter_comparison.png`
- 重要性热力图：`results/profiling/exp3_visualizations/importance_heatmap.png`

**结果表格**：

| 数据集 | Spearman 相关系数 | P值 | 一致性 |
|--------|------------------|-----|--------|
| st_qa | 0.9882 | 8.11e-13 | ✅ 一致 |
| coco_caption | 0.9882 | 8.11e-13 | ✅ 一致 |
| okvqa | 0.9750 | 1.55e-10 | ✅ 一致 |
| science_qa_img | 0.9233 | 3.41e-07 | ✅ 一致 |
| text_vqa | 0.9124 | 8.38e-07 | ✅ 一致 |
| tally_qa | 0.9036 | 1.60e-06 | ✅ 一致 |
| doc_qa | 0.8853 | 5.15e-06 | ⚠️ 不一致 |
| coco_2014_vqa | 0.8374 | 5.19e-05 | ⚠️ 不一致 |
| mmmu | 0.2558 | 3.39e-01 | ⚠️ 不一致 |

**统计摘要**：
- 总计：9 个数据集
- 一致性数据集：6/9 (66.7%)
- 平均相关系数：0.8521

**LaTeX 表格**：见 `results/profiling/exp3_visualizations/consistency_table.tex`

### 关键发现

1. **极高一致性数据集**（相关系数 > 0.98）：
   - `st_qa`: 相关系数 0.9882，几乎完全一致
   - `coco_caption`: 相关系数 0.9882，几乎完全一致
   - `okvqa`: 相关系数 0.9750，高度一致
   - 这些数据集的重要性分数在训练集和验证集上几乎完全一致

2. **高一致性数据集**（相关系数 0.90-0.98）：
   - `science_qa_img`: 相关系数 0.9233
   - `text_vqa`: 相关系数 0.9124
   - `tally_qa`: 相关系数 0.9036
   - 这些数据集的重要性分数在训练集和验证集上高度一致，可以使用训练集进行敏感性分析

3. **中等一致性数据集**（相关系数 0.80-0.90）：
   - `doc_qa`: 相关系数 0.8853，虽然未达到 0.9 的阈值，但趋势相似
   - `coco_2014_vqa`: 相关系数 0.8374，虽然未达到 0.9 的阈值，但趋势相似
   - 前层（block 0）影响最大，中间层（block 3-4）影响较小，后层影响逐渐增大

4. **低一致性数据集**：
   - `mmmu`: 相关系数 0.2558，相关性较低
   - 可能由于数据集特性或样本量问题导致不一致

3. **通用模式**：
   - **Block 0 最重要**：在所有数据集中，移除 block 0 都会导致最大的准确率下降（0.4-0.8）
   - **中间层（3-8）相对不重要**：这些块的重要性分数通常较低（0.01-0.05）
   - **后层（12-15）重要性中等**：重要性分数在 0.02-0.10 之间

### 结论

- **推荐使用训练集进行敏感性分析**：
  - 训练集样本量更大，统计更可靠
  - 对于高一致性数据集（相关系数 > 0.9），训练集和验证集的结果高度一致
  - 块重要性是模型的结构属性，应该在不同数据分割上保持一致

---

## 2. Beam Search 剪枝探索

### 实验方法

- **算法**：Beam Search，考虑块之间的交互作用
- **策略**：
  - Step 1: 移除 1 个块，测试所有 16 种可能性，保留影响最小的 top-3
  - Step 2: 对每个 top-3 配置，再移除 1 个块，测试剩余可能性，保留 top-3
  - Step 3-4: 继续直到最多移除 4 个块（最少保留 12 个块）

### 实验结果（COCO 2014 VQA）

**示例配置：移除 block 4 和 5**
- 准确率：79.46%
- 准确率下降：5.70%
- 基线准确率：85.15%
- 保留块：0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

### 关键发现

1. **块组合的重要性**：
   - 单独移除某个块的影响与组合移除的影响可能不同
   - Beam search 能够发现比单独重要性分析更优的剪枝策略

2. **中间层可剪枝性**：
   - Block 4-5 的组合移除只导致 5.70% 的准确率下降
   - 这与单独重要性分析一致：中间层（3-8）相对不重要

3. **实验状态**：
   - Beam search 实验仍在进行中
   - 已实现断点续传和自动重试功能
   - 支持 CUDA 错误恢复

---

## 3. 技术改进

### 错误处理
- 实现了 CUDA 错误的自动恢复机制
- 添加了内存清理和垃圾回收
- 改进了错误日志记录

### 断点续传
- 每个配置的结果单独保存
- 自动检测已完成配置，跳过重复计算
- 支持从任意中断点恢复

### 自动重试
- 失败时自动重试（最多 3 次）
- 重试间隔 60 秒
- 智能检测结果完整性

---

## 4. 下一步工作

1. **完成 Beam Search 实验**：
   - 完成所有数据集的 beam search 探索
   - 分析不同数据集上的最优剪枝策略

2. **结果分析**：
   - 比较不同数据集上的剪枝策略
   - 识别通用的剪枝模式
   - 评估剪枝对模型性能的影响

3. **优化建议**：
   - 基于实验结果提出剪枝建议
   - 考虑不同任务类型的最优剪枝策略

---

## 5. 文件结构

```
results/profiling/
├── exp3_importance_comparison/     # 训练集 vs 验证集比较结果
│   ├── coco_2014_vqa/
│   ├── text_vqa/
│   ├── science_qa_img/
│   └── ...
└── exp3_beam_search/               # Beam search 剪枝结果
    ├── coco-2014-vqa/
    │   └── train/
    │       ├── sensitivity_block_*.json
    │       ├── beam_search_step*_*.json
    │       └── exp3_accuracy_sensitivity_v2_results.json
    └── logs/
```

---

## 6. 使用建议

### 进行敏感性分析
```bash
python experiments/profiling/knob3_layers/compare_train_val_importance.py \
    --dataset_name coco_2014_vqa \
    --num_samples 5000
```

### 运行 Beam Search 实验
```bash
python experiments/profiling/knob3_layers/run_beam_search_experiment.py coco_2014_vqa
```

### 查看结果
- 重要性比较：`results/profiling/exp3_importance_comparison/{dataset}/importance_comparison_{dataset}.json`
- Beam search：`results/profiling/exp3_beam_search/{dataset}/train/exp3_accuracy_sensitivity_v2_results.json`

