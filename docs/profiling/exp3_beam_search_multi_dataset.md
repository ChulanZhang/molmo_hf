# EXP3 Beam Search 多数据集实验配置

## 概述

本文档描述了使用多个数据集进行 beam search 实验的配置，排除了相关性低的数据集，并同时使用训练集和验证集数据以提高统计显著性。

## 数据集选择

### 排除的数据集

**MMMU** (相关性: 0.2558)
- **原因**：
  - 样本量过小 (900 样本)
  - 训练集和验证集相关性极低 (0.2558)
  - 使用不同的 split (dev vs validation)，可能存在分布偏移
  - 统计显著性不足

### 使用的数据集 (8 个)

| 数据集 | Split 组合 | 样本数 (validation) | 相关性 | 答案长度 |
|--------|-----------|---------------------|--------|----------|
| coco_2014_vqa | train+validation | 214,354 | 0.8374 | 1.4 tokens |
| text_vqa | train+validation | 5,000 | 0.9124 | 2.8 tokens |
| okvqa | train+validation | 5,046 | 0.9750 | 2.2 tokens |
| science_qa_img | train+validation | 2,097 | 0.9233 | 1 token (MC) |
| st_qa | train+validation | 1,024 | 0.9882 | 3.0 tokens |
| doc_qa | train+validation | 5,349 | 0.8853 | 4.6 tokens |
| tally_qa | train+test | 26,451 | 0.9036 | 1.0 token |
| coco_caption | train+validation | 40,504 | 0.9882 | - |

**总计**: 8 个数据集，所有数据集的相关性 ≥ 0.83

## 实验配置

### Split 组合策略

使用 `train+validation` 组合来：
1. **增加样本多样性**：同时使用训练集和验证集
2. **提高统计显著性**：更多样本提供更可靠的重要性估计
3. **减少样本数需求**：由于使用了两个 split，可以减少每个数据集的样本数

**实现方式**：
- 使用 PyTorch 的 `ConcatDataset` 组合多个 split
- Split 参数格式：`"train+validation"` 或 `"train+test"`（对于 tally_qa）

### 样本数配置

**之前**：每个数据集 5,000 样本（仅使用 train split）

**现在**：每个数据集 3,000 样本（使用 train+validation 组合）

**理由**：
- 使用两个 split 后，总样本量增加
- 3,000 样本从两个 split 中采样，提供更好的多样性
- 减少实验时间，同时保持统计显著性

### 其他配置

- **Beam width**: 3
- **Max blocks to remove**: 4
- **Batch size**: 16 (自动调整)
- **Max new tokens**: 根据数据集调整 (16/32/64)

## 代码修改

### 1. 支持 Split 组合 (`exp3_accuracy_sensitivity_v2.py`)

添加了支持组合多个 split 的功能：

```python
# Support combining multiple splits (e.g., "train+validation")
if "+" in split:
    splits = [s.strip() for s in split.split("+")]
    datasets = []
    for s in splits:
        try:
            ds = get_dataset_by_name(dataset_name, split=s)
            datasets.append(ds)
            log.info(f"Loaded {dataset_name} {s} split: {len(ds)} samples")
        except Exception as e:
            log.warning(f"Failed to load {dataset_name} {s} split: {e}, skipping")
    if not datasets:
        raise ValueError(f"Failed to load any split for {dataset_name}")
    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    log.info(f"Combined dataset: {len(dataset)} total samples from {len(datasets)} split(s)")
else:
    dataset = get_dataset_by_name(dataset_name, split=split)
```

### 2. 更新数据集列表 (`run_beam_search_experiment.py`)

```python
datasets = [
    ("coco_2014_vqa", "train+validation", 16),
    ("text_vqa", "train+validation", 64),
    ("okvqa", "train+validation", 16),
    ("science_qa_img", "train+validation", 16),
    ("st_qa", "train+validation", 32),
    ("doc_qa", "train+validation", 32),
    ("tally_qa", "train+test", 16),  # tally_qa uses "test" instead of "validation"
    ("coco_caption", "train+validation", 64),
]
```

### 3. 调整样本数

```python
num_samples = 3000  # Reduced from 5000 since we're using both splits
```

## 使用方法

### 运行所有数据集

```bash
python experiments/profiling/knob3_layers/run_beam_search_experiment.py
```

### 运行单个数据集

```bash
python experiments/profiling/knob3_layers/run_beam_search_experiment.py coco_2014_vqa
```

### 直接使用实验脚本

```bash
torchrun --nproc-per-node=4 experiments/profiling/knob3_layers/exp3_accuracy_sensitivity_v2.py \
    --dataset_name coco_2014_vqa \
    --split train+validation \
    --batch_size 16 \
    --num_samples 3000 \
    --beam_width 3 \
    --max_blocks_to_remove 4
```

## 优势

1. **更高的统计显著性**：
   - 使用两个 split 提供更多样本
   - 8 个数据集提供跨任务验证

2. **更好的数据多样性**：
   - 训练集和验证集的数据分布可能略有不同
   - 组合使用可以捕获更全面的重要性模式

3. **排除不可靠数据**：
   - 排除 mmmu（相关性极低）
   - 只使用相关性 ≥ 0.83 的数据集

4. **合理的实验时间**：
   - 减少每个数据集的样本数（3K vs 5K）
   - 但由于使用两个 split，总样本量仍然充足

## 预期结果

- **8 个数据集**的 beam search 结果
- 每个数据集使用 **train+validation** 组合
- 每个数据集评估 **3,000 样本**
- 总共约 **24,000 样本**用于 beam search（8 数据集 × 3K）

## 注意事项

1. **tally_qa 特殊处理**：
   - 使用 `train+test` 而不是 `train+validation`
   - 因为 tally_qa 没有 validation split

2. **样本采样**：
   - 3,000 样本会从组合后的数据集中均匀采样
   - 如果某个 split 的样本数不足，会自动使用全部可用样本

3. **结果保存**：
   - 结果保存在 `results/profiling/exp3_beam_search/{dataset}/train+validation/`
   - Split 名称会包含在路径中

## 相关文档

- [EXP3 实验结果总结](./exp3_results_summary.md)
- [低相关性数据集分析](./exp3_low_correlation_analysis.md)
- [Layer Importance Analysis](./layer_importance_analysis.md)


