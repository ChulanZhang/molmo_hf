# EXP3 实验结果保存路径

## 基础路径结构

所有 EXP3 实验结果保存在以下基础目录：

```
results/profiling/exp3_beam_search_multi_dataset/
```

## 完整路径格式

### Beam Search 实验结果

**路径格式**：
```
results/profiling/exp3_beam_search_multi_dataset/{dataset-name}/{split}/
```

**说明**：
- `{dataset-name}`: 数据集名称，下划线替换为连字符（如 `coco_2014_vqa` → `coco-2014-vqa`）
- `{split}`: 数据集 split 名称（如 `train`, `validation`, `train+validation`, `train+test`）

### 具体示例

#### 1. 单个 Split（如 `train`）

```
results/profiling/exp3_beam_search_multi_dataset/coco-2014-vqa/train/
```

#### 2. 组合 Split（如 `train+validation`）

```
results/profiling/exp3_beam_search/coco-2014-vqa/train+validation/
```

**注意**：当使用 `train+validation` 时，split 名称会包含 `+` 符号，路径中会保留这个符号。

## 保存的文件

### 1. 敏感性分析结果

**文件**：`layer_importance_scores.json`

**路径**：
```
results/profiling/exp3_beam_search/{dataset-name}/{split}/layer_importance_scores.json
```

**内容**：
```json
{
  "0": 0.6153,
  "1": 0.0403,
  ...
  "15": 0.0338
}
```

### 2. 单个 Block 敏感性结果（断点续传）

**文件格式**：`sensitivity_block_{block_idx}.json`

**路径**：
```
results/profiling/exp3_beam_search/{dataset-name}/{split}/sensitivity_block_0.json
results/profiling/exp3_beam_search/{dataset-name}/{split}/sensitivity_block_1.json
...
results/profiling/exp3_beam_search/{dataset-name}/{split}/sensitivity_block_15.json
```

**内容**：
```json
{
  "block_idx": 0,
  "baseline_accuracy": 0.8515,
  "ablated_accuracy": 0.2362,
  "importance_score": 0.6153,
  "num_samples": 1000,
  "dataset_name": "coco_2014_vqa",
  "split": "train+validation"
}
```

### 3. Beam Search 最终结果

**文件**：`exp3_accuracy_sensitivity_v2_results.json`

**路径**：
```
results/profiling/exp3_beam_search/{dataset-name}/{split}/exp3_accuracy_sensitivity_v2_results.json
```

**内容**：
- 完整的实验配置
- 敏感性分析结果
- Beam search 所有步骤的结果
- 总结信息

### 4. Beam Search 单个配置结果（断点续传）

**文件格式**：`beam_search_step{step}_blocks{num_active}_removed{blocks}.json`

**路径示例**：
```
results/profiling/exp3_beam_search/{dataset-name}/{split}/beam_search_step1_blocks15_removed4.json
results/profiling/exp3_beam_search/{dataset-name}/{split}/beam_search_step2_blocks14_removed4-5.json
results/profiling/exp3_beam_search/{dataset-name}/{split}/beam_search_step3_blocks13_removed4-5-8.json
```

**内容**：
```json
{
  "step": 2,
  "num_active_blocks": 14,
  "num_total_blocks": 16,
  "active_block_indices": [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  "removed_block_indices": [4, 5],
  "accuracy": 0.7946,
  "accuracy_drop": 0.0570,
  "num_samples": 1000,
  "std": 0.3718,
  "baseline_accuracy": 0.8515
}
```

### 5. 日志文件

**路径**：
```
results/profiling/exp3_beam_search/logs/beam_search_{dataset_name}_{timestamp}.log
```

**示例**：
```
results/profiling/exp3_beam_search/logs/beam_search_coco_2014_vqa_20260104_143022.log
results/profiling/exp3_beam_search/logs/beam_search_coco_2014_vqa_20260104_143022_retry1.log
```

## 完整路径示例

### 当前配置（8 个数据集，train+validation）

```
results/profiling/exp3_beam_search_multi_dataset/
├── logs/
│   ├── beam_search_coco_2014_vqa_20260104_143022.log
│   ├── beam_search_text_vqa_20260104_150000.log
│   └── ...
├── coco-2014-vqa/
│   └── train+validation/
│       ├── layer_importance_scores.json
│       ├── sensitivity_block_0.json
│       ├── sensitivity_block_1.json
│       ├── ...
│       ├── sensitivity_block_15.json
│       ├── beam_search_step1_blocks15_removed0.json
│       ├── beam_search_step1_blocks15_removed1.json
│       ├── ...
│       └── exp3_accuracy_sensitivity_v2_results.json
├── text-vqa/
│   └── train+validation/
│       └── ...
├── okvqa/
│   └── train+validation/
│       └── ...
├── science-qa-img/
│   └── train+validation/
│       └── ...
├── st-qa/
│   └── train+validation/
│       └── ...
├── doc-qa/
│   └── train+validation/
│       └── ...
├── tally-qa/
│   └── train+test/  # tally_qa uses "test" instead of "validation"
│       └── ...
└── coco-caption/
    └── train+validation/
        └── ...
```

## 数据集名称映射

| 数据集名称（代码中） | 目录名称（路径中） |
|---------------------|-------------------|
| `coco_2014_vqa` | `coco-2014-vqa` |
| `text_vqa` | `text-vqa` |
| `okvqa` | `okvqa` |
| `science_qa_img` | `science-qa-img` |
| `st_qa` | `st-qa` |
| `doc_qa` | `doc-qa` |
| `tally_qa` | `tally-qa` |
| `coco_caption` | `coco-caption` |

## 查看结果

### 查看单个数据集的结果

```bash
# 查看最终结果
cat results/profiling/exp3_beam_search_multi_dataset/coco-2014-vqa/train+validation/exp3_accuracy_sensitivity_v2_results.json

# 查看重要性分数
cat results/profiling/exp3_beam_search_multi_dataset/coco-2014-vqa/train+validation/layer_importance_scores.json

# 查看特定配置的结果
cat results/profiling/exp3_beam_search_multi_dataset/coco-2014-vqa/train+validation/beam_search_step2_blocks14_removed4-5.json
```

### 查看所有数据集的结果

```bash
# 列出所有数据集
ls results/profiling/exp3_beam_search_multi_dataset/

# 查看每个数据集的最终结果
for dataset in results/profiling/exp3_beam_search_multi_dataset/*/train+validation/; do
    echo "=== $(basename $(dirname $dataset)) ==="
    cat "$dataset/exp3_accuracy_sensitivity_v2_results.json" | jq '.summary | length'
done
```

## 断点续传

实验支持断点续传，通过检查以下文件来判断是否已完成：

1. **敏感性分析**：检查 `sensitivity_block_{idx}.json` 文件
2. **Beam Search**：检查 `beam_search_step{step}_blocks{num_active}_removed{blocks}.json` 文件

如果文件存在且有效，实验会自动跳过已完成的配置。

## 注意事项

1. **Split 名称中的特殊字符**：
   - `train+validation` 中的 `+` 会保留在路径中
   - 某些文件系统可能对 `+` 有特殊处理，但通常不会有问题

2. **路径长度**：
   - 路径可能较长，但通常不会超过文件系统限制

3. **权限**：
   - 确保有写入权限到 `results/profiling/exp3_beam_search_multi_dataset/` 目录

4. **磁盘空间**：
   - 每个数据集的结果文件可能较大（特别是包含所有配置的 JSON 文件）
   - 8 个数据集可能需要几 GB 的磁盘空间

