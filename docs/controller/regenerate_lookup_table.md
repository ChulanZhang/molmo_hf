# 重新生成 Lookup Table

## 问题说明

之前的 lookup table 使用的是 `T_total`（包含 vision + prefill + decode），但 decode latency 因为输出长度不可控而不适合作为控制目标。

现在已修改为使用 **prefill-only latency**（vision + prefill，不包括 decode）。

## 重新生成步骤

### 方法 1: 使用命令行脚本（推荐）

```bash
python experiments/controller/lookup_table_baseline.py \
    --results_dir ./results/core_exp_h100 \
    --output_file ./checkpoints/controller/lookup_table_baseline.json \
    --datasets text_vqa okvqa coco_2014_vqa \
    --aggregation_method mean \
    --tolerance 0.05 \
    --use_prefill_only
```

参数说明：
- `--results_dir`: Profiling 结果目录
- `--output_file`: 输出的 lookup table 文件路径
- `--datasets`: 要包含的数据集（可选，不指定则使用所有可用数据集）
- `--aggregation_method`: 聚合方法（mean/median/max_accuracy）
- `--tolerance`: 延迟预算容差（0.05 = 5%）
- `--use_prefill_only`: 使用 prefill-only latency（默认启用）
- `--use_total_latency`: 如果指定，则使用 total latency（覆盖 --use_prefill_only）

### 方法 2: 使用 Python API

```python
from experiments.controller.lookup_table_baseline import build_lookup_table_from_core_exp

controller = build_lookup_table_from_core_exp(
    results_dir="./results/core_exp_h100",
    dataset_names=["text_vqa", "okvqa", "coco_2014_vqa"],
    output_file="./checkpoints/controller/lookup_table_baseline.json",
    aggregation_method="mean",
    tolerance=0.05,
    use_prefill_only=True,  # 使用 prefill-only latency
)
```

## Latency 类型说明

### Prefill-Only Latency（推荐）
- **组成**: `T_vision_total + T_LLM_prefill`
- **优点**: 
  - 不包含 decode，更可控
  - 适合作为控制目标
  - 与实际推理延迟更相关（decode 长度不可控）
- **使用场景**: 主要控制目标

### Total Latency
- **组成**: `T_vision_total + T_LLM_prefill + T_LLM_decode`
- **缺点**: 
  - Decode 长度不可控，导致延迟变化大
  - 不适合作为精确的控制目标
- **使用场景**: 仅用于分析，不推荐用于控制

## 验证生成的 Lookup Table

生成后，可以检查统计信息：

```bash
python experiments/controller/lookup_table_wrapper.py \
    --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
    --test_budgets 150 200 250 300
```

这会显示：
- 最小/最大延迟范围
- 不同预算下的配置选择
- 统计信息

## 注意事项

1. **数据源**: 确保 profiling 结果包含 `T_vision_total` 和 `T_LLM_prefill` 字段
2. **覆盖**: 重新生成会覆盖现有的 lookup table 文件
3. **备份**: 建议在重新生成前备份现有文件
4. **一致性**: 确保评估时使用的 latency 类型与 lookup table 一致

