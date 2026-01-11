# Lookup Table Baseline Controller 评估实现总结

## 已完成的工作

### 1. 文档设计

✅ **实验设计文档** (`lookup_table_baseline_evaluation.md`)
- 完整的实验目标、设置、流程设计
- 8 个 latency budget 点（170-380ms）
- 多数据集评估计划
- 对比实验设计（与 GRPO controller、静态配置）

### 2. 核心代码实现

✅ **Lookup Table Baseline Controller** (`lookup_table_baseline.py`)
- 从 core_exp profiling 结果构建 lookup table
- 支持多种聚合方法（mean, median, max_accuracy）
- 给定 latency budget，查找最优配置
- 支持保存/加载

✅ **Lookup Table Wrapper** (`lookup_table_wrapper.py`)
- 兼容现有 controller 接口
- 提供便捷的创建函数

### 3. 评估代码实现

✅ **单数据集评估** (`evaluate_lookup_table_baseline.py`)
- 评估单个数据集上的性能
- 计算 accuracy、latency、budget violation rate
- 统计 knob 分布
- 保存详细结果

✅ **批量评估** (`evaluate_lookup_table_baseline_batch.py`)
- 支持多数据集、多 budget 批量评估
- 自动生成结果目录结构
- 错误处理和进度跟踪

✅ **LMms-Eval 集成** (`run_lmms_eval_lookup_table.py`)
- 集成标准 lmms-eval 框架
- 参考 AdaLLaVA 的实现方式
- 支持多个 benchmark 评估

✅ **LMms-Eval 适配器** (`lmms_eval_lookup_table_adapter.py`)
- 将 lookup table controller 适配到 lmms-eval
- 实现标准接口
- 统计跟踪

## 需要完善的部分

### 1. 模型 Forward 调用

⚠️ **当前状态**: `evaluate_lookup_table_baseline.py` 中的 `LookupTableInferenceEngine.infer()` 方法使用了占位符实现。

**需要实现**:
```python
# 在 LookupTableInferenceEngine.infer() 中
# 需要根据项目的实际模型 forward 实现来替换占位符

# 当前占位符:
output = "placeholder_output"  # 需要替换

# 应该使用类似这样的实现:
# 1. 根据 tier 处理图像（设置 max_crops）
# 2. 应用 top_k 和 num_active_blocks 配置
# 3. 调用模型的 forward 方法
# 4. 测量实际 latency
```

**参考实现**:
- 查看 `experiments/controller/adaptive_inference.py` 中的 `AdaptiveInferenceEngine.infer()`
- 查看 `experiments/controller/model_forward_with_dynamic_stage2.py` 中的 forward 实现
- 根据项目的实际模型接口调整

### 2. 图像处理（Tier 应用）

⚠️ **需要实现**: 根据 tier 设置正确的 max_crops 并处理图像

**建议**:
```python
from experiments.controller.adaptive_inference import tier_to_max_crops

max_crops = tier_to_max_crops(tier)
# 然后使用 max_crops 处理图像
# 这需要根据项目的图像处理流程来实现
```

### 3. Block Mask 应用

⚠️ **当前实现**: 使用简单的 prefix blocks（前 N 个 blocks）

**建议改进**: 使用 importance-based block selection
```python
# 应该使用 importance scores 来选择 blocks
# 参考: experiments/controller/profiling_with_importance.py
# 或: results/layer_importance_scores_exp3_recommended.json
```

### 4. LMms-Eval 任务集成

⚠️ **当前状态**: `run_single_task()` 是占位符实现

**需要实现**: 完整的 lmms-eval 任务集成
- 加载任务数据集
- 迭代样本
- 调用 adapter.generate()
- 计算任务特定的 metrics

**参考**: AdaLLaVA 的实现方式

## 使用流程

### 步骤 1: 构建 Lookup Table

```bash
python experiments/controller/lookup_table_baseline.py \
    --results_dir ./results/core_exp_h100 \
    --output_file ./checkpoints/controller/lookup_table_baseline.json \
    --aggregation_method mean \
    --tolerance 0.05
```

### 步骤 2: 单数据集评估（需要先完善模型 forward）

```bash
python experiments/controller/evaluate_lookup_table_baseline.py \
    --model_path checkpoints/molmo \
    --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
    --dataset text_vqa \
    --num_samples 100 \
    --latency_budget 200.0
```

### 步骤 3: 批量评估

```bash
python experiments/controller/evaluate_lookup_table_baseline_batch.py \
    --model_path checkpoints/molmo \
    --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
    --datasets text_vqa okvqa coco_2014_vqa \
    --latency_budgets 170 200 230 260 290 320 350 380 \
    --num_samples 1000
```

### 步骤 4: LMms-Eval 评估（需要先完善任务集成）

```bash
python -m experiments.controller.run_lmms_eval_lookup_table \
    --model_path checkpoints/molmo \
    --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
    --tasks textvqa_val,mme,pope \
    --latency_budget 200.0
```

## 文件结构

```
experiments/controller/
├── lookup_table_baseline.py              # ✅ 核心实现
├── lookup_table_wrapper.py               # ✅ Wrapper
├── evaluate_lookup_table_baseline.py     # ✅ 单数据集评估（需完善模型 forward）
├── evaluate_lookup_table_baseline_batch.py # ✅ 批量评估
├── run_lmms_eval_lookup_table.py        # ✅ LMms-Eval 集成（需完善任务集成）
└── lmms_eval_lookup_table_adapter.py    # ✅ LMms-Eval 适配器

docs/evaluation/
├── lookup_table_baseline_evaluation.md   # ✅ 实验设计文档
└── lookup_table_baseline_implementation.md # ✅ 本文档
```

## 下一步行动

### 优先级 1: 完善模型 Forward 调用

1. 查看现有的 `AdaptiveInferenceEngine` 实现
2. 实现 `LookupTableInferenceEngine.infer()` 中的实际模型调用
3. 确保正确应用 tier、top_k、num_active_blocks 配置
4. 测试单数据集评估

### 优先级 2: 完善图像处理

1. 实现根据 tier 设置 max_crops
2. 确保图像处理流程正确
3. 测试不同 tier 的配置

### 优先级 3: 完善 Block Selection

1. 使用 importance-based block selection
2. 加载 importance scores
3. 根据 num_active_blocks 选择正确的 blocks

### 优先级 4: 完善 LMms-Eval 集成

1. 实现完整的任务加载和评估
2. 测试多个 benchmark
3. 确保输出格式符合标准

## 测试建议

### 单元测试

1. 测试 lookup table 构建和加载
2. 测试配置预测（不同 budget）
3. 测试统计信息计算

### 集成测试

1. 测试单数据集评估（小样本）
2. 测试批量评估（少量数据集和 budget）
3. 测试 LMms-Eval 适配器接口

### 完整测试

1. 运行完整的评估流程
2. 对比与 GRPO controller 的结果
3. 验证 accuracy-latency trade-off 曲线

## 参考资源

- [AdaLLaVA GitHub](https://github.com/zhuoyan-xu/AdaLLaVA)
- [AdaLLaVA Paper](https://arxiv.org/pdf/2503.10905)
- [LMms-Eval Documentation](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- [Lookup Table Baseline Controller 文档](../controller/lookup_table_baseline.md)
- [评估指南](./evaluation_guide.md)

## 总结

✅ **已完成**: 
- 完整的实验设计文档
- Lookup table baseline controller 核心实现
- 评估代码框架（单数据集、批量、LMms-Eval）
- 适配器和 wrapper

⚠️ **需要完善**:
- 模型 forward 调用实现
- 图像处理（tier 应用）
- Block selection（importance-based）
- LMms-Eval 任务集成

🎯 **目标**: 完成上述完善后，即可运行完整的评估实验，对比 lookup table baseline 与 GRPO controller 的性能。

