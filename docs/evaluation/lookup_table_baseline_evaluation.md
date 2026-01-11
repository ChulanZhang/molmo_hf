# Lookup Table Baseline Controller 评估实验设计

## 概述

本文档描述使用 Lookup Table Baseline Controller 进行完整评估的实验设计。该 baseline controller 基于 core_exp 的 offline profiling 结果，不需要训练，可以直接用于评估 accuracy-latency trade-off。

## 实验目标

1. **验证 Lookup Table Baseline 的有效性**：验证基于 profiling 结果的 lookup table 能否有效找到满足 latency budget 的最优配置
2. **对比 GRPO Controller**：比较 lookup table baseline 与 GRPO-trained controller 的性能
3. **评估不同 Latency Budget**：在 170-380ms 范围内评估多个 budget 点的 accuracy-latency trade-off
4. **多数据集评估**：在多个数据集上评估，验证泛化能力

## 实验设计

### 1. 实验设置

#### 1.1 Latency Budget 范围

基于 core_exp 分析结果，使用以下 latency budget 值：

```python
latency_budgets = [
    170,   # 下界（接近最小 latency）
    200,   # 低 budget
    230,   # 中低 budget
    260,   # 中等 budget
    290,   # 中高 budget
    320,   # 高 budget
    350,   # 接近上界
    380,   # 上界（接近最大 latency）
]
```

**理由**：
- 覆盖完整的 Pareto frontier 范围（170-380ms）
- 均匀分布，便于分析 trade-off 曲线
- 与 GRPO controller 训练时的 budget 范围一致

#### 1.2 评估数据集

参考 AdaLLaVA 的评估设置，使用以下数据集：

**主要数据集**（与 AdaLLaVA 一致）：
- **TextVQA** (`textvqa_val`): Text-based Visual Question Answering
- **VQAv2** (`vqav2_val`): Visual Question Answering v2 (validation)
- **OK-VQA** (`okvqa_val`): Outside Knowledge VQA
- **MME** (`mme`): Multimodal Evaluation
- **POPE** (`pope`): Polling-based Object Probing Evaluation
- **MMBench** (`mmbench_en_dev`): Multimodal Benchmark
- **ScienceQA** (`scienceqa_img`): Science Question Answering

**额外数据集**（项目已有）：
- **DocVQA** (`doc_qa`): Document VQA
- **Scene Text VQA** (`st_qa`): Scene Text VQA
- **TallyQA** (`tally_qa`): TallyQA

#### 1.3 评估指标

**主要指标**：
- **Accuracy**: 任务特定的准确率（VQA accuracy, MME score, etc.）
- **Latency**: 实际推理延迟（ms）
- **Latency Satisfaction**: 满足 budget 的样本比例
- **Accuracy-Latency Trade-off**: Pareto frontier 分析

**辅助指标**：
- **Knob Distribution**: tier, top_k, num_active_blocks 的使用分布
- **FLOPs**: 计算量（如果 LLM-Viewer 可用）
- **Memory Usage**: 内存消耗（如果可用）

### 2. 实验流程

#### 2.1 阶段 1: 构建 Lookup Table

**目标**: 从 core_exp profiling 结果构建 lookup table

```bash
# 构建 lookup table
python experiments/controller/lookup_table_baseline.py \
    --results_dir ./results/core_exp_h100 \
    --output_file ./checkpoints/controller/lookup_table_baseline.json \
    --aggregation_method mean \
    --tolerance 0.05 \
    --datasets coco_2014_vqa text_vqa okvqa
```

**输出**:
- `lookup_table_baseline.json`: 保存的 lookup table
- 统计信息：配置数量、latency 范围、accuracy 范围

#### 2.2 阶段 2: 单数据集评估

**目标**: 在单个数据集上评估不同 latency budget

```bash
# 评估 TextVQA
for budget in 170 200 230 260 290 320 350 380; do
    python experiments/controller/evaluate_lookup_table_baseline.py \
        --model_path checkpoints/molmo \
        --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
        --dataset text_vqa \
        --latency_budget $budget \
        --num_samples 1000 \
        --output_path ./results/results/logs_eval/lookup_table_baseline/text_vqa_budget_${budget}/
done
```

**输出**:
- 每个 budget 的评估结果 JSON
- Accuracy 和 latency 统计
- Knob 分布统计

#### 2.3 阶段 3: 多数据集批量评估

**目标**: 在多个数据集上批量评估

```bash
# 批量评估多个数据集
python experiments/controller/evaluate_lookup_table_baseline_batch.py \
    --model_path checkpoints/molmo \
    --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
    --datasets text_vqa okvqa coco_2014_vqa \
    --latency_budgets 170 200 230 260 290 320 350 380 \
    --num_samples 1000 \
    --output_path ./results/logs_eval/lookup_table_baseline/
```

#### 2.4 阶段 4: LMms-Eval 框架评估

**目标**: 使用标准 lmms-eval 框架评估（参考 AdaLLaVA）

```bash
# 使用 lmms-eval 评估多个 benchmark
python -m experiments.controller.run_lmms_eval_lookup_table \
    --model_path checkpoints/molmo \
    --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
    --tasks textvqa_val,mme,pope,mmbench_en_dev,scienceqa_img,okvqa_val \
    --latency_budget 200.0 \
    --output_path ./results/logs_eval/lookup_table_baseline/lmms_eval/
```

**支持多个 budget**:
```bash
for budget in 170 200 230 260 290 320 350 380; do
    python -m experiments.controller.run_lmms_eval_lookup_table \
        --model_path checkpoints/molmo \
        --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
        --tasks textvqa_val,mme,pope \
        --latency_budget $budget \
        --output_path ./results/results/logs_eval/lookup_table_baseline/lmms_eval_budget_${budget}/
done
```

### 3. 对比实验

#### 3.1 与 GRPO Controller 对比

**目标**: 比较 lookup table baseline 与 GRPO-trained controller

```bash
# 评估 GRPO controller（相同设置）
for budget in 170 200 230 260 290 320 350 380; do
    python experiments/controller/evaluate_adaptive_inference.py \
        --model_path checkpoints/molmo \
        --controller_path checkpoints/controller/grpo_best.pt \
        --dataset text_vqa \
        --latency_budget $budget \
        --num_samples 1000 \
        --output_path ./results/logs_eval/grpo_controller/text_vqa_budget_${budget}/
done
```

**对比维度**:
- Accuracy vs Latency trade-off
- Latency satisfaction rate
- Knob 选择分布
- 计算开销（lookup table 几乎为 0，GRPO 需要推理）

#### 3.2 与静态配置对比

**目标**: 对比 adaptive inference 与固定配置

```bash
# 评估固定配置（baseline）
python experiments/controller/evaluate_static_config.py \
    --model_path checkpoints/molmo \
    --tier medium \
    --top_k 8 \
    --num_active_blocks 16 \
    --dataset text_vqa \
    --num_samples 1000 \
    --output_path ./results/logs_eval/static_config/
```

### 4. 结果分析

#### 4.1 Accuracy-Latency Trade-off 曲线

**分析内容**:
- 绘制 accuracy vs latency 曲线
- 识别 Pareto frontier
- 比较不同 controller 的 trade-off

**可视化**:
```python
# 示例分析脚本
python scripts/analyze_evaluation_results.py \
    --results_dir ./results/results/logs_eval/lookup_table_baseline/ \
    --output_plot ./plots/accuracy_latency_tradeoff.png
```

#### 4.2 Knob 分布分析

**分析内容**:
- 不同 budget 下的 knob 选择分布
- tier, top_k, num_active_blocks 的使用频率
- 与 GRPO controller 的 knob 选择对比

#### 4.3 Latency Satisfaction 分析

**分析内容**:
- 满足 budget 的样本比例
- 超出 budget 的样本分析
- Budget violation 的原因分析

### 5. 实验配置

#### 5.1 硬件要求

- **GPU**: NVIDIA H100 / A100 (与 core_exp profiling 一致)
- **内存**: 至少 40GB GPU memory
- **存储**: 足够的空间存储评估结果

#### 5.2 软件环境

- Python 3.8+
- PyTorch 2.0+
- lmms-eval (最新版本或 AdaLLaVA 指定版本)
- 其他依赖见 `requirements.txt`

#### 5.3 数据准备

- Core exp profiling 结果：`./results/core_exp_h100/`
- 模型 checkpoint：`checkpoints/molmo/`
- 评估数据集：通过 HuggingFace 自动下载或手动准备

### 6. 预期结果

#### 6.1 Lookup Table Baseline 性能

**预期**:
- 在给定 budget 下，能够找到满足 budget 且 accuracy 较高的配置
- Accuracy 可能略低于 GRPO controller（因为无法个性化）
- Latency satisfaction rate 应该较高（基于 profiling 数据）

#### 6.2 与 GRPO Controller 对比

**预期差异**:
- **Accuracy**: GRPO 可能略高（个性化决策）
- **Latency Satisfaction**: Lookup table 可能更高（基于统计最优）
- **Overhead**: Lookup table 几乎为 0，GRPO 有推理开销
- **泛化能力**: GRPO 可能更好（学习到的模式）

#### 6.3 不同 Budget 的表现

**预期趋势**:
- Budget 越高，accuracy 越高（可以使用更多资源）
- 在 Pareto frontier 附近，trade-off 最明显
- 极端 budget（170ms 或 380ms）可能表现较差

### 7. 实验时间估算

#### 7.1 单数据集评估

- **单 budget**: ~30-60 分钟（1000 samples）
- **8 个 budget**: ~4-8 小时

#### 7.2 多数据集评估

- **7 个数据集 × 8 个 budget**: ~28-56 小时

#### 7.3 LMms-Eval 评估

- **单 benchmark**: ~1-2 小时
- **多个 benchmark**: ~5-10 小时

**总计**: 约 2-3 天（单 GPU）

### 8. 实验检查清单

#### 8.1 准备阶段

- [ ] Core exp profiling 结果已生成
- [ ] Lookup table 已构建并验证
- [ ] 模型 checkpoint 可用
- [ ] 评估数据集已准备
- [ ] 环境配置正确

#### 8.2 执行阶段

- [ ] 单数据集评估完成
- [ ] 多数据集批量评估完成
- [ ] LMms-Eval 评估完成
- [ ] 对比实验完成（GRPO controller）
- [ ] 静态配置 baseline 完成

#### 8.3 分析阶段

- [ ] Accuracy-latency trade-off 曲线生成
- [ ] Knob 分布分析完成
- [ ] Latency satisfaction 分析完成
- [ ] 对比分析完成
- [ ] 结果可视化完成

### 9. 输出文件结构

```
results/logs_eval/
├── lookup_table_baseline/
│   ├── text_vqa/
│   │   ├── budget_170/
│   │   │   ├── results.json
│   │   │   ├── knob_distribution.json
│   │   │   └── predictions.jsonl
│   │   ├── budget_200/
│   │   └── ...
│   ├── okvqa/
│   ├── coco_2014_vqa/
│   └── lmms_eval/
│       ├── budget_170/
│       │   ├── textvqa_val.json
│       │   ├── mme.json
│       │   └── ...
│       └── ...
├── grpo_controller/
│   └── ... (相同结构)
└── static_config/
    └── ... (相同结构)
```

### 10. 参考

- [AdaLLaVA GitHub](https://github.com/zhuoyan-xu/AdaLLaVA)
- [AdaLLaVA Paper](https://arxiv.org/pdf/2503.10905)
- [LMms-Eval Documentation](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- [Lookup Table Baseline Controller 文档](../controller/lookup_table_baseline.md)

## 下一步

1. 实施评估代码（见实现部分）
2. 运行实验
3. 分析结果
4. 撰写报告

