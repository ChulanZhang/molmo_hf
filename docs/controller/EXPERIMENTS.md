# Controller实验文档

本文档描述所有controller相关的实验，包括实验目的、脚本、具体解释、期待的输出和分析结果。

## 实验概览

| 实验编号 | 实验名称 | 脚本 | 目的 | 状态 |
|---------|---------|------|------|------|
| Exp 1 | 训练Latency Estimator | `train_latency_estimator.py` | 训练latency预测模型，用于RL训练加速 | ✅ 可用 |
| Exp 2 | 训练Supervised Controller | `train_supervised_controller.py` | Baseline: 监督学习controller | ✅ 可用 |
| Exp 3 | 训练Two-Stage Controller (Stage 1) | `train_two_stage_controller.py` | 训练Knob1预测器（监督学习） | ✅ 可用 |
| Exp 4 | 训练Two-Stage Controller (Stage 2) | `train_two_stage_controller.py` | 训练Knob2&Knob3预测器（GRPO） | ✅ 可用 |
| Exp 5 | 训练GRPO Controller (完整) | `train_controller_v2.py` | 端到端GRPO训练 | ✅ 可用 |
| Exp 6 | Importance Score验证 | `validate_importance_consistency.py` | 验证importance score一致性 | ✅ 可用 |
| Exp 7 | Profiling with Importance | `profiling_with_importance.py` | 基于importance的profiling | ✅ 可用 |
| Exp 8 | 测试Adaptive Inference | `test_adaptive_inference.py` | 测试完整推理流程 | ✅ 可用 |

---

## Exp 1: 训练Latency Estimator

### 实验目的

训练一个轻量级的latency estimator，用于在RL训练时快速估计latency，避免每次都需要实际运行模型。这样可以：
- 使用更大的batch size进行训练
- 加速训练过程
- 支持不同硬件的latency预测

### 脚本

```bash
python experiments/controller/train_latency_estimator.py \
    --results_dir results/core_exp_h100 \
    --dataset_names text_vqa coco_2014_vqa \
    --output_dir checkpoints/latency_estimator \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 1e-3 \
    --train_split 0.8 \
    --device cuda \
    --seed 42
```

### 具体解释

**输入数据**:
- 从core experiment结果中加载数据
- 每个样本包含：`vision_tokens`, `text_tokens`, `tier_idx`, `top_k`, `num_active_blocks`, `output_tokens`
- 以及对应的latency组件：`T_vision_encoder`, `T_projector`, `T_LLM_prefill`, `T_LLM_decode`, `T_decode_per_token`

**模型架构**:
- 简单的MLP模型
- 输入：配置参数（tier, top_k, num_blocks, token counts）
- 输出：预测的latency组件

**训练过程**:
- 使用MSE loss预测各个latency组件
- 分别预测：`T_vision_encoder`, `T_projector`, `T_LLM_prefill`, `T_decode_per_token`
- 可以计算总latency：`T_total = T_vision + T_projector + T_prefill + T_decode_per_token * output_tokens`

### 期待的输出

**Checkpoint文件**:
- `checkpoints/latency_estimator/best_latency_estimator.pt`

**训练日志**:
- 每个epoch的train/val loss
- 各个latency组件的预测误差（MAE, RMSE）

**期待的分析结果**:
- **MAE < 5ms**: 对于prefill latency
- **MAE < 1ms**: 对于decode per-token latency
- **R² > 0.9**: 预测准确度
- 不同配置下的预测误差分布

---

## Exp 2: 训练Supervised Controller (Baseline)

### 实验目的

训练一个简单的监督学习controller作为baseline。这个controller学习从latency budget直接映射到最优配置（tier, top_k, num_blocks）。

### 脚本

```bash
python experiments/controller/train_supervised_controller.py \
    --profiling_results results/profiling/exp3_importance_comparison/text_vqa/profiling_results.json \
    --output_dir checkpoints/controller/supervised \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 1e-4 \
    --train_split 0.8 \
    --device cuda \
    --seed 42
```

### 具体解释

**输入数据**:
- Profiling结果JSON文件
- 包含不同配置下的latency和accuracy
- 对于每个latency budget，选择最优配置作为label

**模型架构**:
- 简单的MLP
- 输入：latency budget (encoded)
- 输出：三个knob的logits（tier: 3, top_k: 5, num_blocks: 5）

**训练过程**:
- 使用CrossEntropy loss
- 对于每个budget，选择accuracy最高且满足latency约束的配置作为label
- 如果多个配置都满足，选择accuracy最高的

### 期待的输出

**Checkpoint文件**:
- `checkpoints/controller/supervised/best_supervised_controller.pt`

**训练日志**:
- 每个epoch的train/val accuracy
- 每个knob的单独accuracy

**期待的分析结果**:
- **Overall Accuracy > 60%**: 所有三个knob都预测正确
- **Knob1 Accuracy > 70%**: Tier预测准确度
- **Knob2 Accuracy > 65%**: Top-K预测准确度
- **Knob3 Accuracy > 65%**: Num blocks预测准确度
- 不同budget下的准确度分布

---

## Exp 3: 训练Two-Stage Controller (Stage 1)

### 实验目的

训练Stage 1的Knob1预测器，使用监督学习从core experiment数据中学习。

### 脚本

```bash
python experiments/controller/train_two_stage_controller.py \
    --results_dir results/core_exp_h100 \
    --dataset_names text_vqa coco_2014_vqa \
    --model_path checkpoints/molmo \
    --output_dir checkpoints/two_stage_controller \
    --stage stage1 \
    --batch_size 64 \
    --num_epochs_stage1 50 \
    --lr 1e-4 \
    --train_split 0.8 \
    --device cuda \
    --seed 42
```

### 具体解释

**输入数据**:
- 从core experiment结果加载
- 提取language features（使用tokenizer + WTE）
- 提取budget features（使用LatencyBudgetEncoder）
- Label：tier (low/medium/high)

**模型架构**:
- `Knob1PredictorBudgetLanguage`（默认）
- 输入：language feature + budget feature
- 输出：tier logits (3 classes)

**训练过程**:
- 使用CrossEntropy loss
- 可以尝试不同的Knob1变体（通过修改代码）

### 期待的输出

**Checkpoint文件**:
- `checkpoints/two_stage_controller/stage1/best_stage1_checkpoint.pt`

**训练日志**:
- 每个epoch的train/val loss和accuracy
- Tier预测的confusion matrix

**期待的分析结果**:
- **Accuracy > 75%**: Tier预测准确度
- 不同budget下的tier分布
- 不同数据集上的表现一致性

---

## Exp 4: 训练Two-Stage Controller (Stage 2)

### 实验目的

训练Stage 2的Knob2 & Knob3预测器，使用GRPO（Group Relative Policy Optimization）进行强化学习训练。

### 脚本

```bash
# 首先需要训练Stage 1和Latency Estimator
python experiments/controller/train_two_stage_controller.py \
    --results_dir results/core_exp_h100 \
    --dataset_names text_vqa coco_2014_vqa \
    --model_path checkpoints/molmo \
    --latency_estimator_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --output_dir checkpoints/two_stage_controller \
    --stage stage2 \
    --batch_size 32 \
    --num_epochs_stage2 100 \
    --lr 1e-4 \
    --group_size 5 \
    --train_split 0.8 \
    --device cuda \
    --seed 42
```

### 具体解释

**输入数据**:
- 从core experiment结果加载
- 提取vision features（经过encoder+projector，pooled）
- 提取language features
- 提取budget features

**模型架构**:
- `Knob2Knob3Predictor`
- 输入：vision + language + budget features
- 输出：top_k logits (5) + num_blocks logits (5)

**训练过程**:
- 使用GRPO算法
- 对于每个(sample_id, latency_budget)组，生成多个配置
- 使用latency estimator预测latency
- 实际运行模型获取accuracy
- 计算reward：`accuracy - penalty * (latency - budget)`
- 使用group relative ranking进行优化

**Reward函数**:
```
reward = alpha * accuracy - beta * max(0, latency - budget) / budget
```
其中：
- `alpha = 1.0`: Accuracy权重
- `beta = 0.5`: Latency penalty权重

### 期待的输出

**Checkpoint文件**:
- `checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt`

**训练日志**:
- 每个epoch的GRPO loss
- Average reward
- Average accuracy
- Average latency
- Budget adherence rate

**期待的分析结果**:
- **Reward逐渐增加**: 训练收敛
- **Accuracy > baseline**: 比固定配置更好
- **Latency < budget**: 满足latency约束
- **Budget adherence > 80%**: 大部分配置满足budget
- 不同budget下的accuracy-latency trade-off曲线

---

## Exp 5: 训练GRPO Controller (完整)

### 实验目的

端到端训练完整的controller，使用GRPO同时优化所有三个knob。

### 脚本

```bash
python experiments/controller/train_controller_v2.py \
    --results_dir results/core_exp_h100 \
    --dataset_names text_vqa coco_2014_vqa \
    --model_path checkpoints/molmo \
    --output_dir checkpoints/controller/grpo \
    --batch_size 64 \
    --num_epochs 100 \
    --lr 1e-4 \
    --group_size 5 \
    --train_split 0.8 \
    --device cuda \
    --seed 42
```

### 具体解释

**输入数据**:
- 从core experiment结果加载
- 提取所有features（vision, language, budget）

**模型架构**:
- `LLMBasedController`（使用LLM前N层）
- 或者`TwoStageController`（两阶段架构）

**训练过程**:
- 使用GRPO算法
- 同时优化所有三个knob
- 使用latency estimator加速训练

### 期待的输出

**Checkpoint文件**:
- `checkpoints/controller/grpo/best_controller.pt`

**训练日志**:
- GRPO loss
- Average reward
- 各个knob的预测分布

**期待的分析结果**:
- 比两阶段方法更好的accuracy-latency trade-off
- 训练稳定性分析

---

## Exp 6: Importance Score验证

### 实验目的

验证importance score在不同数据集和任务类型上的一致性。

### 脚本

```bash
python experiments/controller/validate_importance_consistency.py \
    --dataset_names text_vqa coco_2014_vqa okvqa science_qa_img \
    --model_path checkpoints/molmo \
    --num_samples 5000 \
    --output_dir results/importance_validation \
    --device cuda
```

### 具体解释

**验证内容**:
1. **Cross-dataset consistency**: 相同任务类型（VQA）在不同数据集上的importance score是否一致
2. **Cross-task consistency**: 不同任务类型（VQA vs ScienceQA）的importance score差异

**方法**:
- 在不同数据集上计算importance score
- 计算correlation coefficient
- 可视化importance score分布

### 期待的输出

**结果文件**:
- `results/importance_validation/cross_dataset_correlation.json`
- `results/importance_validation/importance_comparison_*.png`

**期待的分析结果**:
- **VQA任务**: coco_2014_vqa, text_vqa, okvqa的importance score correlation > 0.9
- **不同任务**: VQA和ScienceQA的importance score有明显差异
- 验证"Data-Agnostic但Task-Dependent"的假设

---

## Exp 7: Profiling with Importance

### 实验目的

使用importance-based block selection进行profiling，收集不同配置下的latency和accuracy数据。

### 脚本

```bash
python experiments/controller/profiling_with_importance.py \
    --model_path checkpoints/molmo \
    --importance_scores_file results/importance/merged_scores.json \
    --datasets text_vqa coco_2014_vqa \
    --output_dir results/profiling/exp3_importance_comparison \
    --num_samples 500 \
    --batch_size 1
```

### 具体解释

**Profiling配置**:
- Tier: low, medium, high
- Top-K: 4, 6, 8, 10, 12
- Num blocks: 8, 10, 12, 14, 16（使用importance-based selection）

**输出数据**:
- 每个配置的latency组件
- Accuracy（VQA score等）
- 用于训练supervised controller和构建lookup table

### 期待的输出

**结果文件**:
- `results/profiling/exp3_importance_comparison/{dataset}/profiling_results.json`

**期待的分析结果**:
- 完整的accuracy-latency trade-off曲线
- 不同配置的性能分布
- 用于后续controller训练的数据

---

## Exp 8: 测试Adaptive Inference

### 实验目的

测试完整的adaptive inference流程，验证controller在实际推理中的表现。

### 脚本

```bash
python experiments/controller/test_adaptive_inference.py \
    --model_path checkpoints/molmo \
    --controller_path checkpoints/two_stage_controller/stage2/best_stage2_checkpoint.pt \
    --dataset text_vqa \
    --num_samples 100 \
    --latency_budget 200.0 \
    --device cuda
```

### 具体解释

**测试流程**:
1. 加载模型和controller
2. 对于每个样本：
   - Stage 1: 预测Knob1 (tier)
   - 处理图像（根据tier）
   - Vision encoding
   - Stage 2: 预测Knob2 & Knob3
   - 应用配置运行LLM
   - 测量latency和accuracy
3. 统计结果

### 期待的输出

**结果文件**:
- 每个样本的配置、latency、accuracy
- 总体统计：平均latency、平均accuracy、budget adherence

**期待的分析结果**:
- **Accuracy > baseline**: 比固定配置更好
- **Latency < budget**: 满足latency约束
- **Budget adherence > 80%**: 大部分样本满足budget
- 不同budget下的accuracy-latency trade-off

---

## 实验执行顺序

### 推荐执行顺序

1. **Exp 6**: Importance Score验证（验证数据一致性）
2. **Exp 7**: Profiling with Importance（收集profiling数据）
3. **Exp 1**: 训练Latency Estimator（为RL训练准备）
4. **Exp 2**: 训练Supervised Controller（Baseline）
5. **Exp 3**: 训练Two-Stage Controller Stage 1（Knob1）
6. **Exp 4**: 训练Two-Stage Controller Stage 2（Knob2&3）
7. **Exp 8**: 测试Adaptive Inference（验证效果）

### 可选实验

- **Exp 5**: 端到端GRPO训练（对比两阶段方法）

---

## 实验配置建议

### 硬件要求

- **GPU**: 至少1x A100或H100
- **内存**: 至少64GB
- **存储**: 至少500GB（用于checkpoints和结果）

### 超参数建议

**Latency Estimator**:
- `lr = 1e-3`
- `batch_size = 64`
- `num_epochs = 50`

**Supervised Controller**:
- `lr = 1e-4`
- `batch_size = 64`
- `num_epochs = 50`

**Two-Stage Controller Stage 1**:
- `lr = 1e-4`
- `batch_size = 64`
- `num_epochs = 50`

**Two-Stage Controller Stage 2**:
- `lr = 1e-4`
- `batch_size = 32`（RL训练需要更小的batch）
- `num_epochs = 100`
- `group_size = 5`

---

## 故障排除

### 常见问题

1. **内存不足**: 减小batch_size或使用gradient accumulation
2. **训练不稳定**: 减小learning rate或使用learning rate scheduling
3. **Latency estimator误差大**: 增加训练数据或调整模型架构
4. **GRPO训练慢**: 使用latency estimator加速，或减小group_size

---

**最后更新**: 2026-01-01
**维护者**: Controller Team





