# Controller实验文档

本文档描述所有controller相关的实验，包括实验目的、脚本、具体解释、期待的输出和分析结果。

## 实验概览

| 实验编号 | 实验名称 | 脚本 | 目的 | 状态 |
|---------|---------|------|------|------|
| Exp 1 | 训练Joint Controller | `train_joint_controller.py` | Joint Training: Stage1+Stage2一起训练（GRPO） | ✅ 可用 |
| Exp 2 | 测试Adaptive Inference | `test_adaptive_inference.py` | 测试完整推理流程 | ✅ 可用 |
| Exp 3 | Importance Score验证 | `validate_importance_consistency.py` | 验证importance score一致性 | ✅ 可用（可选） |
| Exp 4 | Profiling with Importance | `profiling_with_importance.py` | 基于importance的profiling | ✅ 可用（可选） |

**注意**: 
- **Exp 1是核心实验**，必须执行
- Exp 3和Exp 4是可选的分析实验
- 不再需要单独训练Stage1或Stage2（已合并为Joint Training）

---

## Exp 1: 训练Joint Controller (Stage1 + Stage2)

### 实验目的

训练Joint Controller，同时训练Stage1和Stage2，两个阶段共享reward信号，端到端优化。

**关键特点**:
- **Joint Training**: Stage1和Stage2一起训练
- **Direct Latency Measurement**: 使用hooks直接测量latency（不使用estimator）
- **Budget Token**: 编码为token拼接到输入序列
- **Dynamic Insertion**: Stage1决定Stage2的插入位置

### 脚本

```bash
# 使用训练脚本（推荐）
./experiments/controller/run_training.sh

# 或手动运行
python experiments/controller/train_joint_controller.py \
    --results_dir results/core_exp_h100/5run_2000samples_w_new_importance_score \
    --dataset_names text_vqa coco_2014_vqa okvqa \
    --model_path checkpoints \
    --output_dir checkpoints/joint_controller \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 1e-4 \
    --stage1_lr_ratio 1.0 \
    --group_size 5 \
    --device cuda \
    --seed 42 \
    --use_multi_gpu
```

### 具体解释

**输入数据**:
- 从实际数据集加载样本（text_vqa, coco_2014_vqa, okvqa）
- 每个样本包含：`image`, `prompt`, `metadata`, `sample_id`
- **Latency Budget**: 从[170ms, 380ms]均匀随机采样

**模型架构**:
- **Stage1 Controller**: 
  - 输入：Language Feature + Budget Feature
  - 输出：Tier (low/medium/high) + Insertion Position (1-5)
- **Stage2 Controller**:
  - 输入：Latency Token (from LLM after insertion position)
  - 输出：Top-K (4/5/6/7/8) + Total Blocks (12/13/14/15/16)

**训练过程**:
- **GRPO算法**: Group Relative Policy Optimization
- **Reward函数**: Accuracy + Latency Penalty + Budget Violation Penalty
- **Direct Latency Measurement**: 使用PyTorch hooks测量实际latency
- **Batch Size**: 8（但每个样本单独处理，batch_size=1 per sample）

**关键设计**:
- **Budget Token**: 编码为d_model维token，在prefill阶段拼接到输入序列
- **Decode阶段**: 使用prefill阶段决定的配置，不再运行controller
- **Importance-based Selection**: Knob3使用预计算的importance score选择blocks

### 期待的输出

**Checkpoint文件**:
- `checkpoints/joint_controller/joint_checkpoint_epoch_*.pt`
- 每个epoch保存一次checkpoint

**训练日志**:
- 每个epoch的train/val metrics:
  - `loss`: GRPO loss
  - `reward_mean`: 平均reward
  - `accuracy_mean`: 平均accuracy
  - `latency_mean`: 平均latency (ms)
  - `budget_violation_rate`: Budget违反率

**训练时间**:
- 每个iteration约25-30秒（取决于硬件）
- 一个epoch（15000 samples, batch_size=8）约2-3小时

### 故障排除

**问题1**: 训练速度慢
- **原因**: Direct latency measurement需要实际运行模型
- **解决**: 这是正常的，batch_size=1 per sample确保准确测量

**问题2**: Accuracy始终为0
- **原因**: 可能是accuracy计算问题或数据问题
- **解决**: 检查metadata格式，确认ground truth答案存在

**问题3**: Budget violation rate很高
- **原因**: Controller还在学习阶段
- **解决**: 正常现象，随着训练进行会降低

---

## Exp 2: 测试Adaptive Inference

### 实验目的

测试完整的自适应推理流程，评估controller在实际推理中的表现。

### 脚本

```bash
python experiments/controller/test_adaptive_inference.py \
    --model_path checkpoints \
    --controller_path checkpoints/joint_controller/joint_checkpoint_epoch_100.pt \
    --dataset text_vqa \
    --num_samples 100 \
    --latency_budget 200.0 \
    --device cuda
```

### 具体解释

**输入**:
- `--model_path`: 模型路径
- `--controller_path`: Controller checkpoint路径
- `--dataset`: 测试数据集
- `--num_samples`: 测试样本数
- `--latency_budget`: Latency budget (ms)

**测试流程**:
1. 加载模型和controller
2. 对每个样本：
   - Stage1预测tier和insertion position
   - 运行vision encoder（基于tier）
   - 运行LLM到insertion position，提取latency token
   - Stage2预测top_k和num_blocks
   - 运行剩余LLM layers
   - 生成文本
   - 计算accuracy和latency

**输出指标**:
- Average accuracy
- Average latency
- Budget violation rate
- Tier distribution
- Top-K distribution
- Blocks distribution

### 期待的输出

**控制台输出**:
```
Testing adaptive inference...
Sample 1/100: Accuracy=0.85, Latency=195.3ms
Sample 2/100: Accuracy=0.92, Latency=201.7ms
...
Average Accuracy: 0.87
Average Latency: 198.5ms
Budget Violation Rate: 0.12
```

**详细报告**:
- 每个样本的详细结果
- 配置分布统计
- Accuracy-latency trade-off分析

---

## Exp 3: Importance Score验证（可选）

### 实验目的

验证importance score的一致性和有效性。

### 脚本

```bash
python experiments/controller/validate_importance_consistency.py \
    --importance_scores_file results/layer_importance_scores_exp3_recommended.json \
    --device cuda
```

### 具体解释

**输入**:
- Importance scores JSON文件

**验证内容**:
- Importance score格式
- Score值范围
- Block覆盖情况

### 期待的输出

**验证报告**:
- Importance score统计信息
- 验证结果（通过/失败）
- 建议

---

## Exp 4: Profiling with Importance（可选）

### 实验目的

基于importance score进行profiling，生成用于训练的数据。

### 脚本

```bash
python experiments/controller/profiling_with_importance.py \
    --results_dir results/core_exp_h100 \
    --dataset_names text_vqa coco_2014_vqa \
    --importance_scores_file results/layer_importance_scores_exp3_recommended.json \
    --output_dir results/profiling_with_importance \
    --device cuda
```

### 具体解释

**输入**:
- Core experiment结果
- Importance scores

**输出**:
- Profiling结果（包含importance-based block selection）

### 期待的输出

**Profiling结果**:
- 每个配置的accuracy和latency
- Importance-based block selection统计

---

## 实验执行顺序

### 推荐顺序

```
1. Exp 1: 训练Joint Controller（必须）
   ↓
2. Exp 2: 测试Adaptive Inference（必须）
   ↓
3. Exp 3: Importance Score验证（可选）
   ↓
4. Exp 4: Profiling with Importance（可选）
```

### 最小执行集

如果只想快速验证，只需执行：
1. Exp 1: 训练Joint Controller
2. Exp 2: 测试Adaptive Inference

---

## 常见问题

### Q1: 训练需要多长时间？

**A**: 取决于数据集大小和硬件：
- 15000 samples, batch_size=8: 约2-3小时/epoch
- 100 epochs: 约1-2天

### Q2: 可以使用多GPU训练吗？

**A**: 可以，使用`--use_multi_gpu`参数。注意：每个样本仍单独处理（batch_size=1 per sample）。

### Q3: 如何调整latency budget范围？

**A**: 修改`train_joint_controller.py`中的`latency_budget_min`和`latency_budget_max`参数（默认170-380ms）。

### Q4: 训练过程中accuracy为0正常吗？

**A**: 在训练初期可能正常，但如果持续为0，检查：
- Metadata格式是否正确
- Ground truth答案是否存在
- Accuracy计算逻辑是否正确

### Q5: 如何选择最佳checkpoint？

**A**: 选择validation accuracy最高或reward最高的checkpoint。

---

## 相关文档

- **[training_guide.md](training_guide.md)**: 详细训练指南
- **[TRAINING_FAQ.md](TRAINING_FAQ.md)**: 训练常见问题
- **[TRAINING_PRINCIPLE.md](TRAINING_PRINCIPLE.md)**: 训练原则

---

**最后更新**: 2026-01-10  
**维护者**: Controller Team
