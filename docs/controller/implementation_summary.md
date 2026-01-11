# Controller实现总结

> **最后更新**: 2026-01-10  
> **版本**: 3.0 (Joint Training Only)

## 📋 当前实现状态

### ✅ 核心实现

1. **Joint Training** (唯一训练方式)
   - Stage1和Stage2一起训练，共享reward信号
   - 使用GRPO算法进行端到端优化
   - 文件: `train_joint_controller.py`, `joint_grpo_trainer.py`

2. **Direct Latency Measurement**
   - 使用PyTorch hooks直接测量latency
   - Batch size = 1 per sample（确保准确测量）
   - 不使用latency estimator

3. **Budget Token Integration**
   - 编码为d_model维token（2048-D）
   - 在prefill阶段拼接到输入序列
   - Budget encoder MLP可训练，sinusoidal encoding固定

4. **Dynamic Insertion Position**
   - Stage1预测插入位置（1-5）
   - Stage2在插入位置之后运行
   - 文件: `controller.py`, `model_forward_with_dynamic_stage2.py`

5. **Decode Phase Design**
   - 使用prefill阶段决定的配置
   - 不重新运行controller
   - Budget token只在prefill阶段添加

## 🎯 三个Knob设计

| Knob | 控制内容 | 决策时机 | 实现方式 | 输出空间 |
|------|---------|---------|---------|---------|
| **Knob1** | Vision tokens tier + Insertion Position | Before vision encoder | Stage1 predictor | 3 tiers × 5 positions |
| **Knob2** | MoE top-K | After insertion position | Stage2 predictor | 5 choices (4,5,6,7,8) |
| **Knob3** | Total Blocks | After insertion position | Importance-based | 5 choices (12,13,14,15,16) |

**关键约束**:
- 第一层固定: top_k=8，总是包含
- Knob3值表示总block数（包括第一层和插入位置之前的blocks）
- 使用importance-based selection选择blocks

## 📊 训练配置

### 当前设置

- **Latency Budget Range**: [170ms, 380ms] (uniform sampling)
- **Knob2 Options**: [4, 5, 6, 7, 8]
- **Knob3 Options**: [12, 13, 14, 15, 16] (total blocks)
- **Insertion Positions**: [1, 2, 3, 4, 5] (after block 1-5)
- **First Block**: Fixed top_k=8, always included
- **Max New Tokens**: 64
- **Batch Size**: 8 (samples processed one by one, batch_size=1 per sample)

### 训练模块

**Trainable**:
- Stage1 Controller (`knob1_predictor`)
- Stage2 Controller (`knob2_knob3_predictor`)
- Budget Encoder MLP (`budget_encoder.mlp`)

**Frozen**:
- LLM Model
- Budget Encoder Sinusoidal Encoding
- Language Feature Extractor (wte_layer)

## 🔑 关键设计决策

### 1. Joint Training Only

**理由**:
- Stage1和Stage2相互影响，需要协调优化
- 共享reward信号可以全局优化end-to-end性能
- 分阶段训练不合理（已移除）

### 2. Direct Latency Measurement

**理由**:
- 更准确（实际测量而非估计）
- 可以捕获硬件特定的latency特性
- 简化设计（不需要estimator）

**代价**:
- 训练速度较慢（需要实际运行模型）
- 不能使用大batch size

### 3. Budget Token as Token

**理由**:
- 与vision和language token在同一空间
- 经过attention后包含交互信息
- 简化Stage2输入（只需要latency token）

### 4. Dynamic Insertion Position

**理由**:
- 增加灵活性
- Stage1可以根据budget和prompt决定最佳插入位置
- 可以ablation study不同插入位置的影响

### 5. Decode Phase Configuration Preservation

**理由**:
- 减少controller开销
- 保持配置一致性
- 更快的decode速度

## 📁 核心文件

### 模型文件

- `controller.py`: Controller实现（Stage1和Stage2）
- `feature_extractors.py`: 特征提取（Language, Budget）
- `importance_based_block_selection.py`: Block选择工具

### 训练文件

- `train_joint_controller.py`: 主训练脚本
- `joint_grpo_trainer.py`: Joint GRPO训练器
- `online_training_dataset.py`: 在线训练数据集
- `run_training.sh`: 训练脚本

### 推理文件

- `adaptive_inference.py`: 推理引擎
- `test_adaptive_inference.py`: 测试脚本
- `model_forward_with_dynamic_stage2.py`: 动态forward pass

### 工具文件

- `model_loader.py`: 模型加载工具

## 📚 文档结构

### 核心文档（已更新）

1. **README.md**: 主索引文档
2. **OVERVIEW.md**: 快速开始指南
3. **DESIGN.md**: 统一设计文档
4. **JOINT_TRAINING.md**: Joint Training详细说明（合并了JOINT_TRAINING_DESIGN.md）
5. **EXPERIMENTS.md**: 实验文档
6. **training_guide.md**: 训练指南

### 专题文档

7. **DECODE_PHASE_DESIGN.md**: Decode阶段设计
8. **BUDGET_ENCODER_TRAINING.md**: Budget encoder训练
9. **LATENCY_BUDGET_ANALYSIS.md**: Budget范围分析
10. **TRAINING_PRINCIPLE.md**: 训练原则
11. **TRAINING_FAQ.md**: 训练FAQ
12. **TRAINING_MODULES.md**: 训练模块状态

### 独立模块文档（保留）

- **LATENCY_ESTIMATOR_DESIGN.md**: Latency Estimator设计（独立模块）
- **latency_estimator_commands.md**: Latency Estimator命令（独立模块）

## 🚀 快速开始

### 训练

```bash
./experiments/controller/run_training.sh
```

### 测试

```bash
python experiments/controller/test_adaptive_inference.py \
    --model_path checkpoints \
    --controller_path checkpoints/joint_controller/joint_checkpoint_epoch_100.pt \
    --dataset text_vqa \
    --num_samples 100 \
    --latency_budget 200.0 \
    --device cuda
```

## 📝 更新历史

### 2026-01-10 (v3.0)

**重大更新**:
- ✅ 移除分阶段训练，只保留Joint Training
- ✅ 移除controller训练中的latency estimator，使用direct measurement
- ✅ 实现Budget Token集成（编码为token拼接到输入序列）
- ✅ 实现Dynamic Insertion Position（Stage1预测插入位置）
- ✅ 实现Decode Phase配置保持（使用prefill配置）
- ✅ 更新所有文档以反映当前实现
- ✅ 合并重复文档（JOINT_TRAINING.md和JOINT_TRAINING_DESIGN.md）
- ✅ 合并重复文档（LATENCY_BUDGET_TOKEN_DESIGN.md和LATENCY_BUDGET_ENCODING.md）
- ✅ 更新代码注释

**保留**:
- Latency Estimator作为独立模块（不用于controller训练）
- 相关文档保留（作为独立模块参考）

### 文档整理 (2026-01-10)

**已合并文档**:
- `LATENCY_BUDGET_TOKEN_DESIGN.md` + `LATENCY_BUDGET_ENCODING.md` → `LATENCY_BUDGET_ENCODING.md`
- `DOCUMENTATION_UPDATE_SUMMARY.md` → 内容整合到 `IMPLEMENTATION_SUMMARY.md`

**已删除文档**:
- `DIRECT_LATENCY_MEASUREMENT.md` (已实现，内容已整合到DESIGN.md)
- `STAGE2_FEATURE_EXTRACTION.md` (已实现，内容已整合到DESIGN.md)
- `IMPLEMENTATION_STATUS.md` (过时，功能已实现)
- `IMPROVEMENTS_COMPLETED.md` (过时，改进已完成)
- `TRAINING_IMPROVEMENTS.md` (过时，改进已完成)
- `EXPERIMENT_DESIGN_CHECK.md` (过时，设计已确认)
- `DATASET_LOADING_DESIGN.md` (已实现，内容已整合到代码注释)

## 🔗 相关文档

- **设计文档**: `DESIGN.md`, `JOINT_TRAINING.md`
- **训练指南**: `training_guide.md`, `TRAINING_FAQ.md`
- **实验文档**: `EXPERIMENTS.md`
- **专题文档**: `DECODE_PHASE_DESIGN.md`, `BUDGET_ENCODER_TRAINING.md`

---

**维护者**: Controller Team  
**最后更新**: 2026-01-10
