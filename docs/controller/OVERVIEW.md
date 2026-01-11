# Controller Design Overview

> **快速了解Controller设计的核心内容、文档结构和实验流程**

## 📋 目录

1. [核心设计理念](#核心设计理念)
2. [核心文档](#核心文档)
3. [核心代码](#核心代码)
4. [实验流程](#实验流程)
5. [快速开始](#快速开始)

---

## 🎯 核心设计理念

### 两阶段预测架构（Joint Training）

Controller采用**两阶段预测架构**，根据VLM的执行流程约束设计，**两个阶段jointly训练**：

```
Stage 1 (Before Vision Encoder):
  Input: Language Feature + Budget Token (encoded as d_model-dim token)
  Output: Knob1 (Vision Tokens Tier: low/medium/high) + Insertion Position (1-5)
  
Stage 2 (After Insertion Position):
  Input: Latency Token (from LLM after insertion position)
  Output: Knob2 (MoE Top-K: 4/5/6/7/8) + Knob3 (Total Blocks: 12/13/14/15/16)
```

**关键设计**：
- **Joint Training**: Stage1和Stage2一起训练，共享reward信号
- **Dynamic Insertion**: Stage1决定Stage2的插入位置（在block 1-5之后）
- **Budget Token**: 编码为d_model维token，在prefill阶段拼接到输入序列
- **Decode阶段**: 使用prefill阶段决定的配置，不再运行controller

### 三个控制Knob

| Knob | 控制内容 | 决策时机 | 实现方式 | 输出空间 |
|------|---------|---------|---------|---------|
| **Knob1** | Vision tokens tier + Stage2插入位置 | Before vision encoder | Stage 1 predictor | 3 tiers × 5 positions |
| **Knob2** | MoE top-K | After insertion position | Stage 2 predictor | 5 choices (4,5,6,7,8) |
| **Knob3** | Transformer blocks count | After insertion position | **Importance-based pruning** | 5 choices (12-16 total blocks) |

**关键约束**：
- **第一层固定**: Top-K=8，总是包含
- **Importance-based**: Knob3使用预计算的importance score选择blocks
- **Total Blocks**: Knob3的值表示总block数（包括第一层和插入位置之前的blocks）

### 关键设计决策

1. **为什么两阶段？** Knob1必须在vision encoder之前决定，因为crop数量决定图像处理方式
2. **为什么Joint Training？** 两个阶段相互影响，joint training可以端到端优化
3. **Importance-Based Pruning**: Knob3使用预计算的importance score，简化输出空间（从2^16到5）
4. **Direct Latency Measurement**: 使用hooks直接测量latency，不使用estimator
5. **Budget Token**: 编码为token拼接到输入序列，只在prefill阶段添加

---

## 📚 核心文档

### 🎯 必读文档（4个）

1. **[README.md](README.md)** - **主索引文档**
   - 文档导航和快速开始指南
   - 系统架构概览
   - 关键设计决策总结

2. **[DESIGN.md](DESIGN.md)** - **统一设计文档** ⭐⭐⭐
   - **最核心的设计文档**
   - 完整的设计架构
   - 三个Knob的详细设计
   - 输入特征设计
   - 训练方法
   - Overhead分析
   - 实现细节

3. **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - **训练指南** ⭐⭐
   - 完整的训练流程
   - 逐步指导
   - 超参数调优
   - 故障排除

4. **[EXPERIMENTS.md](EXPERIMENTS.md)** - **实验文档** ⭐⭐
   - 实验指南
   - 每个实验的目的、脚本、期待输出
   - 实验执行顺序
   - 故障排除指南

### 📖 技术文档（6个）

5. **[JOINT_TRAINING.md](JOINT_TRAINING.md)** - **Joint Training设计**
   - Joint training架构
   - GRPO算法细节
   - Reward函数设计
   - 训练过程

6. **[DECODE_PHASE_DESIGN.md](DECODE_PHASE_DESIGN.md)** - **Decode阶段设计**
   - Decode阶段实现
   - 配置保持
   - Budget token处理

7. **[BUDGET_ENCODER_TRAINING.md](BUDGET_ENCODER_TRAINING.md)** - **Budget Encoder训练**
   - Budget encoder架构
   - 训练策略
   - Sinusoidal encoding vs MLP

8. **[LATENCY_BUDGET_ANALYSIS.md](LATENCY_BUDGET_ANALYSIS.md)** - **Latency Budget分析**
   - Budget范围确定
   - Pareto frontier分析
   - Budget采样策略

9. **[TRAINING_PRINCIPLE.md](TRAINING_PRINCIPLE.md)** - **训练原则**
   - GRPO训练原则
   - Reward函数设计
   - 训练优化

10. **[TRAINING_FAQ.md](TRAINING_FAQ.md)** - **训练FAQ**
    - 常见问题
    - 故障排除
    - 最佳实践

---

## 💻 核心代码

### 🎯 核心模型（3个文件）

1. **[controller.py](../experiments/controller/controller.py)** - **Controller实现** ⭐⭐⭐
   - `Knob1PredictorBudgetLanguage`: Stage1预测器（tier + insertion position）
   - `Knob2Knob3Predictor`: Stage2预测器（top_k + num_blocks）
   - 支持动态插入位置

2. **[feature_extractors.py](../experiments/controller/feature_extractors.py)** - **特征提取** ⭐⭐
   - `LanguageFeatureExtractor`: 语言特征提取
   - `LatencyBudgetEncoder`: 预算特征编码（sinusoidal + MLP）
   - MLP部分可训练

3. **[importance_based_block_selection.py](../experiments/controller/importance_based_block_selection.py)** - **Block选择** ⭐⭐
   - `load_importance_scores()`: 加载importance scores
   - `select_blocks_by_importance()`: 选择最重要的blocks

### 🔧 训练相关（3个文件）

4. **[train_joint_controller.py](../experiments/controller/train_joint_controller.py)** - **主训练脚本** ⭐⭐⭐
   - Joint Training（Stage1 + Stage2一起训练）
   - 使用GRPO进行end-to-end优化
   - 两个阶段共享reward信号
   - 直接测量latency（hooks）

5. **[joint_grpo_trainer.py](../experiments/controller/joint_grpo_trainer.py)** - **Joint GRPO训练器** ⭐⭐⭐
   - Joint training for Stage1 and Stage2
   - 两个阶段共享reward信号
   - GRPO算法实现
   - Direct latency measurement

6. **[online_training_dataset.py](../experiments/controller/online_training_dataset.py)** - **在线训练数据集** ⭐⭐
   - 从实际数据集加载样本
   - 随机采样latency budget（170-380ms）
   - 支持多数据集

### 📊 推理相关（2个文件）

7. **[adaptive_inference.py](../experiments/controller/adaptive_inference.py)** - **推理引擎** ⭐⭐
   - `AdaptiveInferenceEngine`: 完整的自适应推理引擎
   - 集成两阶段预测和模型执行

8. **[test_adaptive_inference.py](../experiments/controller/test_adaptive_inference.py)** - **测试脚本** ⭐
   - 测试完整推理流程
   - 性能评估

### 🔧 工具脚本（3个文件）

9. **[model_loader.py](../experiments/controller/model_loader.py)** - **模型加载工具**
   - 加载Molmo模型和tokenizer
   - 处理本地路径

10. **[model_forward_with_dynamic_stage2.py](../experiments/controller/model_forward_with_dynamic_stage2.py)** - **动态Forward**
    - 支持动态插入位置的forward pass
    - 提取latency token

11. **[run_training.sh](../experiments/controller/run_training.sh)** - **训练脚本**
    - 一键启动训练

---

## 🧪 实验流程

### 推荐执行顺序

```
1. Train Joint Controller (Stage1 + Stage2)
   ↓
2. Test Adaptive Inference
   ↓
3. Evaluate Performance
```

### 详细实验说明

所有实验的详细说明、脚本命令、期待输出都在 **[EXPERIMENTS.md](EXPERIMENTS.md)** 中。

---

## 🚀 快速开始

### Step 1: 训练Joint Controller (Stage1 + Stage2)

```bash
./experiments/controller/run_training.sh
```

或者手动运行：

```bash
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

**关键参数**:
- `--batch_size 8`: 每个样本单独处理（batch_size=1 per sample）用于准确测量latency
- `--lr 1e-4`: 学习率
- `--stage1_lr_ratio 1.0`: Stage1学习率比例（相对于Stage2）
- `--group_size 5`: GRPO group size
- `--use_multi_gpu`: 多GPU训练（可选）

**输出**: `checkpoints/joint_controller/joint_checkpoint_epoch_*.pt`

**注意**: 
- Joint training同时训练Stage1和Stage2，两个阶段共享reward信号
- 使用direct latency measurement（hooks），不使用latency estimator
- Latency budget从[170ms, 380ms]均匀采样

### Step 2: 测试Adaptive Inference

```bash
python experiments/controller/test_adaptive_inference.py \
    --model_path checkpoints \
    --controller_path checkpoints/joint_controller/joint_checkpoint_epoch_100.pt \
    --dataset text_vqa \
    --num_samples 100 \
    --latency_budget 200.0 \
    --device cuda
```

---

## 📖 文档阅读建议

### 新手入门
1. 先读 **[README.md](README.md)** 了解整体结构
2. 再读 **[DESIGN.md](DESIGN.md)** 第1-3章了解核心设计
3. 最后读 **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** 了解如何训练

### 深入理解
1. **[DESIGN.md](DESIGN.md)** 完整阅读（所有章节）
2. **[JOINT_TRAINING.md](JOINT_TRAINING.md)** Joint Training详细说明
3. **[DECODE_PHASE_DESIGN.md](DECODE_PHASE_DESIGN.md)** Decode阶段设计

### 实施开发
1. **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** 训练指南
2. 查看代码实现：`controller.py`, `train_joint_controller.py`, `joint_grpo_trainer.py`
3. **[EXPERIMENTS.md](EXPERIMENTS.md)** 实验指南

---

## 🔗 相关资源

- **Importance Score分析**: `docs/profiling/`
- **Core Experiment**: `docs/core_exp/`
- **代码实现**: `experiments/controller/`

---

**最后更新**: 2026-01-10  
**维护者**: Controller Team  
**版本**: 3.0 (Joint Training Only)
