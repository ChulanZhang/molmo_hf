# GRPO Controller训练和使用指南

## 概述

本目录包含使用GRPO（Group Relative Policy Optimization）方法训练Controller的完整实现。Controller的任务是根据输入图像和语言特征以及延迟预算，动态选择最优的模型配置（max_crops, top_k, num_active_blocks），在满足延迟约束的前提下最大化准确率。

## 文件结构

```
experiments/controller/
├── GRPO_CONTROLLER_DESIGN.md    # 详细设计文档
├── README.md                     # 本文件
├── data_preparation.py           # 数据准备模块
├── controller_model.py           # Controller模型定义
├── grpo_trainer.py               # GRPO训练器
└── train_controller.py           # 主训练脚本
```

## 快速开始

### 1. 准备数据

首先需要确保exp5和exp6的结果已经生成。然后运行数据准备：

```bash
python experiments/controller/train_controller.py \
    --exp5_dir ./results/profiling/exp5_accuracy \
    --exp6_dir ./results/profiling/exp6_latency \
    --dataset_name text_vqa \
    --model_path checkpoints \
    --output_data_path ./data/controller_training_data.json \
    --force_recompute
```

### 2. 训练Controller

```bash
python experiments/controller/train_controller.py \
    --exp5_dir ./results/profiling/exp5_accuracy \
    --exp6_dir ./results/profiling/exp6_latency \
    --dataset_name text_vqa \
    --model_path checkpoints \
    --output_data_path ./data/controller_training_data.json \
    --output_dir ./checkpoints/controller \
    --batch_size 64 \
    --num_epochs 100 \
    --lr 1e-4 \
    --group_size 8
```

### 3. 使用训练好的Controller

```python
from experiments.controller.controller_model import GRPOController
import torch

# 加载checkpoint
checkpoint = torch.load('./checkpoints/controller/best_checkpoint.pt')
controller = GRPOController(...)
controller.load_state_dict(checkpoint['controller_state_dict'])
controller.eval()

# 使用Controller选择配置
with torch.no_grad():
    logits = controller(image_feat, lang_feat, budget)
    actions = controller.sample_actions(logits, deterministic=True)
    
    max_crops = actions['max_crops'].item()
    top_k = actions['top_k'].item()
    num_active_blocks = actions['num_active_blocks'].item()
    
    # 应用配置到模型
    model.config.max_crops = max_crops
    # ... 设置其他配置
```

## 详细文档

请参考 `GRPO_CONTROLLER_DESIGN.md` 获取完整的设计文档，包括：

- GRPO原理详解
- 系统架构设计
- Reward函数设计
- 训练流程
- Overhead控制
- 与其他方法的比较

## 参数说明

### 数据准备参数

- `--exp5_dir`: exp5结果目录
- `--exp6_dir`: exp6结果目录
- `--dataset_name`: 数据集名称（如text_vqa, okvqa等）
- `--model_path`: 模型checkpoint路径
- `--output_data_path`: 训练数据保存路径
- `--force_recompute`: 强制重新计算特征

### 训练参数

- `--batch_size`: 批次大小（默认64）
- `--num_epochs`: 训练轮数（默认100）
- `--lr`: 学习率（默认1e-4）
- `--group_size`: GRPO组大小（默认8）
- `--train_split`: 训练/验证集划分比例（默认0.8）

### Reward参数

- `--reward_alpha`: 准确率权重（默认1.0）
- `--reward_beta`: 延迟惩罚权重（默认0.5）
- `--reward_gamma`: 预算违反惩罚权重（默认10.0）
- `--reward_delta`: 效率奖励权重（默认0.1）
- `--reward_epsilon`: 复杂度惩罚权重（默认0.05）

## 注意事项

1. **特征提取开销**：首次运行需要提取特征，可能需要较长时间。建议使用`--force_recompute`避免重复计算。

2. **内存使用**：特征提取和训练可能需要较大内存。如果遇到OOM，可以减小`batch_size`。

3. **数据质量**：确保exp5和exp6的结果完整且正确。缺失的数据会影响训练效果。

4. **延迟预算**：训练数据中会为每个样本生成多个延迟预算的变体。默认使用实际延迟的0.8x, 1.0x, 1.2x作为预算。

## 评估

训练过程中会自动在验证集上评估。评估指标包括：

- `reward_mean`: 平均奖励
- `accuracy_mean`: 平均准确率
- `latency_mean`: 平均延迟

最佳模型会自动保存为`best_checkpoint.pt`。

## 故障排除

### 问题：特征提取失败

**原因**：模型路径不正确或数据集加载失败

**解决**：检查`--model_path`和数据集配置

### 问题：训练loss不下降

**原因**：学习率过大或过小，reward函数设计不合理

**解决**：调整学习率，检查reward参数设置

### 问题：内存不足

**原因**：batch_size过大或特征维度太大

**解决**：减小batch_size，或使用特征降维

## 未来改进

- [ ] 支持在线学习
- [ ] 多数据集联合训练
- [ ] 元学习快速适应新预算
- [ ] 可解释性分析工具

