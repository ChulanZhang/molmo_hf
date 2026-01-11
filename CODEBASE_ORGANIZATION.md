# 代码库组织说明

> **创建日期**: 2026-01-11  
> **当前分支**: one-stage-controller (主力开发)

## 分支策略

### main 分支
- **状态**: 稳定版本，包含two-stage controller实现
- **用途**: 作为two-stage实现的稳定参考点

### two-stage-controller 分支
- **状态**: 备份分支
- **内容**: 完整的two-stage controller实现
- **用途**: 备份和参考，不用于主力开发

### one-stage-controller 分支 ⭐
- **状态**: 主力开发分支
- **内容**: One-stage controller实现
- **用途**: 当前主要开发工作在此分支进行

## 代码文件分类

### Two-Stage Controller相关（保留在two-stage-controller分支）

#### 核心文件
- `experiments/controller/controller.py` - Two-stage controller定义
  - `Knob1PredictorBudgetLanguage` - Stage1预测器
  - `Knob2Knob3Predictor` - Stage2预测器
- `experiments/controller/joint_grpo_trainer.py` - Joint GRPO训练器
- `experiments/controller/train_joint_controller.py` - Joint训练脚本

#### 模型Forward
- `experiments/controller/model_forward_with_dynamic_stage2.py` - 动态Stage2插入
- `experiments/controller/model_forward_with_stage2.py` - 固定Stage2插入

#### 特征提取
- `experiments/controller/feature_extractors.py` - 特征提取器
  - `LanguageFeatureExtractor`
  - `LatencyBudgetEncoder`

### One-Stage Controller相关（在one-stage-controller分支开发）

#### 核心文件
- `experiments/controller/one_stage_controller.py` - One-stage controller定义
- `experiments/controller/one_stage_grpo_trainer.py` - One-stage GRPO训练器
- `experiments/controller/train_one_stage_controller.py` - One-stage训练脚本

#### 特征提取
- 复用 `feature_extractors.py` 中的特征提取器
- 添加vision feature提取（在forward之前）

### 共享文件（两个分支都使用）

#### 工具和辅助
- `experiments/controller/model_loader.py` - 模型加载工具
- `experiments/controller/online_training_dataset.py` - 在线训练数据集
- `experiments/controller/importance_based_block_selection.py` - 重要性选择
- `experiments/controller/lookup_table_baseline.py` - Lookup table baseline

#### 评估和测试
- `experiments/controller/test_adaptive_inference.py` - 自适应推理测试
- `experiments/controller/evaluate_*.py` - 各种评估脚本

#### 分析和可视化
- `experiments/controller/analyze_*.py` - 分析脚本
- `experiments/controller/plot_*.py` - 可视化脚本
- `experiments/controller/visualize_*.py` - 可视化工具

## 文档组织

### Two-Stage文档（two-stage-controller分支）
- `docs/controller/design.md` - Two-stage设计文档
- `docs/controller/joint_training.md` - Joint training说明
- `docs/controller/analysis.md` - Two-stage分析

### One-Stage文档（one-stage-controller分支）
- `docs/controller/ONE_STAGE_CONTROLLER.md` - One-stage实现文档
- `docs/controller/FORWARD_PASS_ANALYSIS.md` - Forward pass分析
- `docs/controller/FORWARD_PASS_OPTIMIZATION.md` - Forward pass优化
- `docs/controller/GRPO_CONFIG_GENERATION.md` - GRPO配置生成

### 共享文档
- `docs/controller/overview.md` - Controller概述
- `docs/controller/training_guide.md` - 训练指南
- `docs/controller/training_faq.md` - 训练FAQ

## 迁移计划

### 从Two-Stage到One-Stage

#### 需要移除/替换的文件
1. **Two-stage特定文件**:
   - `model_forward_with_dynamic_stage2.py` → 移除（one-stage不需要）
   - `model_forward_with_stage2.py` → 移除（one-stage不需要）
   - `joint_grpo_trainer.py` → 替换为 `one_stage_grpo_trainer.py`
   - `train_joint_controller.py` → 替换为 `train_one_stage_controller.py`

2. **Controller定义**:
   - `controller.py` 中的two-stage定义 → 替换为one-stage定义

#### 需要修改的文件
1. **模型Forward**:
   - `molmo/models/modeling_molmoe.py` - 移除budget token拼接（one-stage在forward前预测）

2. **特征提取**:
   - `feature_extractors.py` - 添加vision feature提取

#### 需要保留的文件
- 所有评估、分析、可视化工具
- 模型加载工具
- 数据集处理工具
- 重要性选择工具

## 当前工作流程

### 在one-stage-controller分支
1. 开发one-stage controller实现
2. 测试和验证
3. 更新文档

### 参考two-stage-controller分支
1. 查看two-stage实现作为参考
2. 理解设计决策
3. 借鉴有用的代码模式

---

**最后更新**: 2026-01-11  
**维护者**: Controller Team

