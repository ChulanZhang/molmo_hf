# 代码变更主题分析

> **分析日期**: 2026-01-11  
> **基于**: Git diff from HEAD

## 📊 变更概览

根据git diff分析，当前代码变更可以分成以下几个主题：

## 🎯 主题1: Controller架构重构 - Joint Training实现

### 变更文件
- `experiments/controller/controller.py` (180行变更)
  - 更新Knob1Predictor：添加insertion_position预测
  - 更新Knob2Knob3Predictor：简化输入特征（只使用latency_token）
  - 动态knob3选项计算

- `experiments/controller/stage2_grpo_trainer.py` (删除，462行)
  - 删除旧的Stage2单独训练器
  - 功能已整合到joint_grpo_trainer.py

- `experiments/controller/train_two_stage_controller.py` (删除)
  - 删除两阶段分别训练脚本
  - 只保留joint training

### 变更内容
- Stage1现在预测tier和insertion_position
- Stage2只使用latency_token（不再需要单独的vision/language/budget features）
- 动态插入位置支持（1-5层）
- Knob3选项根据插入位置动态调整

---

## 🎯 主题2: Budget Token集成 - AdaLLaVA风格实现

### 变更文件
- `molmo/models/modeling_molmoe.py` (100行变更)
  - 添加latency_budget和budget_encoder参数
  - 在prefill阶段将budget token拼接到输入序列
  - 更新attention_mask和position_ids
  - 只在prefill阶段添加，decode阶段不添加

- `experiments/controller/feature_extractors.py` (52行变更)
  - LatencyBudgetEncoder更新：输出d_model维token
  - 使用sinusoidal encoding (256-D) + MLP (to d_model)
  - 参考AdaLLaVA实现

### 变更内容
- Budget作为token嵌入到输入序列
- 经过transformer blocks后，latency token包含budget+vision+language交互信息
- Budget encoder的MLP可训练，sinusoidal encoding固定

---

## 🎯 主题3: 直接Latency测量 - 移除Estimator依赖

### 变更文件
- `experiments/controller/joint_grpo_trainer.py` (未在diff中，但相关)
  - 移除latency_estimator相关代码
  - 使用hooks直接测量latency
  - Batch size = 1 per sample

- `experiments/controller/test_adaptive_inference.py` (144行变更)
  - 更新测试脚本以支持新的架构
  - 支持动态插入位置
  - 支持budget token

### 变更内容
- 训练和验证都使用direct measurement
- 不再依赖latency estimator进行训练
- Latency estimator保留为独立模块（用于其他用途）

---

## 🎯 主题4: Latency Estimator改进（独立模块）

### 变更文件
- `experiments/controller/latency_estimator.py` (573行变更)
  - 改进estimator架构
  - 位置化decode latency预测
  - 更好的特征工程

- `experiments/controller/train_latency_estimator.py` (110行变更)
  - 更新训练脚本
  - 支持新的estimator架构

### 变更内容
- Latency estimator作为独立模块保留
- 不用于controller训练，但可用于其他分析
- 改进的预测准确性

---

## 🎯 主题5: 工具和辅助脚本更新

### 变更文件
- `experiments/controller/build_lookup_table.py` (2行变更)
  - 小修复

- `experiments/controller/profiling_with_importance.py` (变更)
  - 更新以支持新的importance score格式

- `experiments/controller/validate_importance_consistency.py` (变更)
  - 验证importance score一致性

### 变更内容
- 工具脚本适配新架构
- 支持新的importance score格式

---

## 🎯 主题6: Core Experiment和Profiling更新

### 变更文件
- `experiments/core_exp/run_multi_datasets_h100.py` (6行变更)
  - 小修复或配置更新

- `experiments/profiling/analyze_knob_dataset_correlation.py` (变更)
- `experiments/profiling/knob3_layers/analyze_task_specific_vs_generic.py` (变更)
- `experiments/profiling/knob5_combined/exp6_accuracy.py` (变更)
- `experiments/profiling/plots/analyze_pareto_overlap.py` (变更)
- `experiments/profiling/plots/plot_core_exp_pareto.py` (变更)

### 变更内容
- Profiling脚本更新
- 分析工具改进
- 可视化更新

---

## 📋 建议的Commit分组

### Commit 1: Controller架构重构 - Joint Training
**文件**:
- `experiments/controller/controller.py`
- `experiments/controller/stage2_grpo_trainer.py` (删除)
- `experiments/controller/train_two_stage_controller.py` (删除)

**主题**: 实现joint training架构，Stage1预测insertion position，Stage2简化输入

---

### Commit 2: Budget Token集成
**文件**:
- `molmo/models/modeling_molmoe.py`
- `experiments/controller/feature_extractors.py`

**主题**: 实现AdaLLaVA风格的budget token集成，拼接到输入序列

---

### Commit 3: 直接Latency测量
**文件**:
- `experiments/controller/test_adaptive_inference.py`
- (joint_grpo_trainer.py的变更，如果还没提交)

**主题**: 移除latency estimator依赖，使用direct measurement

---

### Commit 4: Latency Estimator改进（独立模块）
**文件**:
- `experiments/controller/latency_estimator.py`
- `experiments/controller/train_latency_estimator.py`

**主题**: 改进latency estimator作为独立模块

---

### Commit 5: 工具和辅助脚本更新
**文件**:
- `experiments/controller/build_lookup_table.py`
- `experiments/controller/profiling_with_importance.py`
- `experiments/controller/validate_importance_consistency.py`

**主题**: 更新工具脚本以支持新架构

---

### Commit 6: Profiling和分析工具更新
**文件**:
- `experiments/core_exp/run_multi_datasets_h100.py`
- `experiments/profiling/*.py` (多个文件)

**主题**: 更新profiling和分析工具

---

## 🎯 推荐的分组策略

**策略1: 按功能模块分组（推荐）**
1. Controller核心架构变更
2. Budget Token集成
3. Latency测量方式变更
4. 独立模块改进
5. 工具脚本更新

**策略2: 按影响范围分组**
1. 核心模型变更（modeling_molmoe.py）
2. Controller训练变更（controller.py, trainers）
3. 特征提取变更（feature_extractors.py）
4. 工具和分析脚本

**策略3: 按时间顺序分组**
1. 架构设计变更
2. 实现细节变更
3. 工具和测试更新

---

## 📝 下一步

请选择一种分组策略，我可以帮您：
1. 创建多个主题明确的commit
2. 为每个commit生成详细的commit message
3. 确保每个commit都是逻辑完整、可独立测试的

