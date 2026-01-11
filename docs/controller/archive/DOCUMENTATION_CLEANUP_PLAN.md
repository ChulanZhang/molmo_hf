# Controller文档整理计划

> **创建日期**: 2026-01-10  
> **状态**: 待确认

## 📋 文档分类

### ✅ 核心文档（必须保留）

这些文档是controller设计的核心，必须保留并保持最新：

1. **README.md** - 主索引文档 ✅
2. **OVERVIEW.md** - 快速开始指南 ✅
3. **DESIGN.md** - 统一设计文档 ✅
4. **JOINT_TRAINING.md** - Joint Training详细说明 ✅
5. **EXPERIMENTS.md** - 实验文档 ✅
6. **TRAINING_GUIDE.md** - 训练指南 ✅
7. **TRAINING_FAQ.md** - 训练FAQ ✅
8. **TRAINING_PRINCIPLE.md** - 训练原则 ✅
9. **TRAINING_MODULES.md** - 训练模块状态 ✅
10. **DECODE_PHASE_DESIGN.md** - Decode阶段设计 ✅
11. **BUDGET_ENCODER_TRAINING.md** - Budget encoder训练 ✅
12. **LATENCY_BUDGET_ANALYSIS.md** - Budget范围分析 ✅
13. **REWARD_DESIGN_EXPLANATION.md** - Reward设计说明 ✅
14. **EVALUATION_GUIDE.md** - 评估指南 ✅

### 🔄 需要合并的文档

这些文档内容重复或高度相关，应该合并：

1. **LATENCY_BUDGET_TOKEN_DESIGN.md** + **LATENCY_BUDGET_ENCODING.md**
   - **合并为**: `LATENCY_BUDGET_ENCODING.md`
   - **原因**: 两者都描述budget token编码，内容重复
   - **保留内容**: AdaLLaVA实现参考、编码流程、代码示例

2. **IMPLEMENTATION_SUMMARY.md** + **DOCUMENTATION_UPDATE_SUMMARY.md**
   - **合并为**: `IMPLEMENTATION_SUMMARY.md`
   - **原因**: 两者都是总结性文档，内容有重叠
   - **保留内容**: 当前实现状态、关键设计决策、更新历史

### 🗑️ 需要删除的文档（过时或已实现）

这些文档描述的功能已经实现或不再使用：

1. **DIRECT_LATENCY_MEASUREMENT.md** - 已实现，内容已整合到DESIGN.md
2. **STAGE2_FEATURE_EXTRACTION.md** - 已实现，内容已整合到DESIGN.md
3. **IMPLEMENTATION_STATUS.md** - 过时，功能已实现
4. **IMPROVEMENTS_COMPLETED.md** - 过时，改进已完成
5. **TRAINING_IMPROVEMENTS.md** - 过时，改进已完成
6. **EXPERIMENT_DESIGN_CHECK.md** - 过时，设计已确认
7. **DATASET_LOADING_DESIGN.md** - 已实现，内容已整合到代码注释

### 📦 需要归档的文档（保留但移动到archive）

这些文档有参考价值，但不反映当前实现：

1. **LATENCY_ESTIMATOR_DESIGN.md** - 独立模块，保留但标记为独立
2. **LATENCY_ESTIMATOR_COMMANDS.md** - 独立模块，保留但标记为独立
3. **LATENCY_ESTIMATOR_IMPROVEMENT.md** - 独立模块，保留但标记为独立
4. **DECODE_LATENCY_ANALYSIS.md** - 过时分析，归档
5. **DECODE_LATENCY_PREDICTION_CHALLENGE.md** - 过时分析，归档
6. **DECODE_LATENCY_ACCURACY_IMPROVEMENT.md** - 过时分析，归档
7. **POSITIONED_DECODE_LATENCY_TRAINING.md** - 过时，归档
8. **TEXT_FEATURE_ANALYSIS.md** - 过时分析，归档
9. **ANALYSIS.md** - 部分过时，需要更新或归档

### 🔬 研究文档（保留但标记为研究）

这些文档是研究性质的，可能不反映当前实现：

1. **SEMANTIC_ROUTER_INTEGRATION.md** - 研究文档，保留
2. **ADALORA_DESIGNS.md** - 设计研究，保留

### 📊 需要更新的文档

这些文档需要检查并更新以反映当前实现：

1. **TRAINING_ISSUES_FIXED.md** - 需要更新，移除latency estimator相关内容
2. **ANALYSIS.md** - 需要更新或部分归档

## 📁 整理后的文档结构

```
docs/controller/
├── README.md                          # 主索引
├── OVERVIEW.md                        # 快速开始
├── DESIGN.md                          # 统一设计文档
├── JOINT_TRAINING.md                  # Joint Training详细说明
├── EXPERIMENTS.md                     # 实验文档
├── TRAINING_GUIDE.md                  # 训练指南
├── TRAINING_FAQ.md                    # 训练FAQ
├── TRAINING_PRINCIPLE.md              # 训练原则
├── TRAINING_MODULES.md                # 训练模块状态
├── TRAINING_ISSUES_FIXED.md           # 训练问题修复（更新后）
├── DECODE_PHASE_DESIGN.md             # Decode阶段设计
├── BUDGET_ENCODER_TRAINING.md         # Budget encoder训练
├── LATENCY_BUDGET_ANALYSIS.md         # Budget范围分析
├── LATENCY_BUDGET_ENCODING.md         # Budget编码（合并后）
├── REWARD_DESIGN_EXPLANATION.md       # Reward设计说明
├── EVALUATION_GUIDE.md                # 评估指南
├── IMPLEMENTATION_SUMMARY.md          # 实现总结（合并后）
│
├── research/                          # 研究文档
│   ├── SEMANTIC_ROUTER_INTEGRATION.md
│   └── ADALORA_DESIGNS.md
│
└── archive/                           # 归档文档
    ├── LATENCY_ESTIMATOR_DESIGN.md    # 独立模块文档
    ├── LATENCY_ESTIMATOR_COMMANDS.md
    ├── LATENCY_ESTIMATOR_IMPROVEMENT.md
    ├── DECODE_LATENCY_ANALYSIS.md
    ├── DECODE_LATENCY_PREDICTION_CHALLENGE.md
    ├── DECODE_LATENCY_ACCURACY_IMPROVEMENT.md
    ├── POSITIONED_DECODE_LATENCY_TRAINING.md
    ├── TEXT_FEATURE_ANALYSIS.md
    ├── ANALYSIS.md                    # 如果过时
    ├── DIRECT_LATENCY_MEASUREMENT.md  # 已实现
    ├── STAGE2_FEATURE_EXTRACTION.md    # 已实现
    ├── IMPLEMENTATION_STATUS.md       # 过时
    ├── IMPROVEMENTS_COMPLETED.md      # 过时
    ├── TRAINING_IMPROVEMENTS.md       # 过时
    ├── EXPERIMENT_DESIGN_CHECK.md     # 过时
    └── DATASET_LOADING_DESIGN.md      # 已实现
```

## 🔍 其他目录检查

### docs/analysis/
这些文档主要是latency measurement相关的分析，需要检查：
- 是否与当前实现相关
- 是否需要更新
- 是否可以合并

### docs/profiling/
这些文档是profiling实验相关的，需要检查：
- 是否与controller相关
- 是否需要保留

### docs/core_exp/
这些文档是核心实验相关的，需要检查：
- 是否与controller相关
- 是否需要保留

## ❓ 需要用户确认的问题

1. **Latency Estimator文档**：
   - 是否保留为独立模块文档？
   - 还是完全删除？

2. **ANALYSIS.md**：
   - 是否需要更新以反映当前实现？
   - 还是归档？

3. **docs/analysis/** 目录：
   - 这些文档是否还需要？
   - 是否可以合并或删除？

4. **docs/profiling/** 目录：
   - 这些文档是否与controller相关？
   - 是否需要保留？

5. **docs/core_exp/** 目录：
   - 这些文档是否与controller相关？
   - 是否需要保留？

## 📝 执行步骤

1. ✅ 创建整理计划（本文档）
2. ⏳ 等待用户确认
3. ⏳ 合并重复文档
4. ⏳ 更新需要更新的文档
5. ⏳ 创建archive目录并移动归档文档
6. ⏳ 创建research目录并移动研究文档
7. ⏳ 删除过时文档
8. ⏳ 更新README.md索引
9. ⏳ 更新docs/README.md

---

**待用户确认后执行**

