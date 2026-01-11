# 文档整理总结

> **整理日期**: 2026-01-10  
> **状态**: 进行中

## ✅ 已完成的整理

### 1. Controller目录整理

#### 合并的文档
- ✅ `LATENCY_BUDGET_TOKEN_DESIGN.md` + `LATENCY_BUDGET_ENCODING.md` → `LATENCY_BUDGET_ENCODING.md`
- ✅ `DOCUMENTATION_UPDATE_SUMMARY.md` → 内容整合到 `IMPLEMENTATION_SUMMARY.md`

#### 删除的文档
- ✅ `LATENCY_BUDGET_TOKEN_DESIGN.md` (已合并)
- ✅ `DOCUMENTATION_UPDATE_SUMMARY.md` (已整合)

#### 归档的文档（移动到archive/）
- ✅ `DIRECT_LATENCY_MEASUREMENT.md` (已实现，内容已整合到DESIGN.md)
- ✅ `STAGE2_FEATURE_EXTRACTION.md` (已实现，内容已整合到DESIGN.md)
- ✅ `IMPLEMENTATION_STATUS.md` (过时，功能已实现)
- ✅ `IMPROVEMENTS_COMPLETED.md` (过时，改进已完成)
- ✅ `TRAINING_IMPROVEMENTS.md` (过时，改进已完成)
- ✅ `EXPERIMENT_DESIGN_CHECK.md` (过时，设计已确认)
- ✅ `DATASET_LOADING_DESIGN.md` (已实现，内容已整合到代码注释)
- ✅ `DECODE_LATENCY_ANALYSIS.md` (过时，latency estimator相关)
- ✅ `DECODE_LATENCY_PREDICTION_CHALLENGE.md` (过时，latency estimator相关)
- ✅ `DECODE_LATENCY_ACCURACY_IMPROVEMENT.md` (过时，latency estimator相关)
- ✅ `POSITIONED_DECODE_LATENCY_TRAINING.md` (过时，latency estimator相关)
- ✅ `TEXT_FEATURE_ANALYSIS.md` (过时分析)

#### 研究文档（移动到research/）
- ✅ `SEMANTIC_ROUTER_INTEGRATION.md` (研究文档)
- ✅ `ADALORA_DESIGNS.md` (设计研究)

## ⏳ 待处理的文档

### Profiling目录（docs/profiling/）

发现多个exp3相关的总结文档，可能可以合并：

1. **exp3_quick_summary.md** - 快速总结（推荐去掉1-4个block的最佳选择）
2. **exp3_results_summary.md** - 实验结果总结（训练集vs验证集一致性分析 + Beam Search）
3. **exp3_comprehensive_analysis_summary.md** - 全面分析总结（9个可视化图表）
4. **exp3_comprehensive_analysis_guide.md** - 全面分析指南（如何使用分析工具）

**建议**：
- `exp3_quick_summary.md` + `exp3_results_summary.md` → 合并为 `exp3_results_summary.md`
- `exp3_comprehensive_analysis_summary.md` + `exp3_comprehensive_analysis_guide.md` → 合并为 `exp3_comprehensive_analysis.md`

**需要用户确认**：
- 这些文档是否都需要保留？
- 是否可以合并？

### Analysis目录（docs/analysis/）

这些文档主要是latency measurement相关的分析，对理解测量机制有价值：

1. **key_insights_latency_measurement.md** - 关键洞察总结 ⭐
2. **latency_measurement_issue_summary.md** - 问题总结
3. **latency_measurement_code_locations.md** - 代码位置详解
4. **latency_measurement_refactoring.md** - 完整重构文档 ⭐
5. **decode_measurement_strategy.md** - Decode测量策略
6. **tier_fallback_analysis.md** - Tier fallback分析

**建议**：
- 这些文档对理解latency measurement机制有价值，建议保留
- 可以考虑合并一些内容，但需要谨慎，因为每个文档有不同的侧重点

**需要用户确认**：
- 这些文档是否都需要保留？
- 是否可以合并某些文档？

### Latency Estimator文档

根据用户要求，Latency Estimator作为独立模块保留：

- `LATENCY_ESTIMATOR_DESIGN.md` - 保留（独立模块）
- `latency_estimator_commands.md` - 保留（独立模块）
- `LATENCY_ESTIMATOR_IMPROVEMENT.md` - 需要检查，如果与controller无关，保留；如果相关，归档

## 📋 待确认的问题

### 1. Profiling目录的exp3文档

**问题**：有4个exp3相关的总结文档，内容可能有重叠。

**建议**：
- 合并 `exp3_quick_summary.md` 和 `exp3_results_summary.md` 为 `exp3_results_summary.md`
- 合并 `exp3_comprehensive_analysis_summary.md` 和 `exp3_comprehensive_analysis_guide.md` 为 `exp3_comprehensive_analysis.md`

**请确认**：是否同意合并？

### 2. Analysis目录的文档

**问题**：这些文档主要是latency measurement相关的分析，对理解测量机制有价值。

**建议**：保留所有文档，因为它们有不同的侧重点。

**请确认**：是否同意保留所有文档？

### 3. ANALYSIS.md

**问题**：`docs/controller/ANALYSIS.md` 是controller分析文档，但最后更新是2026-01-01，可能部分过时。

**建议**：检查并更新，或者如果过时则归档。

**请确认**：是否需要检查并更新这个文档？

## 📁 整理后的目录结构

```
docs/controller/
├── README.md                          # 主索引
├── OVERVIEW.md                        # 快速开始
├── DESIGN.md                          # 统一设计文档
├── JOINT_TRAINING.md                  # Joint Training详细说明
├── EXPERIMENTS.md                      # 实验文档
├── training_guide.md                  # 训练指南
├── TRAINING_FAQ.md                    # 训练FAQ
├── TRAINING_PRINCIPLE.md              # 训练原则
├── TRAINING_MODULES.md                # 训练模块状态
├── TRAINING_ISSUES_FIXED.md           # 训练问题修复
├── DECODE_PHASE_DESIGN.md             # Decode阶段设计
├── BUDGET_ENCODER_TRAINING.md         # Budget encoder训练
├── LATENCY_BUDGET_ANALYSIS.md         # Budget范围分析
├── LATENCY_BUDGET_ENCODING.md         # Budget编码（合并后）
├── REWARD_DESIGN_EXPLANATION.md       # Reward设计说明
├── evaluation_guide.md                # 评估指南
├── IMPLEMENTATION_SUMMARY.md          # 实现总结（合并后）
├── ANALYSIS.md                        # Controller分析（需要检查）
│
├── research/                          # 研究文档
│   ├── SEMANTIC_ROUTER_INTEGRATION.md
│   └── ADALORA_DESIGNS.md
│
└── archive/                           # 归档文档
    ├── DIRECT_LATENCY_MEASUREMENT.md
    ├── STAGE2_FEATURE_EXTRACTION.md
    ├── IMPLEMENTATION_STATUS.md
    ├── IMPROVEMENTS_COMPLETED.md
    ├── TRAINING_IMPROVEMENTS.md
    ├── EXPERIMENT_DESIGN_CHECK.md
    ├── DATASET_LOADING_DESIGN.md
    ├── DECODE_LATENCY_ANALYSIS.md
    ├── DECODE_LATENCY_PREDICTION_CHALLENGE.md
    ├── DECODE_LATENCY_ACCURACY_IMPROVEMENT.md
    ├── POSITIONED_DECODE_LATENCY_TRAINING.md
    └── TEXT_FEATURE_ANALYSIS.md
```

---

**待用户确认后继续整理**

