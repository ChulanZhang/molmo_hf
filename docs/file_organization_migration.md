# 文件组织迁移总结

本文档记录了项目文件组织的迁移工作，所有文件已按照新的组织规范重新整理。

## 📋 迁移日期

2026-01-XX

## ✅ 已完成的工作

### 1. 创建文件组织规则

创建了 `.cursor/rules/file-organization.md`，定义了严格的文件组织规范：
- `docs/` - 所有文档文件
- `results/` - 所有实验结果和输出
- `checkpoints/` - 模型和控制器权重
- `experiments/` - 核心实验代码
- `molmo/` - 核心模型代码
- `scripts/` - 后处理和分析脚本
- `tests/` - 功能测试代码
- `configs/` - 配置文件

### 2. 文件移动

以下文件和目录已移动到正确位置：

#### 文档文件
- `code_changes_analysis.md` → `docs/code_changes_analysis.md`

#### 脚本文件
- `prepare_bugfix_commit.sh` → `scripts/prepare_bugfix_commit.sh`
- `prepare_exp3_final_commit.sh` → `scripts/prepare_exp3_final_commit.sh`
- `prepare_profiling_commit.sh` → `scripts/prepare_profiling_commit.sh`

#### 结果和输出文件
- `visualizations/` → `results/visualizations/`
- `analysis_output/` → `results/analysis_output/`
- `logs_eval/` → `results/logs_eval/`
- `test_chart.png` → `results/visualizations/test_chart.png`
- `evaluation_results.json` → `results/evaluation_results.json`
- `joint_controller_training.log` → `results/logs/training/joint_controller_training.log`

### 3. 代码更新

更新了以下文件中的路径引用：

#### Python 代码文件
- `experiments/controller/train_joint_controller.py` - 更新日志文件路径
- `experiments/controller/visualize_latency_estimator.py` - 更新可视化输出路径
- `experiments/controller/analyze_output_tokens_distribution.py` - 更新分析输出路径
- `experiments/controller/evaluate_pareto_frontier.py` - 更新评估输出路径
- `experiments/controller/evaluate_lookup_table_baseline.py` - 更新评估输出路径
- `experiments/controller/plot_pareto_frontier.py` - 更新数据文件路径
- `experiments/controller/evaluate_lookup_table_baseline_batch.py` - 更新输出路径
- `experiments/controller/run_lmms_eval_lookup_table.py` - 更新输出路径
- `experiments/controller/evaluate_adaptive_inference.py` - 更新输出路径
- `experiments/controller/run_lmms_eval.py` - 更新输出路径
- `experiments/controller/test_adaptive_inference.py` - 更新输出路径
- `scripts/plot_e1_stage_latency_stacks.py` - 更新输出路径
- `experiments/profiling/plots/plot_knob_coupling_proof.py` - 更新输出路径

#### 文档文件
- `docs/controller/training_guide.md` - 更新日志文件路径
- `docs/controller/latency_estimator_commands.md` - 更新可视化路径
- `docs/controller/evaluation_guide.md` - 更新可视化路径
- `docs/evaluation/pareto_frontier_evaluation.md` - 更新所有路径引用
- `docs/evaluation/lookup_table_baseline_evaluation.md` - 更新所有路径引用
- `docs/evaluation/evaluation_guide.md` - 更新所有路径引用
- `docs/evaluation/lmms_eval_integration.md` - 更新所有路径引用
- `results/analysis_output/e2_knob_coupling/README.md` - 更新输出路径

## 📁 新的目录结构

```
molmo_hf/
├── docs/                          # 所有文档
│   ├── evaluation/               # 评估文档
│   ├── controller/               # 控制器文档
│   ├── experiments/              # 实验文档
│   └── ...
├── results/                       # 所有结果和输出
│   ├── logs/                     # 日志文件
│   │   └── training/            # 训练日志
│   ├── visualizations/           # 可视化图表
│   │   └── latency_estimator/  # 延迟估计器可视化
│   ├── analysis_output/          # 分析输出
│   │   ├── e1_stage_latency_stacks/
│   │   └── e2_knob_coupling/
│   ├── logs_eval/                # 评估日志和结果
│   │   ├── pareto_frontier/
│   │   └── lookup_table_baseline/
│   └── ...
├── checkpoints/                   # 模型和控制器权重
│   ├── molmo/                    # 主模型权重
│   └── controller/               # 控制器权重
├── experiments/                   # 实验代码
│   ├── controller/               # 控制器实验
│   ├── core_exp/                 # 核心实验
│   └── profiling/                # 性能分析
├── molmo/                         # 核心模型代码
├── scripts/                       # 后处理和分析脚本
├── tests/                         # 测试代码
└── configs/                       # 配置文件
```

## 🔄 路径映射表

| 旧路径 | 新路径 |
|--------|--------|
| `./logs_eval/` | `./results/logs_eval/` |
| `visualizations/` | `results/visualizations/` |
| `analysis_output/` | `results/analysis_output/` |
| `joint_controller_training.log` | `results/logs/training/joint_controller_training.log` |
| `evaluation_results.json` | `results/evaluation_results.json` |
| `test_chart.png` | `results/visualizations/test_chart.png` |
| `code_changes_analysis.md` | `docs/code_changes_analysis.md` |
| `prepare_*.sh` | `scripts/prepare_*.sh` |

## 📝 注意事项

1. **向后兼容性**：如果某些脚本或文档仍使用旧路径，需要手动更新
2. **Git 跟踪**：文件移动后，Git 可能需要重新跟踪这些文件
3. **符号链接**：如果有符号链接指向旧路径，需要更新
4. **环境变量**：检查是否有环境变量指向旧路径

## ✅ 验证清单

- [x] 所有文档文件已移动到 `docs/`
- [x] 所有脚本文件已移动到 `scripts/`
- [x] 所有可视化文件已移动到 `results/visualizations/`
- [x] 所有分析输出已移动到 `results/analysis_output/`
- [x] 所有评估日志已移动到 `results/logs_eval/`
- [x] 所有日志文件已移动到 `results/logs/`
- [x] 所有代码中的路径引用已更新
- [x] 所有文档中的路径引用已更新
- [x] 文件组织规则已创建并加入 Git 跟踪

## 🚀 后续工作

1. **测试**：运行关键脚本，确保路径更新正确
2. **文档**：更新 README 和其他文档，反映新的目录结构
3. **Git 提交**：提交所有更改，包括文件移动和路径更新
4. **团队通知**：通知团队成员新的文件组织规范

## 📚 相关文档

- `.cursor/rules/file-organization.md` - 文件组织规范
- `.cursor/rules/project-conventions.md` - 项目约定
- `.cursor/rules/experiment-design-patterns.md` - 实验设计模式

