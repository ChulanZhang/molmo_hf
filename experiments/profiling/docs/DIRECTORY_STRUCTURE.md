# 目录结构说明 (Directory Structure)

本文档说明 `experiments/profiling/` 目录的组织结构。

## 目录结构

```
experiments/profiling/
├── docs/                          # 文档目录
│   ├── PLOTTING_STYLE_GUIDE.md   # 绘图风格指南
│   ├── MULTI_DATASET_EXPERIMENTS.md  # 多数据集实验说明
│   └── DATASET_SIZE_ESTIMATION.md    # 数据集大小估算
│
├── plots/                         # 绘图脚本目录
│   ├── plot_accuracy_latency_pareto.py    # Pareto frontier 图
│   ├── plot_accuracy_latency_pareto.sh
│   ├── plot_exp1_context_scaling.py
│   ├── plot_exp1_exp2_exp3_accuracy.py
│   ├── plot_exp2_moe_topk.py
│   ├── plot_exp3_transformer_blocks_mask.py
│   ├── plot_exp4_output_tokens.py
│   └── plot_context_scaling.py
│
├── knob1_tokens/                  # 实验1：Vision tokens 控制
│   ├── exp1_accuracy.py
│   └── exp_context_scaling.py
│
├── knob2_topk/                    # 实验2：MoE Top-K 控制
│   ├── exp2_accuracy.py
│   └── exp_moe_topk.py
│
├── knob3_layers/                  # 实验3：Transformer layers 控制
│   ├── exp3_accuracy.py
│   ├── exp3_accuracy_sensitivity.py
│   └── exp_transformer_blocks_mask.py
│
├── knob4_output_tokens/           # 实验4：Output tokens 控制
│   └── exp_output_tokens_scaling.py
│
├── knob5_combined/                # 实验5/6：组合控制
│   ├── exp5_accuracy.py            # 实验5：准确率测量
│   ├── exp6_accuracy.py           # 实验6：延迟测量
│   └── SPARSE_SAMPLING_STRATEGIES.md
│
├── utils/                         # 工具脚本
│   ├── compare_pareto_frontiers.py
│   ├── analyze_*.py
│   └── ...
│
└── *.sh                           # Bash 脚本（运行实验和绘图）
    ├── run_exp*.sh                # 运行实验脚本
    ├── plot_*.sh                  # 绘图脚本
    └── test_*.sh                  # 测试脚本
```

## 主要目录说明

### `docs/`
存放所有文档文件，包括：
- **PLOTTING_STYLE_GUIDE.md**: 统一的绘图风格指南，定义颜色、字体、尺寸等规范
- **MULTI_DATASET_EXPERIMENTS.md**: 多数据集实验的使用说明
- **DATASET_SIZE_ESTIMATION.md**: 数据集大小和实验时间估算

### `plots/`
存放所有绘图相关的 Python 脚本和 Shell 脚本：
- 所有 `plot_*.py` 文件
- 绘图相关的 Shell 脚本

### `knob*/`
各个实验的代码目录，按控制维度组织：
- `knob1_tokens/`: Vision tokens 相关实验
- `knob2_topk/`: MoE Top-K 相关实验
- `knob3_layers/`: Transformer layers 相关实验
- `knob4_output_tokens/`: Output tokens 相关实验
- `knob5_combined/`: 组合控制实验（exp5 和 exp6）

### `utils/`
工具脚本目录，包含各种辅助分析脚本。

### 根目录的 Shell 脚本
- `run_*.sh`: 运行实验的脚本
- `plot_*.sh`: 调用绘图脚本的 Shell 脚本
- `test_*.sh`: 测试脚本

## 使用说明

### 运行实验
```bash
# 运行单个实验
./run_exp5_accuracy.sh

# 运行多数据集实验
./run_exp5_exp6_multi_datasets.sh exp5
```

### 生成图表
```bash
# 绘制 Pareto frontier
./plots/plot_accuracy_latency_pareto.sh

# 绘制所有实验图表
./plot_all_experiments.sh
```

### 查看文档
```bash
# 查看绘图风格指南
cat docs/PLOTTING_STYLE_GUIDE.md

# 查看多数据集实验说明
cat docs/MULTI_DATASET_EXPERIMENTS.md
```

## 文件命名规范

- **实验脚本**: `exp{数字}_{类型}.py` (如 `exp5_accuracy.py`)
- **绘图脚本**: `plot_{实验名}.py` (如 `plot_exp1_context_scaling.py`)
- **运行脚本**: `run_{实验名}.sh` (如 `run_exp5_accuracy.sh`)
- **文档**: `{主题}.md` (如 `PLOTTING_STYLE_GUIDE.md`)

## 更新历史

- **2025-12-08**: 重组目录结构，将文档移到 `docs/`，绘图脚本移到 `plots/`

