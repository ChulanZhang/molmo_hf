# EXP3 实验结果可视化指南

## 生成的图表和表格

所有可视化文件保存在：`results/profiling/exp3_visualizations/`

### 1. 图表文件（PNG 格式，300 DPI）

#### 1.1 相关性条形图
**文件**：`correlation_bar_chart.png`

**内容**：
- 横向条形图显示每个数据集的 Spearman 相关系数
- 绿色表示一致性数据集（相关系数 ≥ 0.9）
- 红色表示不一致数据集（相关系数 < 0.9）
- 橙色虚线标记一致性阈值（0.9）

**适用场景**：
- 论文中的结果展示
- 演示文稿
- 快速比较不同数据集的一致性

#### 1.2 散点图对比
**文件**：`scatter_comparison.png`

**内容**：
- 6 个数据集的散点图，每个显示训练集 vs 验证集的重要性分数
- 对角线表示完全一致
- 每个点标注对应的 block 编号
- 显示 Spearman 相关系数

**适用场景**：
- 详细分析训练集和验证集的一致性
- 识别异常值或特定 block 的差异
- 论文中的详细分析部分

#### 1.3 重要性热力图（训练集）
**文件**：`importance_heatmap_train.png`

**内容**：
- 显示前 6 个数据集的所有 16 个 block 在训练集上的重要性分数
- 颜色越深表示重要性越高
- 每个单元格显示具体数值

**适用场景**：
- 识别跨数据集的通用模式
- 发现哪些 block 在不同任务中都重要/不重要
- 论文中的模式分析

#### 1.4 重要性热力图（验证集）
**文件**：`importance_heatmap_validation.png`

**内容**：
- 显示前 6 个数据集的所有 16 个 block 在验证集上的重要性分数
- 颜色越深表示重要性越高
- 每个单元格显示具体数值
- 与训练集热力图使用相同的颜色刻度，便于比较

**适用场景**：
- 比较训练集和验证集的重要性分数差异
- 验证重要性分数的一致性
- 识别数据集特定的模式

### 2. LaTeX 表格文件

#### 2.1 一致性分析表格
**文件**：`consistency_table.tex`

**内容**：
```latex
\begin{table}[htbp]
\centering
\caption{Train vs Validation Importance Score Consistency Analysis}
\label{tab:exp3_consistency}
\begin{tabular}{lccc}
\toprule
\textbf{Dataset} & \textbf{Spearman $\rho$} & \textbf{P-value} & \textbf{Consistent} \\
\midrule
st\_qa & 0.9882 & 8.11e-13 & \checkmark \\
coco\_caption & 0.9882 & 8.11e-13 & \checkmark \\
okvqa & 0.9750 & 1.55e-10 & \checkmark \\
...
\bottomrule
\end{tabular}
\end{table}
```

**使用方法**：
1. 在 LaTeX 文档中 `\input{consistency_table.tex}`
2. 需要 `booktabs` 包：`\usepackage{booktabs}`
3. 需要 `amssymb` 包用于 `\checkmark`：`\usepackage{amssymb}`

#### 2.2 Block 重要性排名表格
**文件**：`block_importance_table.tex`

**内容**：
- 显示每个数据集的前 5 个最不重要的 block
- 包含 block 编号和重要性分数
- 使用 `table*` 环境（跨双栏）

**使用方法**：
1. 在 LaTeX 文档中 `\input{block_importance_table.tex}`
2. 需要 `booktabs` 包
3. 需要 `graphicx` 包用于 `\resizebox`

## 图表格式建议

### 论文使用
- **主图**：相关性条形图（简洁明了）
- **详细分析**：散点图对比（展示细节）
- **模式分析**：重要性热力图（跨数据集比较）

### 演示文稿
- 使用 PNG 格式（已生成，300 DPI）
- 可以直接插入 PowerPoint 或 Keynote
- 建议使用相关性条形图作为主图

### 报告文档
- Markdown 文档中可以直接引用 PNG 图片
- LaTeX 文档中使用生成的 `.tex` 文件

## 重新生成图表

如果需要修改图表样式或添加新数据集，运行：

```bash
python experiments/profiling/knob3_layers/visualize_exp3_results.py
```

## 自定义选项

在 `visualize_exp3_results.py` 中可以调整：
- `top_n`：显示的数据集数量
- 图表尺寸和样式
- 颜色方案
- 字体大小

## LaTeX 依赖包

使用生成的表格时，确保包含以下包：

```latex
\usepackage{booktabs}      % 用于 \toprule, \midrule, \bottomrule
\usepackage{amssymb}       % 用于 \checkmark
\usepackage{graphicx}      % 用于 \resizebox（仅 block_importance_table.tex）
```

