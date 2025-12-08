# 绘图风格指南 (Plotting Style Guide)

本文档定义了所有实验绘图脚本应遵循的统一风格规范，确保所有图表具有一致的外观和可读性。

## 颜色调色板 (Color Palette)

使用 **ggthemes Classic_10** 调色板，这是一个高饱和度、色盲友好的调色板。

### 标准颜色定义

```python
colors = {
    'primary': '#1F77B4',      # Deep blue (Classic_10 color 1)
    'secondary': '#FF7F0E',    # Bright orange (Classic_10 color 2)
    'tertiary': '#2CA02C',     # Deep green (Classic_10 color 3)
    'quaternary': '#D62728',   # Deep red (Classic_10 color 4)
    'quinary': '#9467BD',      # Deep purple (Classic_10 color 5)
    'senary': '#8C564B',       # Brown (Classic_10 color 6)
    'septenary': '#E377C2',    # Pink (Classic_10 color 7)
    'octonary': '#7F7F7F',     # Gray (Classic_10 color 8)
    'nonary': '#BCBD22',       # Yellow-green (Classic_10 color 9)
    'denary': '#17BECF',       # Cyan (Classic_10 color 10)
}
```

### 特殊用途颜色

```python
# 直方图
'histogram': '#1F77B4',        # Deep blue

# 百分位数标记
'p50': '#1F77B4',              # Deep blue (或 '#D62728' 红色)
'p95': '#FF7F0E',              # Bright orange
'p99': '#D62728',              # Deep red (或 '#9467BD' 紫色)

# Pareto frontier
'pareto': '#D62728',           # Deep red
'pareto_line': '#D62728',      # Deep red
'non_pareto': '#7F7F7F',       # Gray
```

**参考链接**: https://emilhvitfeldt.github.io/r-color-palettes/discrete/ggthemes/Classic_10/

## 字体大小 (Font Sizes)

```python
# 标题
title_fontsize = 18

# 轴标签
label_fontsize = 16

# 图例
legend_fontsize = 14

# 刻度标签
tick_fontsize = 14

# 注释/文本
annotation_fontsize = 10-11
```

## 图形尺寸 (Figure Sizes)

```python
# 标准图形
figsize = (8, 6)      # 宽度 x 高度（英寸）

# 较大图形（如 Pareto frontier）
figsize = (10, 7)     # 或 (12, 8)

# DPI
dpi = 300             # 用于高质量输出
```

## 线条和边框样式 (Line and Border Styles)

```python
# 线条宽度
linewidth = 2.0-2.5   # 主要线条
linewidth = 1.5       # 次要线条
linewidth = 1.0       # 边框

# 边框
edgecolor = 'black'
edgecolors = 'black'  # 对于散点图

# 网格
grid = True
grid_alpha = 0.3
```

## 透明度 (Alpha/Opacity)

```python
# 散点图
scatter_alpha = 0.5-0.9

# 填充区域
fill_alpha = 0.6-0.8

# 网格
grid_alpha = 0.3
```

## 标记样式 (Marker Styles)

```python
# 散点图大小
scatter_size = 100-200  # 根据重要性调整

# Pareto frontier 点
pareto_marker = '*'     # 星形标记
pareto_size = 200

# 普通点
normal_marker = 'o'      # 圆形
normal_size = 100
```

## 布局 (Layout)

```python
# 使用 tight_layout
plt.tight_layout()

# 保存时使用 bbox_inches='tight'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
```

## 文本框样式 (Text Box Styles)

```python
# 注释框
bbox = dict(
    boxstyle='round,pad=0.3',
    facecolor='white',      # 或 'wheat' 用于说明框
    alpha=0.9,
    edgecolor='black',
    linewidth=1.0
)

# 字体
fontsize = 10-11
family = 'monospace'  # 对于代码/配置说明
```

## 示例代码模板

```python
import matplotlib.pyplot as plt
import numpy as np

# 1. 定义颜色调色板
colors = {
    'primary': '#1F77B4',
    'secondary': '#FF7F0E',
    'tertiary': '#2CA02C',
    'quaternary': '#D62728',
}

# 2. 创建图形
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 3. 绘图
ax.scatter(
    x_data, y_data,
    c=colors['primary'],
    s=100,
    alpha=0.7,
    edgecolors='black',
    linewidths=1.0,
    label='Data Points'
)

# 4. 设置标签和标题
ax.set_xlabel('X Label', fontsize=16, fontweight='bold')
ax.set_ylabel('Y Label', fontsize=16, fontweight='bold')
ax.set_title('Plot Title', fontsize=18, fontweight='bold')

# 5. 设置网格和图例
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=14)
ax.tick_params(labelsize=14)

# 6. 布局和保存
plt.tight_layout()
plt.savefig('output.png', dpi=300, bbox_inches='tight')
plt.close()
```

## Pareto Frontier 特殊样式

对于 Pareto frontier 图：

```python
# Pareto 点
ax.scatter(
    pareto_latencies, pareto_accuracies,
    c=colors['pareto'],      # '#D62728' (Deep red)
    s=200,
    alpha=0.9,
    edgecolors='black',
    linewidths=1.5,
    marker='*',
    zorder=5,
    label='Pareto Frontier'
)

# Pareto 线
ax.plot(
    pareto_latencies_sorted, pareto_accuracies_sorted,
    color=colors['pareto_line'],
    linestyle='--',
    linewidth=2.5,
    alpha=0.8,
    zorder=4,
    label='Pareto Frontier Line'
)

# 非 Pareto 点
ax.scatter(
    non_pareto_latencies, non_pareto_accuracies,
    c=colors['non_pareto'],  # '#7F7F7F' (Gray)
    s=100,
    alpha=0.5,
    edgecolors='black',
    linewidths=1.0,
    zorder=1,
    label='Non-Pareto'
)
```

## 注意事项

1. **一致性**: 所有绘图脚本应遵循此风格指南
2. **可访问性**: Classic_10 调色板是色盲友好的
3. **可读性**: 使用足够的字体大小和对比度
4. **专业性**: 使用黑色边框和适当的透明度
5. **高质量**: 始终使用 300 DPI 保存图形

## 参考实现

- `experiments/motivate/plot_exp1.py` - 延迟分布图
- `experiments/motivate/plot_exp2.py` - 组件分析图
- `experiments/motivate/plot_exp3.py` - Vision tokens vs latency
- `experiments/profiling/plots/plot_accuracy_latency_pareto.py` - Pareto frontier 图

