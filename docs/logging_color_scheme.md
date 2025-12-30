# 日志颜色方案

## 概述

为了便于在终端中阅读日志，项目现在使用统一的颜色方案来突出显示关键信息。

## 颜色方案

### 日志级别颜色（通过 colorlog 或 RichHandler）

- **DEBUG**: 青色 (cyan) - 用于调试信息，如 Timer 的启动/完成消息
- **INFO**: 绿色 (green) - 常规信息
- **WARNING**: 黄色 (yellow) - 警告信息
- **ERROR**: 红色 (red) - 错误信息
- **CRITICAL**: 红色背景白色文字 (red,bg_white) - 严重错误

### 核心信息颜色（使用 ANSI 颜色代码）

#### 1. Benchmark/Dataset 名称
- **颜色**: 亮青色边框 + 亮洋红色名称
- **示例**: `Running Combined Profiling on coco_2014_vqa (validation)`
- **用途**: 突出显示当前正在处理的 benchmark

#### 2. 配置信息（重要）
- **颜色**: 亮黄色键名 + 亮白色值
- **示例**: `Dataset: coco_2014_vqa/validation`
- **用途**: 关键配置参数（dataset, tier, 等）

#### 3. 配置信息（常规）
- **颜色**: 青色键名 + 普通值
- **示例**: `Number of samples: 36`
- **用途**: 常规配置参数

#### 4. 配置节标题
- **颜色**: 亮蓝色
- **示例**: `Experiment Configuration`
- **用途**: 配置部分的标题

#### 5. 配置项（每个配置循环）
- **颜色**: 亮绿色边框 + 亮黄色值
- **示例**: `Configuration 1/2: tier=high (crops: 9-12), top_k=8, num_active_blocks=16`
- **用途**: 突出显示当前正在处理的配置

#### 6. 完成信息
- **颜色**: 亮绿色（成功信息）
- **示例**: `Experiment completed! Results saved:`
- **用途**: 实验完成和结果保存信息

#### 7. 文件列表
- **颜色**: 青色
- **示例**: `  - coco-2014-vqa_imgsizetier-high_crops6_topk8_blocks16.json`
- **用途**: 保存的结果文件列表

## 修改内容

### 1. Timer 类统一使用 logging.debug

**之前**:
```python
print(f"{Color.CYAN}[DEBUG] Starting: {self.name}...{Color.ENDC}")
```

**现在**:
```python
self.logger.debug(f"Starting: {self.name}...")
```

这样 DEBUG 消息会通过统一的 logging 系统输出，并显示为青色（如果使用 colorlog）。

### 2. 核心信息添加颜色

在 `acc_lat_profiling.py` 中添加了辅助函数：
- `log_benchmark_name()`: 突出显示 benchmark 名称
- `log_config_info()`: 带颜色的配置信息
- `log_config_section()`: 配置节标题

### 3. 启用 DEBUG 级别日志

将 logging 级别设置为 `DEBUG`，这样 Timer 的调试信息可以显示。

## 使用示例

### Benchmark 名称
```
============================================================
Running Combined Profiling on coco_2014_vqa (validation)
============================================================
```
（亮青色边框，亮洋红色名称）

### 配置信息
```
Experiment Configuration
Dataset: coco_2014_vqa/validation
Tier-based vision token control: ['low', 'high']
Number of samples: 36
```
（重要配置用亮黄色，常规配置用青色）

### 配置项
```
============================================================
Configuration 1/2: tier=high (crops: 9-12), top_k=8, num_active_blocks=16
============================================================
```
（亮绿色边框，亮黄色值）

### 完成信息
```
============================================================
Experiment completed! Results saved:
  - coco-2014-vqa_imgsizetier-high_crops6_topk8_blocks16.json
============================================================
```
（亮绿色成功信息，青色文件列表）

## 兼容性

- **有 colorlog**: 日志级别颜色 + 核心信息颜色（ANSI 代码）
- **无 colorlog**: 使用 RichHandler 的日志级别颜色 + 核心信息颜色（ANSI 代码）
- **ANSI 颜色代码**: 在所有支持 ANSI 的终端中都能工作，不依赖 colorlog

## 效果

使用颜色后，终端日志会更容易阅读：
- 一眼就能看到当前处理的 benchmark（亮洋红色）
- 关键配置参数突出显示（亮黄色）
- 配置项清晰可见（亮绿色边框）
- DEBUG 信息统一格式（青色）
- 成功信息明显（亮绿色）

