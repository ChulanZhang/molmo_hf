# 文件组织规范

本文档定义了项目中所有文件的组织规范，所有 AI Agent 必须严格遵循这些规则。

**相关文档**：
- `.cursor/rules/document-naming-conventions.md` - 详细的文档命名规范（**必须阅读**）

## 📁 目录结构规范

### 1. `docs/` - 所有文档文件

**用途**：存放所有 Markdown 文档、说明文档、指南等

**包含内容**：
- 所有 `.md` 文件（除了根目录的 `README.md`）
- 评估指南、实验说明、API 文档等
- 按主题组织子目录：
  - `docs/evaluation/` - 评估相关文档
  - `docs/controller/` - 控制器相关文档
  - `docs/experiments/` - 实验相关文档
  - `docs/profiling/` - 性能分析文档
  - `docs/analysis/` - 分析文档
  - `docs/core_exp/` - 核心实验文档
  - `docs/knobs/` - 控制旋钮文档
  - `docs/mechanisms/` - 机制说明文档

**规则**：
- ✅ 所有文档必须放在 `docs/` 或其子目录下
- ❌ 禁止在根目录或其他目录创建 `.md` 文件（除了 `README.md`）
- ✅ 文档应该按主题组织到相应的子目录

### 2. `results/` - 所有实验结果和输出

**用途**：存放所有实验结果、可视化、日志、分析输出等

**包含内容**：
- 实验结果 JSON 文件
- 可视化图表（PNG、PDF 等）
- 日志文件
- 分析输出
- 评估结果

**子目录结构**：
```
results/
├── logs/                    # 训练和实验日志
│   ├── training/           # 训练日志
│   └── experiments/        # 实验日志
├── visualizations/         # 所有可视化图表
│   ├── latency_estimator/  # 延迟估计器可视化
│   ├── pareto_frontier/    # Pareto 前沿图
│   └── ...                 # 其他可视化
├── analysis_output/        # 分析脚本的输出
│   ├── e1_stage_latency_stacks/
│   └── e2_knob_coupling/
├── logs_eval/              # 评估日志和结果
│   ├── pareto_frontier/
│   ├── lookup_table_baseline/
│   └── ...
├── profiling/              # 性能分析结果
│   ├── exp3_visualizations/
│   └── ...
└── core_exp_h100/          # 核心实验结果
    └── ...
```

**规则**：
- ✅ 所有实验结果必须保存在 `results/` 目录下
- ✅ 所有可视化文件必须保存在 `results/visualizations/` 或其子目录
- ✅ 所有日志文件必须保存在 `results/logs/` 或其子目录
- ✅ 所有分析输出必须保存在 `results/analysis_output/` 或其子目录
- ✅ 评估结果必须保存在 `results/logs_eval/` 或其子目录
- ❌ 禁止在根目录或其他位置创建结果文件
- ✅ 使用有意义的子目录名称组织文件

### 3. `checkpoints/` - 模型和控制器权重

**用途**：存放模型检查点、控制器权重等

**包含内容**：
- 模型权重文件（`.pt`, `.pth`, `.safetensors` 等）
- 控制器检查点
- 模型配置文件（如果与权重一起保存）

**子目录结构**：
```
checkpoints/
├── molmo/                  # 主模型权重
├── controller/            # 控制器权重
│   ├── lookup_table_baseline.json
│   └── ...
└── ...
```

**规则**：
- ✅ 所有模型权重必须保存在 `checkpoints/` 目录下
- ✅ 控制器权重保存在 `checkpoints/controller/` 下
- ❌ 禁止在 `results/` 或其他目录保存权重文件

### 4. `experiments/` - 核心实验代码

**用途**：存放所有实验脚本和实验相关代码

**包含内容**：
- 实验脚本（`.py` 文件）
- 实验配置文件
- 实验相关的工具函数

**子目录结构**：
```
experiments/
├── controller/            # 控制器相关实验
├── core_exp/              # 核心实验
├── profiling/             # 性能分析实验
├── motivate/              # 动机研究实验
└── ...
```

**规则**：
- ✅ 所有实验代码必须放在 `experiments/` 目录下
- ✅ 按功能组织到相应的子目录
- ❌ 禁止在根目录或其他位置创建实验脚本

### 5. `molmo/` - 核心模型代码

**用途**：存放核心模型实现、数据处理、工具函数等

**包含内容**：
- 模型定义（`models/`）
- 数据预处理（`preprocessors/`）
- 工具函数（`utils/`）
- 数据加载（`data/`）
- 评估函数（`eval/`）

**规则**：
- ✅ 所有核心模型代码必须放在 `molmo/` 目录下
- ✅ 按功能模块组织到相应的子目录
- ❌ 禁止在 `experiments/` 或其他位置放置核心模型代码

### 6. `scripts/` - 后处理和分析脚本

**用途**：存放后处理、分析、工具脚本

**包含内容**：
- 数据分析脚本
- 后处理脚本
- 工具脚本（如数据下载、格式转换等）
- Shell 脚本（`.sh` 文件）

**规则**：
- ✅ 所有后处理和分析脚本放在 `scripts/` 目录下
- ✅ Shell 脚本（`.sh`）也放在 `scripts/` 目录下
- ❌ 禁止在根目录或其他位置创建脚本文件

### 7. `tests/` - 功能测试代码

**用途**：存放所有测试代码

**包含内容**：
- 单元测试
- 集成测试
- 测试工具和辅助函数

**规则**：
- ✅ 所有测试代码必须放在 `tests/` 目录下
- ✅ 测试文件应该与源码结构对应

### 8. `configs/` - 配置文件

**用途**：存放配置文件

**包含内容**：
- 模型配置（`configs/model/`）
- 分词器配置（`configs/tokenizer/`）
- 实验配置（`configs/experiments/`）

**规则**：
- ✅ 所有配置文件放在 `configs/` 目录下
- ✅ 按类型组织到相应的子目录

### 9. 根目录文件

**允许的文件**：
- `README.md` - 项目主 README
- `setup.py` - Python 包安装配置
- `requirements.txt` 或 `pyproject.toml` - 依赖管理
- `.gitignore` - Git 忽略文件
- `.cursor/` - Cursor 配置目录
- `activate_env.sh` - 环境激活脚本（可以保留，因为这是项目入口脚本）

**禁止的文件**：
- ❌ 其他 `.md` 文件（应放在 `docs/`）
- ❌ 其他 `.sh` 脚本（应放在 `scripts/`）
- ❌ 实验结果文件（应放在 `results/`）
- ❌ 可视化文件（应放在 `results/visualizations/`）
- ❌ 日志文件（应放在 `results/logs/`）

## 📋 文件命名规范

### 文档文件（重要：统一使用小写）

**规则**：所有文档文件必须使用**小写字母**，使用下划线（`snake_case`）分隔单词。

- ✅ **正确**：`evaluation_guide.md`, `lookup_table_baseline.md`, `training_guide.md`
- ❌ **错误**：`evaluation_guide.md`, `EvaluationGuide.md`, `evaluation-guide.md`

**详细规范**：请参考 `.cursor/rules/document-naming-conventions.md` 获取完整的文档命名规范、命名模式、检查清单等。

**命名格式**：
- 使用 `snake_case`（小写字母 + 下划线）
- 文件名应该清晰描述文档内容
- 避免使用缩写，除非是广泛使用的（如 `api`, `ui`, `cli`）

**示例**：
- `evaluation_guide.md` - 评估指南
- `lookup_table_baseline_evaluation.md` - 查找表基线评估
- `pareto_frontier_evaluation.md` - Pareto 前沿评估
- `training_guide.md` - 训练指南
- `latency_estimator_design.md` - 延迟估计器设计

### 结果文件
- 使用描述性名称，包含实验名称、数据集、时间戳等
- 示例：`exp3_results_20240115.json`, `text_vqa_validation_accuracy.png`

### 日志文件
- 包含时间戳和实验名称
- 示例：`training_20240115_143022.log`, `experiment_core_exp_20240115.log`

## 🔄 迁移规则

当创建新文件时，AI Agent 必须：

1. **检查文件类型**：确定文件应该放在哪个目录
2. **遵循目录规范**：将文件放在正确的目录下
3. **更新引用**：如果移动了文件，更新所有引用该文件的代码
4. **创建子目录**：如果需要的子目录不存在，先创建子目录

## 📝 文档和代码同步更新

**重要**：在开发新功能时，必须同时更新文档和代码。

### 开发新功能的工作流

1. **开发前：查阅现有文档**
   - 搜索 `docs/` 目录下的相关文档
   - 理解现有的设计模式和约定
   - 参考现有文档的结构

2. **开发中：同步更新文档**
   - 在编写代码的同时更新或创建文档
   - 文档应该反映代码的当前状态

3. **开发后：验证一致性**
   - 确保文档中的路径遵循文件组织规范
   - 验证文档中的示例代码可以运行
   - 检查文档是否完整

### 路径检查清单

在保存文件前，检查路径是否符合规范：

- [ ] 可视化文件是否保存到 `results/visualizations/`？
- [ ] 日志文件是否保存到 `results/logs/`？
- [ ] 分析输出是否保存到 `results/analysis_output/`？
- [ ] 评估结果是否保存到 `results/logs_eval/`？
- [ ] 检查点是否保存到 `checkpoints/`？
- [ ] 文档是否保存到 `docs/`？

## ✅ 检查清单

在创建或移动文件前，检查：

- [ ] 文档文件是否放在 `docs/` 或其子目录？
- [ ] 实验结果是否放在 `results/` 或其子目录？
- [ ] 可视化文件是否放在 `results/visualizations/` 或其子目录？
- [ ] 日志文件是否放在 `results/logs/` 或其子目录？
- [ ] 脚本文件是否放在 `scripts/` 目录？
- [ ] 实验代码是否放在 `experiments/` 目录？
- [ ] 核心模型代码是否放在 `molmo/` 目录？
- [ ] 测试代码是否放在 `tests/` 目录？
- [ ] 配置文件是否放在 `configs/` 目录？
- [ ] 权重文件是否放在 `checkpoints/` 目录？

## 🚫 常见错误

**错误示例**：
- ❌ 在根目录创建 `experiment_results.json`
- ❌ 在根目录创建 `visualization.png`
- ❌ 在根目录创建 `analysis.md`
- ❌ 在 `experiments/` 目录创建核心模型代码
- ❌ 在 `results/` 目录保存模型权重

**正确做法**：
- ✅ `results/experiment_results.json`
- ✅ `results/visualizations/visualization.png`
- ✅ `docs/analysis.md`
- ✅ 核心模型代码放在 `molmo/`
- ✅ 模型权重放在 `checkpoints/`

## 📝 示例

### 创建新文档
```python
# ✅ 正确
doc_path = Path("docs/evaluation/new_guide.md")

# ❌ 错误
doc_path = Path("new_guide.md")
```

### 保存实验结果
```python
# ✅ 正确
output_dir = Path("results/core_exp_h100")
output_file = output_dir / "experiment_results.json"

# ❌ 错误
output_file = Path("experiment_results.json")
```

### 保存可视化
```python
# ✅ 正确
viz_dir = Path("results/visualizations/pareto_frontier")
plt.savefig(viz_dir / "pareto_plot.png")

# ❌ 错误
plt.savefig("pareto_plot.png")
```

### 保存日志
```python
# ✅ 正确
log_dir = Path("results/logs/training")
log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# ❌ 错误
log_file = Path("training.log")
```

