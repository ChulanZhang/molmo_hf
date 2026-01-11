# 文档命名规范

本文档定义了项目中所有文档文件的命名规范，所有 AI Agent 必须严格遵循。

## 📋 核心规则

**所有文档文件必须使用小写字母和下划线（`snake_case`）**

## ✅ 正确的命名格式

### 基本格式
- 使用小写字母：`a-z`
- 使用下划线分隔单词：`_`
- 文件扩展名：`.md`

### 命名示例

**评估相关文档**：
- ✅ `evaluation_guide.md`
- ✅ `lookup_table_baseline_evaluation.md`
- ✅ `pareto_frontier_evaluation.md`
- ✅ `lmms_eval_integration.md`
- ✅ `adallava_datasets.md`

**控制器相关文档**：
- ✅ `training_guide.md`
- ✅ `latency_estimator_design.md`
- ✅ `joint_training.md`
- ✅ `grpo_explanation.md`
- ✅ `reward_design_explanation.md`

**实验相关文档**：
- ✅ `experiment_design.md`
- ✅ `core_exp_guide.md`
- ✅ `profiling_results.md`

**分析文档**：
- ✅ `latency_measurement_analysis.md`
- ✅ `decode_latency_analysis.md`
- ✅ `tier_fallback_analysis.md`

## ❌ 错误的命名格式

以下命名方式**禁止使用**：

1. **全大写**：
   - ❌ `evaluation_guide.md`
   - ❌ `training_guide.md`
   - ❌ `API_REFERENCE.md`

2. **PascalCase（首字母大写）**：
   - ❌ `EvaluationGuide.md`
   - ❌ `TrainingGuide.md`
   - ❌ `ApiReference.md`

3. **camelCase（驼峰命名）**：
   - ❌ `evaluationGuide.md`
   - ❌ `trainingGuide.md`

4. **kebab-case（连字符）**：
   - ❌ `evaluation-guide.md`
   - ❌ `training-guide.md`
   - ❌ `api-reference.md`

5. **混合大小写**：
   - ❌ `Evaluation_Guide.md`
   - ❌ `evaluation_Guide.md`
   - ❌ `EVALUATION_guide.md`

## 📝 命名原则

### 1. 清晰描述内容
文件名应该清晰描述文档的内容，使用完整的单词而不是缩写。

- ✅ `latency_estimator_design.md` - 清晰描述内容
- ❌ `lat_est_design.md` - 缩写不清晰

### 2. 使用下划线分隔
多个单词之间使用下划线分隔，不要使用空格或其他字符。

- ✅ `lookup_table_baseline.md`
- ❌ `lookup table baseline.md`（包含空格）
- ❌ `lookup-table-baseline.md`（使用连字符）

### 3. 避免特殊字符
文件名中只使用小写字母、数字和下划线。

- ✅ `api_v2_reference.md`
- ❌ `api-v2-reference.md`（使用连字符）
- ❌ `api.v2.reference.md`（使用点号）

### 4. 保持一致性
在同一目录下，使用一致的命名风格。

- ✅ `evaluation_guide.md` 和 `training_guide.md`（都使用下划线）
- ❌ `evaluation_guide.md` 和 `training-guide.md`（混合使用）

## 🔍 命名模式

### 指南类文档
使用 `*_guide.md` 后缀：
- `evaluation_guide.md`
- `training_guide.md`
- `installation_guide.md`

### 设计文档
使用 `*_design.md` 后缀：
- `latency_estimator_design.md`
- `controller_design.md`
- `architecture_design.md`

### 评估文档
使用 `*_evaluation.md` 后缀：
- `lookup_table_baseline_evaluation.md`
- `pareto_frontier_evaluation.md`
- `performance_evaluation.md`

### 实现文档
使用 `*_implementation.md` 后缀：
- `lookup_table_baseline_implementation.md`
- `controller_implementation.md`

### 分析文档
使用 `*_analysis.md` 后缀：
- `latency_measurement_analysis.md`
- `decode_latency_analysis.md`
- `performance_analysis.md`

### 说明文档
使用 `*_explanation.md` 后缀：
- `grpo_explanation.md`
- `reward_design_explanation.md`

## ✅ 检查清单

在创建新文档前，检查：

- [ ] 文件名是否全部使用小写字母？
- [ ] 是否使用下划线（`_`）而不是连字符（`-`）？
- [ ] 文件名是否清晰描述文档内容？
- [ ] 是否遵循了命名模式（如 `*_guide.md`, `*_design.md`）？
- [ ] 是否与同一目录下的其他文件命名风格一致？

## 🚫 常见错误

**错误示例**：
- ❌ `evaluation_guide.md` - 全大写
- ❌ `EvaluationGuide.md` - PascalCase
- ❌ `evaluation-guide.md` - 使用连字符
- ❌ `Evaluation_Guide.md` - 混合大小写

**正确做法**：
- ✅ `evaluation_guide.md` - 小写 + 下划线

## 📚 迁移现有文档

如果发现现有文档使用了错误的命名格式，应该：

1. **重命名文件**：将文件重命名为符合规范的名称
2. **更新引用**：更新所有引用该文件的代码和文档
3. **Git 提交**：提交重命名操作（Git 可以跟踪文件重命名）

## 🔄 自动化检查

在创建新文档时，AI Agent 必须：

1. **检查命名格式**：确保文件名符合规范
2. **验证一致性**：检查同一目录下的其他文件命名风格
3. **更新引用**：如果重命名了文件，更新所有引用

## 📝 示例

### 创建新文档

```python
# ✅ 正确
doc_path = Path("docs/evaluation/new_evaluation_method.md")

# ❌ 错误
doc_path = Path("docs/evaluation/NewEvaluationMethod.md")
doc_path = Path("docs/evaluation/NEW_EVALUATION_METHOD.md")
doc_path = Path("docs/evaluation/new-evaluation-method.md")
```

### 重命名现有文档

```bash
# 重命名文件
mv docs/evaluation/evaluation_guide.md docs/evaluation/evaluation_guide.md

# 更新所有引用（使用 grep 查找）
grep -r "evaluation_guide" docs/ scripts/ experiments/
# 然后更新所有找到的引用
```

## 🎯 总结

- **规则**：所有文档文件必须使用小写字母和下划线（`snake_case`）
- **格式**：`lowercase_with_underscores.md`
- **禁止**：大写字母、PascalCase、camelCase、kebab-case
- **原则**：清晰、一致、描述性

