# 文档命名标准化

本文档记录了文档命名规范的标准化工作。

## 📋 标准化日期

2026-01-XX

## ✅ 完成的工作

### 1. 创建文档命名规范

创建了 `.cursor/rules/document-naming-conventions.md`，详细定义了文档命名规范：

- **核心规则**：所有文档文件必须使用小写字母和下划线（`snake_case`）
- **命名格式**：`lowercase_with_underscores.md`
- **禁止格式**：全大写、PascalCase、camelCase、kebab-case
- **命名模式**：指南类、设计文档、评估文档等的命名模式
- **检查清单**：创建新文档前的检查项

### 2. 更新规则文件

更新了以下规则文件，确保所有 AI Agent 遵循文档命名规范：

#### `.cursor/rules/file-organization.md`
- 添加了文档命名规范的引用
- 更新了文件命名规范章节，强调使用小写

#### `.cursor/rules/project-conventions.md`
- 在文档组织结构部分添加了命名规范说明

#### `.cursor/rules/ai-behavior.md`
- 在文档更新部分添加了命名要求

## 📝 命名规范总结

### 核心规则

**所有文档文件必须使用小写字母和下划线（`snake_case`）**

### 正确示例

- ✅ `evaluation_guide.md`
- ✅ `lookup_table_baseline_evaluation.md`
- ✅ `pareto_frontier_evaluation.md`
- ✅ `training_guide.md`
- ✅ `latency_estimator_design.md`

### 错误示例

- ❌ `evaluation_guide.md` - 全大写
- ❌ `EvaluationGuide.md` - PascalCase
- ❌ `evaluation-guide.md` - 使用连字符
- ❌ `Evaluation_Guide.md` - 混合大小写

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

### 评估文档
使用 `*_evaluation.md` 后缀：
- `lookup_table_baseline_evaluation.md`
- `pareto_frontier_evaluation.md`

### 实现文档
使用 `*_implementation.md` 后缀：
- `lookup_table_baseline_implementation.md`

### 分析文档
使用 `*_analysis.md` 后缀：
- `latency_measurement_analysis.md`
- `decode_latency_analysis.md`

## ✅ 检查清单

在创建新文档前，AI Agent 必须检查：

- [ ] 文件名是否全部使用小写字母？
- [ ] 是否使用下划线（`_`）而不是连字符（`-`）？
- [ ] 文件名是否清晰描述文档内容？
- [ ] 是否遵循了命名模式（如 `*_guide.md`, `*_design.md`）？
- [ ] 是否与同一目录下的其他文件命名风格一致？

## 📚 规则文件位置

所有规则文件都在 `.cursor/rules/` 目录下：

1. **document-naming-conventions.md** - 详细的文档命名规范（新建）
2. **file-organization.md** - 文件组织规范（已更新）
3. **project-conventions.md** - 项目约定（已更新）
4. **ai-behavior.md** - AI Agent 行为规范（已更新）

## 🚀 后续工作

### 现有文档重命名（可选）

如果需要统一现有文档的命名，可以：

1. **识别需要重命名的文件**：
   ```bash
   find docs -name "*.md" -type f | grep -E "[A-Z]"
   ```

2. **重命名文件**：
   ```bash
   # 示例：重命名全大写文件
   mv docs/evaluation/evaluation_guide.md docs/evaluation/evaluation_guide.md
   ```

3. **更新引用**：
   - 使用 `grep` 查找所有引用
   - 更新代码和文档中的引用

### 自动化检查（建议）

可以考虑添加 pre-commit hook 或 CI 检查，确保新文档遵循命名规范。

## 📖 相关文档

- `.cursor/rules/document-naming-conventions.md` - 详细的文档命名规范
- `.cursor/rules/file-organization.md` - 文件组织规范
- `.cursor/rules/project-conventions.md` - 项目约定
- `.cursor/rules/ai-behavior.md` - AI Agent 行为规范

