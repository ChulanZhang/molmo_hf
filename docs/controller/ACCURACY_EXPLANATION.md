# Accuracy 计算说明

## Accuracy 的含义

在controller训练中，**accuracy** 表示模型生成的答案与ground truth答案的匹配程度。

### 计算流程

1. **模型生成**：使用controller选择的配置运行模型，生成文本回答
2. **答案提取**：从生成的文本中提取答案（处理"Answer:"前缀、换行等）
3. **Ground Truth提取**：从metadata中提取标准答案
4. **评分**：使用数据集特定的metric计算匹配分数

### 不同数据集的Accuracy Metric

根据数据集类型，使用不同的评分方法：

#### 1. VQA Score (text_vqa, coco_2014_vqa, okvqa)
- **Metric**: `vqa_score`
- **计算方式**：
  - 对预测答案和标准答案进行预处理（小写、去除标点等）
  - 如果标准答案是列表，返回 `min(count(pred) / 3.0, 1.0)`
  - 如果标准答案是单个字符串，返回 `1.0` 如果完全匹配，否则 `0.0`
- **范围**: `[0.0, 1.0]`

#### 2. Multiple Choice (science_qa_img)
- **Metric**: `mc`
- **计算方式**：
  - 从预测文本中提取选项（A/B/C/D等）
  - 比较预测的选项索引与正确答案索引
- **范围**: `0.0` 或 `1.0`

#### 3. Exact Match (某些数据集)
- **Metric**: `em`
- **计算方式**：完全字符串匹配（忽略大小写）
- **范围**: `0.0` 或 `1.0`

#### 4. ANLS Score (st_qa, doc_qa)
- **Metric**: `ansl_em`
- **计算方式**：Average Normalized Levenshtein Similarity
- **范围**: `[0.0, 1.0]`

## 为什么Accuracy是0？

### 可能的原因

1. **Metadata中没有answers字段**
   - **问题**：`OnlineTrainingDataset` 可能没有正确保存answers
   - **解决**：已修复，现在会将answers合并到metadata中

2. **答案格式不匹配**
   - **问题**：生成的答案格式与标准答案格式不一致
   - **解决**：代码已经处理了"Answer:"前缀和换行符

3. **Metric选择错误**
   - **问题**：使用了错误的metric（例如对MC问题使用vqa_score）
   - **解决**：代码会根据dataset_name自动选择正确的metric

4. **模型生成质量差**
   - **问题**：模型在训练初期生成质量差，无法匹配标准答案
   - **解决**：这是正常的，随着训练进行会改善

### 调试方法

如果accuracy一直是0，可以：

1. **检查metadata结构**：
   ```python
   # 在代码中添加日志
   log.info(f"Metadata keys: {metadata.keys() if isinstance(metadata, dict) else 'not a dict'}")
   log.info(f"Answers: {answers}")
   log.info(f"Predicted: {pred_text}")
   ```

2. **检查答案提取**：
   - 确认生成的文本格式
   - 确认答案提取逻辑是否正确

3. **检查metric选择**：
   - 确认dataset_name是否正确
   - 确认使用的metric是否适合该数据集

## 当前实现的状态

### 已修复的问题

1. ✅ **Answers字段丢失**：已修复 `OnlineTrainingDataset`，现在会将answers合并到metadata中
2. ✅ **Dataset name传递**：已添加dataset_name/style到metadata中，用于自动选择metric
3. ✅ **答案格式处理**：已处理"Answer:"前缀、换行符等

### 预期行为

- **训练初期**：accuracy可能较低（0.0-0.1），因为模型还在学习
- **训练中期**：accuracy应该逐渐提升
- **训练后期**：accuracy应该稳定在合理水平（取决于数据集和模型配置）

### 如果Accuracy仍然是0

如果修复后accuracy仍然是0，可能的原因：

1. **模型生成质量确实很差**（训练初期正常）
2. **答案格式问题**（需要检查实际生成的文本）
3. **Metric计算问题**（需要检查vqa_score等函数的实现）

建议：
- 先检查几个样本的实际生成文本和标准答案
- 确认答案提取逻辑是否正确
- 确认metric计算是否正确

