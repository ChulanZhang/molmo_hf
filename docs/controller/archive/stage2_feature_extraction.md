# Stage2特征提取设计

## 问题

**用户提问**：Stage2有必要提取所有token的特征吗？还是只选最后一个token就够了？因为最后一个应该是latency token，其跟vision tokens和language tokens经过第一层，应该是做了attention，信息量够吗？

## 回答

**只使用最后一个token（latency token）就足够了！**

### 原因分析

1. **序列结构**：
   - 在Molmo模型中，序列通常是：`[CLS] token, vision tokens, language tokens, latency token`
   - 经过第一个transformer block后，latency token已经和vision tokens、language tokens做了attention

2. **信息量充足**：
   - Latency token经过第一个block的attention机制，已经聚合了vision和language的信息
   - 使用最后一个token可以：
     - 减少计算量（不需要mean pooling）
     - 保持特征维度一致（d_model）
     - 利用attention后的丰富信息

3. **实现优势**：
   - 更简单：直接取`x[:, -1, :]`即可
   - 更高效：不需要额外的pooling操作
   - 更合理：latency token本身就是为控制设计的特殊token

### 实现

```python
# Step 4: Extract features from first block output for Stage2
# Use the last token (latency token) which has attended to vision and language tokens
# The last token contains rich information after attention with vision and language tokens
stage2_input_feat = x[:, -1, :]  # (B, d_model) - last token (latency token)
```

### 对比

**之前（mean pooling）**：
```python
stage2_input_feat = x.mean(dim=1)  # (B, d_model)
```
- 需要计算所有token的平均值
- 可能丢失重要信息（平均化会稀释）

**现在（last token）**：
```python
stage2_input_feat = x[:, -1, :]  # (B, d_model)
```
- 直接使用latency token
- 保留了attention后的完整信息
- 计算更简单

## 总结

✅ **使用最后一个token（latency token）是更好的选择**
- 信息量充足（经过attention）
- 计算更简单
- 更符合设计意图

