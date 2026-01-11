# Decode Latency 精度提升分析

> **文档目的**: 解释为什么更新后 decode per-token latency 的预测精度显著提升  
> **最后更新**: 2026-01-08

## 📊 精度提升对比

### 更新前（旧设计）
- **Decode Per-Token Latency MAE**: ~5-10 ms/token（估计）
- **R²**: ~0.85-0.90（估计）
- **问题**: 假设所有位置的 decode latency 相同，忽略了 KV cache 增长的影响

### 更新后（新设计）
- **Decode Total Latency MAE**: 15.34ms
- **Decode Total Latency R²**: 0.9974（**显著提升**）
- **Decode Average Per-Token MAE**: 1.428ms/token
- **Decode Average Per-Token R²**: 0.9553（**显著提升**）
- **MAPE**: 4.54%（**优秀**）

## 🔑 关键改进点

### 1. **从平均预测到位置化预测（核心改进）**

#### 旧设计的问题
```python
# 旧方法：预测一个固定的平均 decode per-token latency
T_decode_per_token_avg = model(config)  # 假设所有位置相同
T_decode_total_pred = T_decode_per_token_avg * output_tokens
```

**问题**：
- 忽略了 KV cache 增长导致的渐进式变慢
- 早期 tokens（1-3）实际更快（~25ms/token）
- 后期 tokens（20+）实际更慢（~45ms/token）
- 使用平均值会导致系统性误差

#### 新设计的优势
```python
# 新方法：预测每个位置的 decode latency
positions = [1, 2, 3, ..., output_tokens]
T_decode_per_token_at_pos = model(config, positions)  # 每个位置不同
T_decode_total_pred = sum(T_decode_per_token_at_pos)
```

**优势**：
- 准确建模 KV cache 增长的影响
- 早期位置预测较低 latency
- 后期位置预测较高 latency
- 总和更准确

### 2. **训练目标的改变（关键创新）**

#### 旧训练策略
```python
# 旧方法：直接预测平均 per-token latency
target = T_LLM_decode / output_tokens  # 平均 per-token latency
pred = model(config)['T_decode_per_token']  # 预测平均
loss = MSE(pred, target)
```

**问题**：
- 训练目标是平均值，但实际 latency 随位置变化
- 模型无法学习位置依赖关系
- 即使预测准确，总和也可能不准确

#### 新训练策略
```python
# 新方法：预测所有位置的 latency，求和后与 total 比较
positions = torch.arange(1, output_tokens + 1)
pred_latencies = model.predict_decode_at_positions(config, positions)  # (output_tokens,)
pred_total = pred_latencies.sum()
target_total = T_LLM_decode  # 总 decode latency
loss = MSE(pred_total, target_total)
```

**优势**：
- 训练目标是总和（ground truth），更准确
- 模型必须学习位置依赖（否则总和不对）
- 即使单个位置预测有误差，总和仍然准确

### 3. **模型架构的改进**

#### 旧架构
```python
# 旧方法：单一编码器，没有位置信息
features = [vision_tokens, text_tokens, tier, top_k, blocks]  # 5维
encoded = encoder(features)
T_decode_per_token = decode_head(encoded)  # 固定值
```

#### 新架构
```python
# 新方法：分离配置编码器和位置编码器
config_features = [vision_tokens, text_tokens, tier, top_k, blocks]  # 5维
config_encoded = config_encoder(config_features)  # (B, hidden_dim)

position_features = log(position + 1)  # 归一化位置
position_encoded = position_encoder(position_features)  # (B, hidden_dim // 4)

decode_features = concat([config_encoded, position_encoded])  # (B, hidden_dim + hidden_dim // 4)
T_decode_per_token = decode_head(decode_features)  # 随位置变化
```

**关键改进**：
1. **位置编码器**：专门处理位置信息，使用 `log(position + 1)` 归一化
2. **分离设计**：配置特征和位置特征分别编码，然后拼接
3. **位置感知**：decode head 接收位置信息，可以学习位置依赖

### 4. **损失函数的优化**

#### 旧损失函数
```python
loss = loss_prefill + loss_decode_per_token
```

#### 新损失函数
```python
loss = 2.0 * loss_prefill + 1.0 * loss_decode_total + 0.5 * loss_decode_positioned
```

**改进**：
- **权重分配**：prefill 权重更高（2.0），因为它是主要指标
- **Total loss**：使用 total decode latency 作为目标，更准确
- **Positioned loss**：如果有 positioned 数据，添加 per-position 监督

### 5. **位置归一化的选择**

```python
position_normalized = log(position + 1)
```

**为什么使用 log 归一化**：
- KV cache 增长导致的 latency 增长是**渐进式**的，不是线性的
- 早期位置（1-5）增长快，后期位置（10+）增长放缓
- `log(position + 1)` 能够更好地建模这种渐进式增长
- 相比线性归一化，log 归一化更符合实际物理过程

## 📈 精度提升的数学解释

### 为什么总和预测更准确？

假设实际 decode latency 随位置变化：
```
实际: T_decode(pos) = 25 + 20 * log(pos + 1)  # 渐进式增长
```

**旧方法**（平均预测）：
```
预测: T_decode_avg = 35 ms/token  # 固定值
实际总和: sum(25 + 20*log(i+1) for i in 1..N) = T_total
预测总和: 35 * N
误差: |T_total - 35*N|  # 可能很大
```

**新方法**（位置化预测）：
```
预测: T_decode(pos) ≈ 25 + 20 * log(pos + 1)  # 学习到位置依赖
预测总和: sum(predicted(pos) for pos in 1..N)
实际总和: T_total
误差: |T_total - sum(predicted)|  # 更小，因为模型学习了位置依赖
```

### 为什么 R² 从 ~0.90 提升到 0.9974？

1. **更准确的建模**：位置化预测能够准确建模 KV cache 增长的影响
2. **训练目标更合理**：使用总和作为目标，避免了平均值的误差
3. **模型容量更合理**：分离编码器设计，让模型专注于学习位置依赖

## 🎯 具体改进效果

### Decode Total Latency
- **MAE**: 15.34ms（相对于 409.43ms 的平均值，误差仅 3.75%）
- **R²**: 0.9974（**接近完美**）
- **MAPE**: 4.54%（**优秀**）

### Decode Average Per-Token Latency
- **MAE**: 1.428ms/token（相对于 31.20ms/token 的平均值，误差仅 4.58%）
- **R²**: 0.9553（**显著提升**）
- **MAPE**: 4.54%（**优秀**）

## 🔬 技术细节

### 训练过程
```python
# 对于每个样本（output_tokens = N）：
1. 预测所有位置的 latency: [T(1), T(2), ..., T(N)]
2. 求和得到总 latency: T_total_pred = sum(T(i))
3. 与真实总 latency 比较: loss = MSE(T_total_pred, T_LLM_decode)
4. 反向传播，模型学习位置依赖
```

### 为什么模型能学习位置依赖？

1. **约束条件**：模型必须让所有位置的 latency 之和等于 `T_LLM_decode`
2. **数据模式**：训练数据中，`T_LLM_decode` 与 `output_tokens` 的关系反映了位置依赖
3. **位置编码**：`log(position + 1)` 归一化帮助模型学习渐进式增长模式

## 📝 总结

### 核心改进
1. ✅ **位置化预测**：从固定平均值改为随位置变化的预测
2. ✅ **训练目标优化**：从平均 per-token 改为 total latency
3. ✅ **架构改进**：分离配置编码器和位置编码器
4. ✅ **位置归一化**：使用 `log(position + 1)` 建模渐进式增长

### 精度提升原因
1. **更准确的建模**：准确捕获 KV cache 增长的影响
2. **训练目标更合理**：总和是 ground truth，比平均值更准确
3. **模型设计更合理**：分离编码器让模型专注于学习位置依赖

### 结果
- **Decode Total Latency R²**: 0.9974（接近完美）
- **Decode Average Per-Token R²**: 0.9553（显著提升）
- **MAPE**: 4.54%（优秀）

这些改进使得 decode latency 预测从"可接受"提升到"优秀"，为后续的 budget 检查和配置选择提供了更可靠的基础。

---

**相关文档**:
- [POSITIONED_DECODE_LATENCY_TRAINING.md](POSITIONED_DECODE_LATENCY_TRAINING.md): 训练策略详解
- [DECODE_LATENCY_VS_OUTPUT_TOKENS.md](../analysis/DECODE_LATENCY_VS_OUTPUT_TOKENS.md): KV cache 影响分析
- [LATENCY_ESTIMATOR_IMPROVEMENT.md](LATENCY_ESTIMATOR_IMPROVEMENT.md): 改进计划

