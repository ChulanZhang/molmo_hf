# Decode Latency 预测效果分析

> **文档目的**: 分析decode per-token latency预测效果较差的原因和改进方案

## 📊 当前性能表现

### 评估结果对比

| 指标 | Prefill Latency | Decode Per-Token Latency |
|------|----------------|-------------------------|
| **MAE** | 5.49ms | 6.51ms/token |
| **RMSE** | 7.30ms | 7.77ms/token |
| **MAPE** | 3.54% | 21.01% |
| **R²** | 0.9300 | **0.2483** ⚠️ |
| **Relative Error** | 3.52% | 20.88% |

### 关键问题

**Decode latency的R²只有0.2483**，说明模型只能解释24.83%的方差，有75%的方差无法用当前输入特征解释。

---

## 🔍 原因分析

### 1. Decode Latency本身变异性更大

**变异性对比**:
- **Prefill CV**: 18.00% (变异系数)
- **Decode CV**: 29.87% (变异系数)
- **Decode是Prefill的1.66倍更不稳定**

即使在**相同配置**下，decode latency的CV也达到19-20%，说明：
- 即使配置完全相同，decode latency也有很大变异性
- 可能有其他因素影响decode latency，不仅仅是配置参数

### 2. 可能的影响因素

Decode latency可能受到以下因素影响，但这些因素**不在当前模型的输入特征中**：

1. **输出token的内容和长度分布**
   - 不同token的计算复杂度不同
   - 长输出和短输出的per-token latency可能不同
   - 但我们不知道会生成多少个token（这是latency estimator的设计前提）

2. **硬件状态**
   - GPU温度、频率波动
   - 内存带宽竞争
   - 系统负载

3. **测量误差**
   - Decode阶段通常很快（每个token 20-30ms）
   - 测量精度可能不够
   - 短输出的测量误差相对更大

4. **模型内部状态**
   - KV cache的状态
   - Attention patterns的变化
   - 这些难以量化

### 3. 数据分布问题

从评估结果看：
- **Mean Target**: 31.20ms/token
- **Std Target**: 8.97ms/token
- **CV**: 28.7%

即使在同一配置下，decode latency的分布也很宽，说明：
- 数据本身就有很大噪声
- 模型难以学习到清晰的模式

---

## 💡 改进方案

### 方案1: 接受当前性能（推荐）

**理由**:
- Decode latency的变异性本身很大（CV ~30%）
- 相对误差20.88%虽然接近阈值，但在可接受范围内
- 对于latency estimator的主要用途（RL训练加速），这个精度可能已经足够

**使用建议**:
- 在RL训练中使用时，可以适当增加safety margin
- 对于budget检查，可以设置更保守的阈值

### 方案2: 改进模型架构

**思路**: 使用更复杂的模型来捕捉decode latency的模式

```python
# 可能的改进
class LatencyEstimator(nn.Module):
    def __init__(self):
        # 为decode使用更深的网络
        self.decode_encoder = nn.Sequential(
            # 更深的网络
            nn.Linear(5, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.decode_head = nn.Linear(256, 1)
```

**风险**: 可能过拟合，因为数据本身噪声大

### 方案3: 数据清洗和增强

**思路**: 
1. 更严格的异常值过滤（例如，使用IQR方法）
2. 增加更多训练数据
3. 使用数据增强（但需要谨慎，避免引入偏差）

### 方案4: 分离模型

**思路**: 为decode latency训练单独的、更复杂的模型

```python
# 两个独立的estimator
prefill_estimator = LatencyEstimator(...)  # 简单模型
decode_estimator = LatencyEstimator(...)   # 更复杂的模型
```

### 方案5: 使用分位数回归

**思路**: 不预测均值，而是预测分位数（如P50, P95），提供不确定性估计

```python
# 预测多个分位数
decode_p50 = model.predict_p50(...)
decode_p95 = model.predict_p95(...)
# 使用P95进行保守估计
```

---

## 📈 实际影响评估

### 对Controller训练的影响

在GRPO训练中使用latency estimator时：
- **Prefill预测准确**（R²=0.93）：可以准确估计prefill latency
- **Decode预测有误差**（R²=0.25）：但相对误差20.88%可能仍可接受

**建议**:
- 在计算reward时，可以给decode latency增加safety margin（如+25%）
- 或者使用P95分位数而不是均值

### 对Budget检查的影响

在检查configuration是否满足budget时：
- 如果使用均值预测，可能低估latency
- 建议使用保守估计（均值 + 1-2个标准差）

---

## ✅ 推荐方案

**短期（当前）**:
1. **接受当前性能**：20.88%的相对误差在可接受范围内
2. **使用保守估计**：在budget检查时，给decode latency增加20-25%的safety margin
3. **监控实际使用效果**：在RL训练中观察实际效果

**中期（如果需要改进）**:
1. **尝试更复杂的模型**：为decode使用更深的网络
2. **数据清洗**：更严格的异常值过滤
3. **分位数回归**：预测P95分位数，提供不确定性估计

**长期（如果问题严重）**:
1. **重新设计**：考虑是否需要预测decode latency，或者只预测prefill
2. **在线校准**：在推理时进行在线校准

---

## 📚 相关文档

- **[LATENCY_ESTIMATOR_DESIGN.md](LATENCY_ESTIMATOR_DESIGN.md)**: Latency Estimator设计文档
- **[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)**: 评估指南

---

**最后更新**: 2026-01-08  
**维护者**: Controller Team



