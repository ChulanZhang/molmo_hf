# Decode Phase Design

## 关键设计原则

**Decode阶段不需要运行controller，只按照prefill阶段选择的配置运行模型即可。**

## 当前实现分析

### Prefill阶段

1. **Stage1 Controller**：预测 tier 和 insertion_position
2. **运行模型到insertion_position**：提取 latency_token
3. **Stage2 Controller**：预测 knob2 (top_k) 和 knob3 (num_blocks)
4. **应用配置**：
   - `_set_top_k(top_k, start_layer=insertion_position)` - 设置后续blocks的top_k
   - `_set_block_mask(num_active_blocks, start_block=insertion_position)` - 选择active blocks
5. **执行Prefill**：运行完整的prefill forward pass

### Decode阶段

**当前实现**：
- 配置（top_k和num_active_blocks）在prefill之前已经设置到blocks上
- `model.generate()`在decode阶段会多次调用`model.forward()`
- 每次`forward()`调用都会使用相同的block配置（因为`block.mlp.top_k`已经被设置）
- **问题**：在decode阶段，`model.forward()`可能还会重新添加budget token

### 需要修复的问题

1. **Budget Token在Decode阶段**：
   - 当前：每次`forward()`调用都会检查`latency_budget`和`budget_encoder`
   - 在decode阶段（`past_key_values is not None`），不应该重新添加budget token
   - Budget token只在prefill阶段添加一次

2. **Controller在Decode阶段**：
   - 当前：配置在prefill之前设置，decode阶段应该保持不变
   - 需要确保decode阶段不会重新运行controller

## 修复方案

### 1. Budget Token只在Prefill阶段添加

修改`model.forward()`，只在prefill阶段（`past_key_values is None`）添加budget token：

```python
# Add latency budget token to the sequence (if provided)
# Only add in prefill phase, not in decode phase
if latency_budget is not None and budget_encoder is not None and past_key_values is None:
    # ... add budget token logic
```

### 2. 确保Decode阶段使用Prefill配置

当前实现已经正确：
- `_set_top_k()`在prefill之前设置，blocks的`top_k`属性会被保持
- `_set_block_mask()`选择的blocks在decode阶段也会被使用（通过importance scores）

### 3. Budget Encoder MLP是否需要训练？

**建议**：可以训练MLP，但sinusoidal encoding保持固定。

**理由**：
- Sinusoidal encoding是确定性的，不需要学习
- MLP可以学习如何将256维编码映射到d_model维token embedding
- 这可能有助于优化budget信息的表示

**实现**：将budget_encoder的MLP参数加入到optimizer中。

