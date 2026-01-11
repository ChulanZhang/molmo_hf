# Budget Encoder Training Design

## 问题

Budget encoder的sinusoidal encoding是确定性的，但两层MLP是否需要训练？

## 设计决策

### Sinusoidal Encoding（确定性）

- **不需要训练**：Sinusoidal positional encoding是确定性的数学函数
- 将scalar budget映射到256维向量
- 公式：`sin(pos / 10000^(2i/d_model))` 和 `cos(pos / 10000^(2i/d_model))`

### MLP（可学习）

- **可以训练**：MLP的参数是可学习的
- 将256维sinusoidal编码映射到`d_model`维token embedding
- 可以学习如何更好地表示budget信息

## 实现

### 当前实现（已更新）

```python
# Optimizer: train controllers and budget encoder MLP
optimizer_params = [
    {'params': self.knob1_predictor.parameters(), 'lr': stage1_lr},
    {'params': self.knob2_knob3_predictor.parameters(), 'lr': lr},
]

# Add budget encoder MLP parameters (only MLP, not sinusoidal encoding)
if budget_encoder is not None:
    if hasattr(budget_encoder, 'mlp'):
        optimizer_params.append({
            'params': budget_encoder.mlp.parameters(),
            'lr': lr,  # Use same LR as Stage2
        })
```

### 训练的参数

1. **Stage1 Controller**：预测tier和insertion_position
2. **Stage2 Controller**：预测top_k和num_blocks
3. **Budget Encoder MLP**：学习如何将256维编码映射到d_model维token

### 不训练的参数

1. **Sinusoidal Encoding**：确定性的，不需要训练
2. **LLM Model**：Frozen，不训练
3. **Language Feature Extractor**：wte_layer被freeze

## 优势

1. **学习更好的表示**：MLP可以学习如何更好地将budget信息编码为token
2. **端到端优化**：Budget token的表示可以与controller和模型一起优化
3. **保持确定性**：Sinusoidal encoding保持确定性，确保编码的一致性

## 总结

- ✅ **训练Budget Encoder的MLP**：可以学习更好的budget token表示
- ❌ **不训练Sinusoidal Encoding**：保持确定性
- ✅ **Joint Training**：Budget encoder MLP与controllers一起训练

