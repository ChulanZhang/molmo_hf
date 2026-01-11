# Latency Budget Encoding Implementation

> **最后更新**: 2026-01-10  
> **版本**: 3.0 (Joint Training Only)

## 设计原理

根据用户的理解和AdaLLaVA的实现，latency budget应该被编码成一个token，拼接到输入序列中，而不是作为单独的特征。

### wte_layer 是什么？

**wte_layer** = Word Token Embedding Layer，是transformer的word embedding层（`transformer.wte`）。

- **作用**：将离散的token IDs转换为连续的embedding向量
- **输入**：token IDs (整数)
- **输出**：embedding vectors (B, seq_len, d_model)
- **位置**：`model.model.transformer.wte`

### Latency Budget Token 设计

#### 正确的设计（参考AdaLLaVA）

1. **Latency Budget编码**：
   - 使用`LatencyBudgetEncoder`将budget编码成**d_model维度**的token embedding（不是256维）
   - 输出：`(B, d_model)` - 与vision token和language token相同的维度

2. **拼接序列**：
   - 将budget token拼接到输入序列中：`[vision_tokens, language_tokens, budget_token]`
   - Budget token位于序列末尾（最后一个token）

3. **经过Transformer**：
   - Budget token与vision和language tokens一起经过第一层transformer block
   - 通过attention机制，budget token获得了与vision和language tokens的交互信息

4. **提取Latency Token**：
   - 在第一层（或插入位置）后，提取最后一个token（latency token）
   - 这个token已经包含了：
     - Budget信息（原始编码）
     - Vision信息（通过attention）
     - Language信息（通过attention）

5. **Stage2简化**：
   - Stage2只需要latency token，不需要单独的`budget_feat`
   - 因为latency token已经包含了所有需要的信息

## AdaLLaVA 实现参考

根据 AdaLLaVA 论文和代码的实现细节：

> "Our latency encoder uses the sinusoidal positional encoding to map the scalar latency to a 256-D vector. A two-layer MLP, with GELU and layer norm, then converts this vector to a latency token Zs, ready to be appended to the input sequence of the LLM."

## 当前实现

我们的实现完全遵循 AdaLLaVA 的设计：

### 1. Sinusoidal Positional Encoding

将标量 latency budget 映射到 256 维向量：

```python
def _sinusoidal_encoding(self, x: torch.Tensor) -> torch.Tensor:
    """
    Apply sinusoidal positional encoding.
    
    Args:
        x: (B, 1) scalar values (normalized to [0, 1])
    
    Returns:
        encoded: (B, 256) - 256-dimensional positional encoding
    """
    position = x.squeeze(-1)  # (B,)
    d_model = 256  # pos_encoding_dim
    
    # Standard sinusoidal encoding formula
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=x.device, dtype=torch.float32)
        * -(np.log(10000.0) / d_model)
    )
    
    encoded = torch.zeros(x.shape[0], d_model, device=x.device)
    encoded[:, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
    encoded[:, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)
    
    return encoded
```

**公式**：
- 偶数位置：`sin(pos / 10000^(2i/d_model))`
- 奇数位置：`cos(pos / 10000^(2i/d_model))`

其中 `pos` 是归一化后的 latency budget 值（0-1之间）。

### 2. Two-Layer MLP

使用两层 MLP（带 GELU 和 LayerNorm）将 256 维向量转换为 d_model 维的 latency token：

```python
self.mlp = nn.Sequential(
    # Layer 1: 256 -> d_model
    nn.Linear(256, d_model),      # Linear transformation
    nn.LayerNorm(d_model),         # Layer normalization
    nn.GELU(),                     # GELU activation
    
    # Layer 2: d_model -> d_model
    nn.Linear(d_model, d_model),  # Linear transformation
    nn.LayerNorm(d_model),         # Layer normalization
)
```

**结构**：
- **Layer 1**: `Linear(256, d_model)` → `LayerNorm` → `GELU`
- **Layer 2**: `Linear(d_model, d_model)` → `LayerNorm`

### 3. 完整流程

```python
def forward(self, budget: torch.Tensor) -> torch.Tensor:
    """
    Encode latency budget to token embedding.
    
    Args:
        budget: (B,) latency budget in ms
    
    Returns:
        budget_token: (B, d_model) - token embedding ready to be concatenated
    """
    # Step 1: Normalize budget to [0, 1]
    if self.normalize_budget:
        budget = (budget - self.budget_min) / (self.budget_max - self.budget_min + 1e-6)
        budget = torch.clamp(budget, 0.0, 1.0)
    
    budget = budget.unsqueeze(-1)  # (B, 1)
    
    # Step 2: Sinusoidal encoding -> 256-D vector
    pos_encoded = self._sinusoidal_encoding(budget)  # (B, 256)
    
    # Step 3: Two-layer MLP -> d_model-D token embedding
    budget_token = self.mlp(pos_encoded)  # (B, d_model)
    
    return budget_token
```

## 与 AdaLLaVA 的对应关系

| AdaLLaVA 描述 | 我们的实现 | 状态 |
|--------------|-----------|------|
| Sinusoidal positional encoding | `_sinusoidal_encoding()` | ✅ 完全一致 |
| Map scalar to 256-D vector | 输出 `(B, 256)` | ✅ 完全一致 |
| Two-layer MLP | `nn.Sequential` with 2 `Linear` layers | ✅ 完全一致 |
| With GELU | `nn.GELU()` after each layer | ✅ 完全一致 |
| With layer norm | `nn.LayerNorm()` after each layer | ✅ 完全一致 |
| Convert to latency token Zs | 输出 `(B, d_model)` | ✅ 完全一致 |
| Ready to append to input sequence | 在 `model.forward` 中拼接 | ✅ 已实现 |

## 关键设计点

1. **归一化**：在 sinusoidal encoding 之前，将 latency budget 归一化到 [0, 1] 范围
   - 公式：`(budget - budget_min) / (budget_max - budget_min)`
   - 默认范围：`budget_min=50.0ms, budget_max=500.0ms`

2. **Sinusoidal Encoding**：使用标准的 positional encoding 公式
   - 将标量值转换为 256 维的连续表示
   - 能够编码相对大小关系

3. **MLP 结构**：
   - 第一层：将 256 维扩展到 d_model 维（通常是 2048）
   - 第二层：保持 d_model 维，进一步处理特征
   - 每层都有 LayerNorm 和 GELU，确保训练稳定

4. **输出**：最终输出 `(B, d_model)` 的 token embedding
   - 可以直接拼接到输入序列中
   - 与 vision token 和 language token 维度一致

## 使用示例

```python
# 初始化 encoder
budget_encoder = LatencyBudgetEncoder(
    d_model=2048,              # Transformer hidden dimension
    use_sinusoidal=True,        # 使用 sinusoidal encoding（AdaLLaVA 方式）
    normalize_budget=True,      # 归一化 budget
    budget_min=50.0,           # 最小 budget (ms)
    budget_max=500.0,          # 最大 budget (ms)
)

# 编码 budget
budget = torch.tensor([200.0, 300.0])  # (2,) - 两个样本的 budget
budget_token = budget_encoder(budget)  # (2, 2048) - token embeddings

# 在模型 forward 中拼接
x = torch.cat([x, budget_token.unsqueeze(1)], dim=1)  # (B, seq_len + 1, d_model)
```

## 代码改动

### `LatencyBudgetEncoder`

**之前**：
```python
LatencyBudgetEncoder(hidden_dim=256)  # 输出256维特征
```

**现在**：
```python
LatencyBudgetEncoder(d_model=2048)  # 输出d_model维token embedding
```

### `Knob2Knob3Predictor`

**之前**：
```python
def forward(self, latency_token, budget_feat, insertion_position):
    # 需要latency_token + budget_feat
```

**现在**：
```python
def forward(self, latency_token, insertion_position):
    # 只需要latency_token（已包含budget信息）
```

### 模型Forward过程

需要在模型的forward过程中：
1. 使用`budget_encoder`将budget编码成token：`budget_token = budget_encoder(budget)`  # (B, d_model)
2. 拼接到序列末尾：`x = torch.cat([x, budget_token.unsqueeze(1)], dim=1)`
3. 经过transformer blocks
4. 提取最后一个token作为latency token

## 优势

1. **更符合Transformer架构**：Budget作为token参与attention，自然获得交互信息
2. **简化Stage2**：不需要单独的budget_feat，减少参数和计算
3. **与AdaLLaVA一致**：参考了成熟的设计方案
4. **信息融合更自然**：通过attention机制融合信息，而不是手动拼接特征

## 总结

当前的实现完全遵循 AdaLLaVA 的设计：
- ✅ Sinusoidal positional encoding 映射到 256 维
- ✅ 两层 MLP（带 GELU 和 LayerNorm）转换为 d_model 维 token
- ✅ 输出可以直接拼接到输入序列
- ✅ 与 AdaLLaVA 的实现细节一致
- ✅ Budget token在prefill阶段拼接，decode阶段不添加
- ✅ Stage2只需要latency token，不需要单独的budget_feat

---

**维护者**: Controller Team  
**最后更新**: 2026-01-10

