# Controller分析文档（统一版）

> **文档状态**: 本文档整合了所有controller分析文档，提供完整的技术分析。
> **最后更新**: 2026-01-01
> **版本**: 2.0

## 目录

1. [概述](#1-概述)
2. [输入特征设计分析](#2-输入特征设计分析)
3. [输出格式设计](#3-输出格式设计)
4. [Controller架构分析](#4-controller架构分析)
5. [训练方法分析](#5-训练方法分析)
6. [Overhead分析](#6-overhead分析)
7. [可行性分析](#7-可行性分析)
8. [关键技术点](#8-关键技术点)

---

## 1. 概述

### 1.1 分析目标

本文档深入分析controller的设计选择，包括：
- 输入特征的设计和提取方式
- Controller架构的选择
- 训练方法的对比
- Overhead和可行性的详细分析

### 1.2 核心问题

**关键设计问题**: Knob1必须在vision encoder之前决定，但vision feature需要经过vision encoder才能获得。

**解决方案**: 两阶段预测架构
- Stage 1: 在vision encoder之前预测Knob1（不使用vision feature）
- Stage 2: 在vision encoder之后预测Knob2 & Knob3（可以使用vision feature）

---

## 2. 输入特征设计分析

### 2.1 Vision Token Feature

#### 方案选择：Global Crop + CLIP Vision Encoder

**推荐方案**：
- 使用**global crop**（单张resize后的图像，336×336）
- 通过CLIP vision encoder提取特征
- **不过projector**：projector是用于将vision feature映射到LLM的hidden dimension，对于controller来说，我们只需要vision的语义信息
- **使用pooling**：建议使用**mean pooling**或**CLS token**来获得固定长度的特征向量

**理由**：
1. **Global crop的优势**：
   - 计算开销小（只需处理一张图像，而非多个crops）
   - 保留了图像的全局信息，足以让controller判断图像复杂度
   - 与实际推理时的多crop处理不同，但作为特征提取足够有效

2. **不过projector的原因**：
   - Projector的作用是将vision feature对齐到LLM的embedding space
   - Controller不需要这种对齐，只需要vision的语义特征
   - 减少计算开销和参数

3. **Pooling的必要性**：
   - Vision encoder输出是patch-level的特征（如24×24=576个patches）
   - Controller需要固定长度的特征向量
   - Mean pooling或CLS token都能提供全局图像表示

**⚠️ 注意**: 对于Stage 2，vision feature必须经过projector，因为Stage 2在vision encoder+projector之后。

### 2.2 Language Token Feature

#### 方案选择：Tokenizer + WTE + Pooling

**推荐方案**：
- 使用tokenizer将prompt转换为token IDs
- 通过Word Token Embedding (WTE)获得embeddings
- 使用**mean pooling**获得固定长度的特征向量

**理由**：
1. **简单高效**：不需要额外的编码器
2. **语义信息**：WTE已经包含了语言的语义信息
3. **固定长度**：Pooling后得到固定长度的特征，便于后续处理

**实现**：
```python
def extract_language_feature(prompt: str, tokenizer, wte):
    """Extract language feature from prompt."""
    # Tokenize
    tokens = tokenizer(prompt, return_tensors="pt")
    
    # Get embeddings
    embeddings = wte(tokens['input_ids'])
    
    # Mean pooling
    feature = embeddings.mean(dim=1)  # (B, d_model)
    
    return feature
```

### 2.3 Latency Budget Feature

#### 方案选择：Latency Encoder（参考AdaLLaVA）

**推荐方案**：
- 使用sinusoidal positional encoding将scalar转换为256-D向量
- 使用两层MLP（GELU + layer norm）转换为latency token

**实现**：
```python
class LatencyBudgetEncoder(nn.Module):
    """Encode latency budget to feature."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        # Sinusoidal encoding
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, budget: torch.Tensor) -> torch.Tensor:
        """Encode budget scalar to feature."""
        # Sinusoidal encoding
        pos_feat = self.pos_encoder(budget.unsqueeze(-1))
        
        # MLP
        feat = self.mlp(pos_feat)
        
        return feat
```

**简化方案**（用于最小overhead）：
- 直接使用简单的MLP：`nn.Linear(1, hidden_dim)`
- 不需要sinusoidal encoding
- 更小的参数量和计算量

---

## 3. 输出格式设计

### 3.1 Knob1输出

**格式**: 3个类别的分类（low, medium, high）

**实现**: 
- 输出logits: (B, 3)
- 使用CrossEntropy loss训练
- 推理时使用argmax或softmax采样

### 3.2 Knob2输出

**格式**: 5个类别的分类（4, 6, 8, 10, 12）

**实现**:
- 输出logits: (B, 5)
- 使用CrossEntropy loss训练
- 推理时使用argmax或softmax采样

### 3.3 Knob3输出

**格式**: 5个类别的分类（8, 10, 12, 14, 16）

**实现**:
- 输出logits: (B, 5)
- 使用CrossEntropy loss训练
- 推理时使用argmax或softmax采样
- **Block selection**: 基于预计算的importance score（确定性）

**关键设计**: 不需要输出mask（2^16种可能），只需要输出num_blocks（5种可能）

---

## 4. Controller架构分析

### 4.1 架构选项对比

#### 选项A：使用LLM前几层（AdaLLaVA风格）

**架构**:
```
Vision/Lang/Budget Tokens → LLM First N Layers → Scheduler → Knob Heads
```

**优点**:
- 可以利用LLM的表示能力
- 特征已经对齐到LLM空间

**缺点**:
- Overhead大（需要forward pass前几层）
- 需要修改LLM的forward流程
- 不符合最小overhead原则

#### 选项B：独立轻量级MLP（推荐）

**架构**:
```
Pooled Features → MLP → Knob Heads
```

**优点**:
- Overhead小（只有MLP计算）
- 简单直接
- 符合SIGMETRICS标准

**缺点**:
- 可能不如LLM层表达能力强（但足够有效）

**当前实现**: 选项B

### 4.2 两阶段架构

**Stage 1**: 轻量级MLP
- 输入: Language + Budget
- 输出: Knob1 (tier)

**Stage 2**: 轻量级MLP
- 输入: Vision (pooled) + Language (pooled) + Budget
- 输出: Knob2 (top_k) + Knob3 (num_blocks)

---

## 5. 训练方法分析

### 5.1 Supervised Learning

**数据来源**: Core experiment JSON文件

**训练流程**:
1. 加载core experiment结果
2. 提取features（language, budget, vision）
3. 使用ground truth knob values作为label
4. 训练controller（CrossEntropy loss）

**优点**:
- 简单稳定
- 训练快
- 不需要online execution

**缺点**:
- 可能无法学习latency budget约束
- 依赖profiling数据的质量

### 5.2 RL方法（GRPO）

**数据来源**: Online execution + Latency Estimator

**训练流程**:
1. Controller预测knob configuration
2. 使用Latency Estimator预估latency
3. 真实执行模型获取accuracy
4. 计算reward
5. GRPO更新

**优点**:
- 可以学习复杂的accuracy-latency trade-off
- 可以学习latency budget约束
- 样本效率高（GRPO）

**缺点**:
- 训练复杂度高
- 需要Latency Estimator

### 5.3 混合方法（推荐）

**Stage 1**: Supervised Learning
- 简单稳定
- 快速收敛

**Stage 2**: GRPO
- 学习复杂约束
- 高效样本利用

---

## 6. Overhead分析

### 6.1 实际Overhead

**关键理解**：
- **前几层不是overhead**：无论有没有controller，前几层都要forward（如果使用LLM层）
- **Controller overhead = Controller本身的计算**：
  - Stage 1: 轻量级MLP（几万参数）
  - Stage 2: 轻量级MLP（几万到几十万参数）
  - **总参数量很小**（<250K参数）
  - **计算开销可忽略**（<0.2ms）

**节省的时间来源**：
- 后12层通过减少top_k（如从8减到4）节省计算
- 后12层通过跳过某些blocks节省计算
- **节省量 = 后12层的计算减少量**

### 6.2 Overhead估算

**Controller Overhead**:
```
Stage 1: ~10K-50K parameters, ~0.01-0.1ms
Stage 2: ~50K-200K parameters, ~0.1ms
Total: ~60K-250K parameters, ~0.11-0.2ms
```

**相对Overhead**:
- 典型inference时间: ~200-500ms
- Controller overhead: 0.11-0.2ms / 300ms = **0.037-0.067%**

### 6.3 零Overhead的Knob应用

**Top-K应用**: 直接修改属性
- `block.mlp.top_k = new_value`
- 零overhead
- 不影响计算图

**Block Mask应用**: 使用BlockMaskWrapper
- 跳过blocks时只做identity pass-through
- Overhead可忽略

---

## 7. 可行性分析

### 7.1 动态改变top_k

**可行性**: ✅ 完全可行，零overhead

**技术细节**:
- `top_k`是`MolmoeSparseMoeBlock`的一个**普通Python属性**（不是模型参数）
- 在forward时直接读取：`torch.topk(routing_weights, self.top_k, dim=-1)`
- **可以直接修改**，无需重新构建计算图

**实现方式**:
```python
# 直接修改属性（推荐，最简单高效）
for i in range(4, 16):
    block = model.transformer.blocks[i]
    if hasattr(block, 'mlp') and hasattr(block.mlp, 'top_k'):
        block.mlp.top_k = new_top_k  # 直接修改，零overhead
```

### 7.2 Block Masking

**可行性**: ✅ 完全可行

**实现方式**: 使用BlockMaskWrapper
- 在forward过程中检查mask
- 如果mask为False，跳过block（identity pass-through）
- 如果mask为True，正常执行

### 7.3 Joint Training

**可行性**: ✅ 完全可行，且强烈推荐

**方案**: 两阶段训练
1. Stage 1: 训练controller（LLM frozen）
2. Stage 2: Joint training（controller + 后12层）

**关键技术**:
- Soft top_k（Gumbel-Softmax）
- Soft block mask（sigmoid）

---

## 8. 关键技术点

### 8.1 Importance-Based Block Selection

**关键设计**: 使用预计算的importance score进行block selection

**优势**:
- 简化输出空间（2^16 → 5）
- 确定性选择，稳定可靠
- 数据无关（基于预计算的importance）

**实现**:
```python
def select_blocks_by_importance(
    importance_scores: Dict[int, float],
    num_blocks: int,
) -> List[int]:
    """Select top-N most important blocks."""
    sorted_blocks = sorted(
        importance_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [block_idx for block_idx, _ in sorted_blocks[:num_blocks]]
```

### 8.2 Latency Estimator

**设计**: 轻量级MLP（2-3层）

**输入特征**:
- vision_tokens, text_tokens, output_tokens
- tier_idx, top_k, num_active_blocks

**输出预测**: 阶段分解
- T_vision_encoder, T_projector
- T_LLM_prefill, T_LLM_decode_per_token

**用途**: 在RL训练中预估latency，避免batch_size=1限制

### 8.3 两阶段训练

**Stage 1**: Supervised Learning
- 快速收敛
- 稳定可靠

**Stage 2**: GRPO
- 学习复杂约束
- 高效样本利用

---

## 附录

### A. 相关文档

- `DESIGN.md`: 统一的设计文档
- `controller_implementation_details.md`: 实现细节
- `knob1_predictor_variants.md`: Knob1变体分析

### B. 代码实现

- `experiments/controller/two_stage_controller.py`: 两阶段controller
- `experiments/controller/feature_extractors.py`: 特征提取
- `experiments/controller/latency_estimator.py`: Latency estimator

---

**文档维护**: 本文档整合了所有controller分析文档，提供完整的技术分析。





