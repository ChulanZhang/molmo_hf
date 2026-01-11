# Controller分析文档（统一版）

> **文档状态**: 本文档整合了所有controller分析文档，提供完整的技术分析。
> **最后更新**: 2026-01-10
> **版本**: 3.0 (Joint Training Only)

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

**解决方案**: 两阶段预测架构（Joint Training）
- Stage 1: 在vision encoder之前预测Knob1（vision tokens tier + insertion position）
- Stage 2: 在插入位置之后预测Knob2 & Knob3（使用latency token，已包含budget + vision + language交互信息）
- **训练方式**: Joint GRPO Training（端到端训练，共享reward信号）

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

**⚠️ 注意**: Stage 2现在只使用latency token（从插入位置后的transformer block提取），不再需要单独的vision和language features，因为latency token已经包含了与vision和language tokens的交互信息。

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

#### 方案选择：Budget Token（参考AdaLLaVA）

**当前实现**：
- 使用sinusoidal positional encoding将scalar转换为256-D向量
- 使用两层MLP（GELU + layer norm）转换为d_model维token embedding
- **拼接到输入序列**：`[vision_tokens, language_tokens, budget_token]`
- Budget token位于序列末尾，经过transformer blocks后获得交互信息

**实现**：
```python
class LatencyBudgetEncoder(nn.Module):
    """Encode latency budget to token embedding."""
    def __init__(self, d_model=2048, use_sinusoidal=True):
        super().__init__()
        self.d_model = d_model
        self.use_sinusoidal = use_sinusoidal
        
        if use_sinusoidal:
            self.pos_encoding_dim = 256
            self.mlp = nn.Sequential(
                nn.Linear(256, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
    
    def forward(self, budget: torch.Tensor) -> torch.Tensor:
        """Encode budget to token embedding (B, d_model)."""
        # Sinusoidal encoding -> 256-D
        pos_encoded = self._sinusoidal_encoding(budget)
        # MLP -> d_model-D token
        return self.mlp(pos_encoded)
```

**关键设计**：
- Budget token在prefill阶段拼接到输入序列
- 经过transformer blocks后，latency token（最后一个token）包含budget + vision + language交互信息
- Stage2只需要latency token，不需要单独的budget_feat

---

## 3. 输出格式设计

### 3.1 Knob1输出

**格式**: 3个类别的分类（low, medium, high）

**实现**: 
- 输出logits: (B, 3)
- 使用CrossEntropy loss训练
- 推理时使用argmax或softmax采样

### 3.2 Knob2输出

**格式**: 5个类别的分类（4, 5, 6, 7, 8）

**实现**:
- 输出logits: (B, 5)
- 使用GRPO loss训练（joint training）
- 推理时使用argmax或softmax采样
- **应用范围**: 插入位置之后的所有blocks（第一层固定top_k=8）

### 3.3 Knob3输出

**格式**: 5个类别的分类（12, 13, 14, 15, 16 total blocks）

**实现**:
- 输出logits: (B, 5)，根据插入位置动态mask
- 使用GRPO loss训练（joint training）
- 推理时使用argmax或softmax采样
- **Block selection**: 基于预计算的importance score（确定性）
- **动态选项**: 根据插入位置动态调整可用选项数量

**关键设计**: 
- 不需要输出mask（2^16种可能），只需要输出num_blocks（5种可能）
- Knob3值表示总block数（包括第一层和插入位置之前的blocks）
- 第一层固定包含，总是使用

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
- 输入: Language Feature + Budget Token (encoded)
- 输出: Knob1 (tier: low/medium/high) + Insertion Position (1-5)
- 决策时机: Before vision encoder

**Stage 2**: 轻量级MLP
- 输入: Latency Token (从插入位置后的transformer block提取)
- 输出: Knob2 (top_k: 4/5/6/7/8) + Knob3 (num_blocks: 12/13/14/15/16)
- 决策时机: After insertion position
- **简化**: 只需要latency token，不需要单独的vision/language/budget features

---

## 5. 训练方法分析

### 5.1 Joint GRPO Training（唯一训练方式）

**数据来源**: Online execution（实际运行模型）

**训练流程**:
1. 加载真实数据集样本（images + prompts）
2. 为每个样本采样latency budget（从[170ms, 380ms]均匀采样）
3. Stage1预测tier和insertion position
4. 根据tier处理图像，运行vision encoder + projector
5. 运行LLM到插入位置，提取latency token
6. Stage2预测top_k和num_blocks
7. 应用配置，执行模型生成，测量实际latency
8. 计算accuracy和reward
9. GRPO更新（Stage1和Stage2一起更新）

**优点**:
- 端到端优化，Stage1和Stage2协调
- 可以学习复杂的accuracy-latency trade-off
- 可以学习latency budget约束
- 样本效率高（GRPO）
- 使用direct latency measurement，更准确

**缺点**:
- 训练复杂度高
- 训练速度较慢（需要实际运行模型）
- 不能使用大batch size（batch_size=1 per sample）

### 5.2 Latency Measurement

**当前实现**: Direct Measurement（使用PyTorch hooks）

**测量方法**:
- 使用hooks在vision_backbone和transformer blocks上测量
- 区分prefill和decode阶段
- Batch size = 1 per sample（确保准确测量）

**不再使用**: Latency Estimator（保留为独立模块，不用于controller训练）

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

**可行性**: ✅ 完全可行，且强烈推荐（当前唯一训练方式）

**方案**: Joint GRPO Training
1. Stage1和Stage2一起训练，共享reward信号
2. LLM frozen（不训练）
3. Budget encoder MLP可训练，sinusoidal encoding固定

**关键技术**:
- GRPO算法（Group Relative Policy Optimization）
- Direct latency measurement（使用hooks）
- Budget token集成（拼接到输入序列）
- Dynamic insertion position（Stage1预测）

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

### 8.2 Latency Measurement

**当前设计**: Direct Measurement（使用PyTorch hooks）

**测量方法**:
- 使用hooks在vision_backbone和transformer blocks上测量
- 区分prefill和decode阶段
- 在同一个流程中测量所有组件，确保环境一致

**不再使用**: Latency Estimator（保留为独立模块，不用于controller训练）

**优势**:
- 更准确（实际测量而非估计）
- 可以捕获硬件特定的latency特性
- 简化设计（不需要estimator）

### 8.3 Joint Training

**唯一训练方式**: Joint GRPO Training
- Stage1和Stage2一起训练
- 共享reward信号
- 端到端优化

**训练模块**:
- Stage1 Controller (trainable)
- Stage2 Controller (trainable)
- Budget Encoder MLP (trainable)
- LLM Model (frozen)
- Budget Encoder Sinusoidal Encoding (frozen)

---

## 附录

### A. 相关文档

- `DESIGN.md`: 统一的设计文档
- `JOINT_TRAINING.md`: Joint Training详细说明
- `IMPLEMENTATION_SUMMARY.md`: 实现总结

### B. 代码实现

- `experiments/controller/controller.py`: Controller实现（Stage1和Stage2）
- `experiments/controller/feature_extractors.py`: 特征提取（Language, Budget）
- `experiments/controller/joint_grpo_trainer.py`: Joint GRPO训练器
- `experiments/controller/train_joint_controller.py`: 主训练脚本
- `experiments/controller/model_forward_with_dynamic_stage2.py`: 动态forward pass

---

**文档维护**: 本文档整合了所有controller分析文档，提供完整的技术分析。  
**最后更新**: 2026-01-10  
**版本**: 3.0 (Joint Training Only)







