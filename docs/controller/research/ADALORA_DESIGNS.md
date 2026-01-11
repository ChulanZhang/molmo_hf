# AdaLoRA-Inspired设计（两种思路）

## 概述

本文档详细描述两种AdaLoRA-inspired的设计思路，两种方案都保留并实现。

## 思路1: 两阶段预测（当前实现）

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│              Two-Stage AdaLoRA-Inspired Design               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1: Knob1 Prediction (BEFORE Vision Encoder)         │
│  ├─ Input: Budget only 或 Budget + Language                 │
│  ├─ Network: Tiny MLP 或 Semantic Router                   │
│  └─ Output: Vision Tokens Tier (low/medium/high)           │
│                                                              │
│  ↓ Image Preprocessing (based on Knob1)                   │
│  ↓ Vision Encoder + Projector                              │
│                                                              │
│  Stage 2: Knob2 & Knob3 Prediction (AFTER Projector)       │
│  ├─ Input: Vision + Language + Budget tokens               │
│  ├─ Method: 借用LLM前3层做attention融合                    │
│  ├─ Network: LLM Layers 0-2 (unified attention)            │
│  └─ Output: Top-K + Transformer Blocks (layers 3-15)     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 详细设计

#### Stage 1: Knob1预测

**时机**: 在vision encoder之前

**输入选项**:
- **选项A**: Budget only（最小overhead）
- **选项B**: Budget + Language（中等复杂度，可集成Semantic Router）

**网络**: 
- 选项A: Tiny MLP (~10K params)
- 选项B: Lightweight MLP 或 Semantic Router

**输出**: Tier (low/medium/high)

#### Stage 2: Knob2 & Knob3预测

**时机**: 在vision encoder+projector之后，LLM之前

**输入**: 
- Vision tokens (经过encoder+projector)
- Language tokens (从prompt)
- Budget token (latency budget编码)

**方法**: 借用LLM前3层做attention融合

**架构**:
```python
class AdaLoRAStage2Predictor(nn.Module):
    """
    Stage 2 predictor using LLM first 3 layers.
    """
    def __init__(self, base_llm, hidden_dim=256):
        super().__init__()
        self.base_llm = base_llm
        self.num_llm_layers = 3  # Use first 3 layers
        
        # Projections to LLM dimension
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.lang_proj = nn.Linear(lang_dim, d_model)
        self.budget_proj = nn.Linear(budget_dim, d_model)
        
        # Scheduler (lightweight MLP)
        self.scheduler = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Two heads
        self.knob2_head = nn.Linear(hidden_dim, 5)  # top_k
        self.knob3_head = nn.Linear(hidden_dim, 5)  # num_blocks
    
    def forward(self, vision_tokens, lang_tokens, budget_token):
        # Project to LLM dimension
        vision_proj = self.vision_proj(vision_tokens)  # (B, num_vision, d_model)
        lang_proj = self.lang_proj(lang_tokens)  # (B, num_lang, d_model)
        budget_proj = self.budget_proj(budget_token)  # (B, 1, d_model)
        
        # Concatenate all tokens
        all_tokens = torch.cat([vision_proj, lang_proj, budget_proj], dim=1)  # (B, seq_len, d_model)
        
        # Process through LLM first 3 layers (unified attention)
        hidden = all_tokens
        for i in range(self.num_llm_layers):
            hidden = self.base_llm.transformer.blocks[i](hidden, use_cache=False)[0]
        
        # Extract budget token hidden state
        budget_hidden = hidden[:, -1, :]  # (B, d_model) - last token is budget token
        
        # Scheduler
        scheduler_hidden = self.scheduler(budget_hidden)  # (B, hidden_dim)
        
        # Predict knobs
        knob2_logits = self.knob2_head(scheduler_hidden)  # (B, 5)
        knob3_logits = self.knob3_head(scheduler_hidden)  # (B, 5)
        
        return {
            'knob2_logits': knob2_logits,
            'knob3_logits': knob3_logits,
        }
```

**执行流程**:
1. Vision tokens, Language tokens, Budget token经过projection对齐到LLM维度
2. 所有tokens拼接，通过LLM前3层做unified attention
3. 提取Budget token的hidden state（经过3层attention后）
4. 通过scheduler（轻量级MLP）处理
5. 两个head分别预测Knob2 (top_k)和Knob3 (num_blocks)

**优势**:
- 利用LLM的表示能力
- Unified attention融合所有信息
- 符合AdaLoRA的设计理念

**Overhead**:
- LLM前3层forward: ~5-10ms（取决于模型大小）
- Scheduler + Heads: ~0.1ms
- **Total**: ~5-10ms

### 完整执行流程

```
1. Input: Image + Prompt + Latency Budget
    ↓
2. Stage 1: Predict Knob1
   - Budget only 或 Budget + Language
   - Output: Tier
    ↓
3. Image Preprocessing (based on Knob1)
    ↓
4. Vision Encoding
   - Vision Encoder + Projector
   - Output: Vision tokens
    ↓
5. Stage 2: Predict Knob2 & Knob3
   - Vision + Language + Budget tokens
   - LLM Layers 0-2 (unified attention)
   - Scheduler + Heads
   - Output: Top-K + Num Blocks
    ↓
6. Apply Knobs to LLM
   - Set top_k for layers 3-15
   - Select blocks by importance (layers 3-15)
    ↓
7. LLM Forward (layers 3-15 with adaptive knobs)
```

## 思路2: 一阶段预测（备选方案）

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│              Single-Stage AdaLoRA-Inspired Design            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Single Stage: All Knobs (AFTER Vision Encoder)            │
│  ├─ Input: Budget + Language + Vision (global crop)        │
│  ├─ Method: 融合所有特征，直接预测三个knob                │
│  ├─ Network: MLP 或 Transformer                            │
│  └─ Output: Tier + Top-K + Transformer Blocks              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 详细设计

#### 输入特征

**Vision Feature**: 
- 使用global crop（单张resize后的图像）
- 通过vision encoder提取特征
- **需要额外的vision encoder pass**

**Language Feature**: 
- Tokenizer + WTE + Pooling

**Budget Feature**: 
- Latency budget编码

#### 网络架构

**选项A: MLP Fusion**
```python
class SingleStageController(nn.Module):
    """
    Single-stage controller predicting all three knobs.
    """
    def __init__(
        self,
        vision_feat_dim=2048,
        lang_feat_dim=2048,
        budget_feat_dim=128,
        hidden_dim=256,
    ):
        super().__init__()
        # Feature projections
        self.vision_proj = nn.Linear(vision_feat_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_feat_dim, hidden_dim)
        self.budget_proj = nn.Linear(budget_feat_dim, hidden_dim)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Three heads
        self.knob1_head = nn.Linear(hidden_dim, 3)  # tier
        self.knob2_head = nn.Linear(hidden_dim, 5)  # top_k
        self.knob3_head = nn.Linear(hidden_dim, 5)  # num_blocks
    
    def forward(self, vision_feat, lang_feat, budget_feat):
        # Project features
        v = F.relu(self.vision_proj(vision_feat))
        l = F.relu(self.lang_proj(lang_feat))
        b = F.relu(self.budget_proj(budget_feat))
        
        # Fuse
        fused = self.fusion(torch.cat([v, l, b], dim=-1))
        
        # Predict all knobs
        return {
            'knob1_logits': self.knob1_head(fused),
            'knob2_logits': self.knob2_head(fused),
            'knob3_logits': self.knob3_head(fused),
        }
```

**选项B: Transformer Fusion**
```python
class SingleStageTransformerController(nn.Module):
    """
    Single-stage controller with transformer fusion.
    """
    def __init__(self, d_model=2048, nhead=8, num_layers=2):
        super().__init__()
        # Feature projections to d_model
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.lang_proj = nn.Linear(lang_dim, d_model)
        self.budget_proj = nn.Linear(budget_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling and heads
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.knob1_head = nn.Linear(d_model, 3)
        self.knob2_head = nn.Linear(d_model, 5)
        self.knob3_head = nn.Linear(d_model, 5)
    
    def forward(self, vision_feat, lang_feat, budget_feat):
        # Project and stack
        v = self.vision_proj(vision_feat).unsqueeze(1)  # (B, 1, d_model)
        l = self.lang_proj(lang_feat).unsqueeze(1)  # (B, 1, d_model)
        b = self.budget_proj(budget_feat).unsqueeze(1)  # (B, 1, d_model)
        
        tokens = torch.cat([v, l, b], dim=1)  # (B, 3, d_model)
        
        # Transformer fusion
        fused = self.transformer(tokens)  # (B, 3, d_model)
        
        # Pool
        pooled = self.pool(fused.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        
        # Predict all knobs
        return {
            'knob1_logits': self.knob1_head(pooled),
            'knob2_logits': self.knob2_head(pooled),
            'knob3_logits': self.knob3_head(pooled),
        }
```

### 执行流程

```
1. Input: Image + Prompt + Latency Budget
    ↓
2. Vision Feature Extraction (Global Crop)
   - Run vision encoder on global crop
   - Extract vision feature
   - Overhead: ~30-50ms
    ↓
3. Language Feature Extraction
   - Tokenizer + WTE + Pooling
    ↓
4. Budget Feature Extraction
   - Latency budget encoding
    ↓
5. Single-Stage Prediction
   - Fuse all features
   - Predict all three knobs simultaneously
   - Output: Tier + Top-K + Num Blocks
    ↓
6. Image Preprocessing (based on Knob1)
   - If tier requires more crops, run vision encoder again
    ↓
7. Vision Encoding (full)
   - Process all crops
    ↓
8. Apply Knobs to LLM
   - Set top_k
   - Select blocks by importance
    ↓
9. LLM Forward (with adaptive knobs)
```

### 优化建议

**问题**: 需要额外的vision encoder pass（global crop）

**优化方案**:

1. **轻量级Vision Encoder**:
   - 使用MobileViT或类似的轻量级模型
   - 减少overhead到~10-20ms

2. **缓存策略**:
   - 如果tier是low，global crop可能就是最终结果
   - 可以复用global crop的vision feature

3. **知识蒸馏**:
   - 训练一个小模型来预测vision complexity
   - 不需要完整的vision encoder

4. **两阶段Vision Encoding**:
   - 第一阶段: Global crop（用于controller）
   - 第二阶段: 根据tier决定是否需要更多crops

## 两种方案对比

| 特性 | 思路1（两阶段） | 思路2（一阶段） |
|------|----------------|----------------|
| **决策时机** | Knob1在vision encoder前，Knob2&3在vision encoder后 | 所有knobs在vision encoder后 |
| **Vision Encoder Pass** | 1次（正常流程） | 2次（global crop + full） |
| **Overhead** | ~5-10ms（LLM前3层） | ~30-50ms（额外vision encoder） |
| **信息利用** | 分阶段利用信息 | 同时利用所有信息 |
| **复杂度** | 中等 | 较高 |
| **灵活性** | 高（可以独立优化每个stage） | 中（需要同时优化所有knobs） |
| **推荐场景** | SIGMETRICS（低overhead） | 准确性优先 |

## 实施计划

### 阶段1: 实现思路1（两阶段）

1. ✅ 已有基础实现（`two_stage_controller.py`）
2. 完善Stage 2的LLM前3层集成
3. 测试和优化

### 阶段2: 实现思路2（一阶段）

1. 实现single-stage controller
2. 优化vision encoder overhead
3. 对比两种方案

### 阶段3: 对比评估

1. 性能对比（accuracy, latency, overhead）
2. 选择最优方案或混合方案

---

**状态**: 两种方案都保留
**优先级**: 思路1优先（当前实现），思路2作为备选







