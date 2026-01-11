# Controller设计文档（统一版）

> **文档状态**: 本文档整合了所有controller设计文档，基于现有代码实现和SIGMETRICS标准。
> **最后更新**: 2026-01-10
> **版本**: 3.0 (Joint Training Only)

## 目录

1. [设计概述](#1-设计概述)
2. [系统约束与执行流程](#2-系统约束与执行流程)
3. [架构设计](#3-架构设计)
4. [Knob设计细节](#4-knob设计细节)
5. [输入特征设计](#5-输入特征设计)
6. [训练方法](#6-训练方法)
7. [Overhead分析](#7-overhead分析)
8. [实现细节](#8-实现细节)
9. [性能指标](#9-性能指标)
10. [关键设计决策](#10-关键设计决策)

---

## 1. 设计概述

### 1.1 设计目标

**SIGMETRICS标准**：
1. **低Overhead**: Controller开销 <0.1% of total inference
2. **高效性**: 决策时间 <0.2ms
3. **有效性**: 显著提升accuracy-latency trade-off
4. **简洁性**: 设计简单，易于部署
5. **可扩展性**: 适用于不同硬件和模型规模

### 1.2 核心设计理念

**两阶段预测架构**：
- **Stage 1**: 在vision encoder之前预测Knob1（vision tokens tier）
- **Stage 2**: 在vision encoder之后预测Knob2 & Knob3（MoE top-K和transformer blocks）

**关键原则**：
- 最小化controller开销
- 符合系统执行流程约束
- 使用importance-based pruning简化Knob3

### 1.3 三个Knob的最终设计

| Knob | 控制内容 | 决策时机 | 实现方式 | 输出空间 |
|------|---------|---------|---------|---------|
| **Knob1** | Vision tokens tier (low/medium/high) + Insertion Position (1-5) | Before vision encoder | Stage 1 predictor | 3 tiers × 5 positions |
| **Knob2** | MoE top-K (4/5/6/7/8) | After insertion position | Stage 2 predictor | 5 choices |
| **Knob3** | Transformer blocks count (12/13/14/15/16 total blocks) | After insertion position | **Importance-based pruning** | 5 choices |

**关键改进**：
- Knob3从mask预测（2^16）简化为num_blocks预测（5），基于预计算的importance score
- Block selection是确定性的，不依赖输入

---

## 2. 系统约束与执行流程

### 2.1 系统约束

**VLM架构**: `Vision Encoder → Projector → LLM`

**关键约束**：
1. **Knob1**: 必须在vision encoder之前决定
   - Crop数量决定图像处理方式（tiling, resize）
   - 一旦进入vision encoder，crop数量就固定了
2. **Knob2 & Knob3**: 必须在LLM前几层或之前决定
   - Top-K和blocks影响后续计算
   - 避免重复计算

### 2.2 执行流程

```
1. Input: Image + Prompt + Latency Budget
    ↓
2. Stage 1: Predict Knob1
   - Extract: Language Feature (from prompt)
   - Extract: Budget Token (encoded as d_model-dim token, concatenated to input)
   - Predict: Vision Tokens Tier (low/medium/high) + Insertion Position (1-5)
   - Overhead: ~0.01-0.1ms
    ↓
3. Image Preprocessing (based on Knob1 tier)
   - Determine crop count from tier
   - Apply tiling and resize
    ↓
4. Vision Encoding
   - Vision Encoder: Process crops
   - Projector: Map to LLM space
    ↓
5. LLM Forward to Insertion Position
   - Run LLM blocks up to insertion position
   - Extract: Latency Token (last token after insertion position)
    ↓
6. Stage 2: Predict Knob2 & Knob3
   - Input: Latency Token (contains budget + vision + language interaction)
   - Predict: MoE Top-K (4/5/6/7/8) + Total Blocks (12/13/14/15/16)
   - Overhead: ~0.1ms
    ↓
7. Apply Knobs to Remaining LLM Blocks
   - Set top_k for blocks after insertion position (zero overhead, attribute modification)
   - Select blocks by importance (deterministic, O(n log n))
   - First block fixed: top_k=8, always included
    ↓
8. LLM Forward (with adaptive knobs)
   - Prefill: Generate with all knobs applied
   - Decode: Use prefill configuration (no controller re-run)
   - Generate output
```

### 2.3 训练时流程（Joint GRPO Training）

```
1. Controller predicts knob configuration
   - Stage 1: Predict Knob1 (tier + insertion position)
   - Process images with Knob1 tier
   - Vision encoding
   - Run LLM to insertion position, extract latency token
   - Stage 2: Predict Knob2 & Knob3 (based on latency token)
    ↓
2. Execute model (real execution, batch_size=1 per sample)
   - Use predicted knobs
   - Measure actual latency using hooks (prefill + decode)
   - Get accuracy from model output
    ↓
3. Compute reward
   - accuracy + latency constraints + budget violation penalty
    ↓
4. Update controller (Joint GRPO)
   - Both Stage1 and Stage2 contribute to same reward
   - End-to-end optimization
```

**关键设计**:
- **Direct Latency Measurement**: 使用PyTorch hooks直接测量latency（不使用estimator）
- **Batch Size**: 每个样本单独处理（batch_size=1 per sample）以确保准确测量
- **Budget Token**: 在prefill阶段编码为token并拼接到输入序列
- **Decode Phase**: 使用prefill阶段决定的配置，不重新运行controller

---

## 3. 架构设计

### 3.1 两阶段架构

```
┌─────────────────────────────────────────────────────────────┐
│              Two-Stage Controller Architecture              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1: Knob1 Prediction (BEFORE Vision Encoder)        │
│  ├─ Input: Language Feature + Budget Token (encoded)      │
│  ├─ Network: Lightweight MLP                               │
│  └─ Output: Tier (low/medium/high) + Insertion Position (1-5)│
│                                                              │
│  ↓ Image Preprocessing (based on Knob1 tier)              │
│  ↓ Vision Encoder + Projector                              │
│  ↓ LLM Forward to Insertion Position                       │
│  ↓ Extract Latency Token                                   │
│                                                              │
│  Stage 2: Knob2 & Knob3 Prediction (AFTER Insertion)       │
│  ├─ Input: Latency Token (from LLM)                        │
│  ├─ Network: Lightweight MLP                               │
│  └─ Output: Top-K (4/5/6/7/8) + Total Blocks (12-16)      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Stage 1: Knob1 Predictor

**设计选项（三个选择，复杂度递增）**：

#### 选项A: Budget-Only（最小Overhead，推荐用于SIGMETRICS）
- **Input**: Latency Budget only
- **Network**: Tiny MLP (~10K params)
- **Overhead**: ~0.01ms
- **优点**: 最小overhead，简单直接
- **缺点**: 无法利用prompt信息
- **状态**: ✅ 优先实现

#### 选项B: Budget + Language（中等复杂度）
- **Input**: Language Feature + Budget Feature
- **Network**: Lightweight MLP (~50K params) 或 Semantic Router
- **Overhead**: ~0.1ms
- **优点**: 可以利用prompt信息，更智能的决策
- **缺点**: 稍高的overhead
- **状态**: 🔄 调研Semantic Router集成
- **调研方向**: 参考 [Semantic Router](https://github.com/aurelio-labs/semantic-router) 进行快速语义路由决策

#### 选项C: Budget + Language + Vision（最高复杂度）
- **Input**: Vision Feature (global crop) + Language Feature + Budget Feature
- **Network**: MLP或Transformer
- **Overhead**: ~30-50ms（需要额外运行vision encoder）
- **优点**: 最准确，可以利用图像复杂度信息
- **缺点**: 需要多过一遍vision encoder，overhead大
- **状态**: ⚠️ 需要优化建议
- **优化建议**: 
  - 考虑使用轻量级vision encoder（如MobileViT）
  - 缓存global crop的vision feature
  - 使用知识蒸馏训练小模型

**当前代码实现**: 选项B（`controller.py`中的`Knob1PredictorBudgetLanguage`）

**关键改进**:
- 同时预测tier和insertion position（Stage2插入位置）
- Budget token编码为d_model维token，拼接到输入序列

### 3.3 Stage 2: Knob2 & Knob3 Predictor

**架构**: 独立轻量级MLP

```python
Latency Token (B, d_model)  # From LLM after insertion position
    ↓
Projection → (B, hidden_dim)
    ↓
Fusion MLP → (B, hidden_dim)
    ↓
Two Heads → (B, 5) each [top_k: 4,5,6,7,8] [blocks: 12,13,14,15,16]
```

**参数量**: ~10K-30K parameters（轻量级设计）

**关键设计**:
- **只使用Latency Token**: 已经包含budget、vision和language的交互信息
- **动态插入位置**: Stage1预测插入位置（1-5），Stage2在插入位置之后运行
- **动态Knob3选项**: 根据插入位置动态调整可选的block数量

---

## 4. Knob设计细节

### 4.1 Knob1: Vision Tokens Tier

**控制内容**: Vision tokens的数量（通过tier控制crop数量）

**决策时机**: 必须在vision encoder之前

**实现方式**: Stage 1 predictor

**输出**: 3个选择（low, medium, high）

**详细设计**: 参见`knob1_predictor_variants.md`

### 4.2 Knob2: MoE Top-K

**控制内容**: MoE层的expert数量

**决策时机**: 在vision encoder之后，LLM之前

**实现方式**: Stage 2 predictor

**输出**: 5个选择（4, 5, 6, 7, 8）

**关键约束**:
- 第一层固定top_k=8（总是包含）
- 只应用于插入位置之后的blocks

**应用方式**: 直接修改`block.mlp.top_k`属性（零overhead）

### 4.3 Knob3: Transformer Blocks

**控制内容**: 激活的transformer block数量

**决策时机**: 在vision encoder之后，LLM之前（两阶段）或与Knob1同时（一阶段）

**实现方式**: Importance-based pruning

**输出**: 5个选择（12, 13, 14, 15, 16 total blocks）

**关键设计**:
- **Total Blocks**: 值表示总block数（包括第一层和插入位置之前的blocks）
- **动态选项**: 根据插入位置动态调整可选范围
- **第一层固定**: 总是包含第一层（top_k=8）

**Importance Score理解**:
- **Data-Agnostic**: Importance score与数据来源无关（coco vqa和text vqa的importance score接近）
- **Task-Dependent**: Importance score与任务类型相关（science-qa与VQA任务的importance score差距较大）
- **应用策略**: 
  - 对于相同任务类型，可以使用统一的importance score
  - 对于不同任务类型，可能需要任务特定的importance score或动态选择

**关键设计**: 
- Controller只预测num_blocks（5个选择）
- Block selection基于预计算的importance score（确定性）
- 不需要学习mask（2^16种可能）
- 可以根据任务类型选择不同的importance score

**实现**:
```python
# Controller预测num_blocks
num_blocks = controller.predict_knob3(...)  # 8, 10, 12, 14, or 16

# 根据任务类型选择importance score（可选）
task_type = infer_task_type(language_prompt)  # VQA, ScienceQA, etc.
importance_scores = get_importance_scores(task_type)

# 基于importance score选择blocks（确定性）
selected_blocks = select_top_k_by_importance(importance_scores, num_blocks)

# 应用mask
apply_block_mask(model, selected_blocks)
```

**优势**:
- 输出空间：2^16 → 5（大幅简化）
- 训练简单：只需要学习5个选择
- 稳定可靠：基于预计算的importance
- 任务感知：可以根据任务类型选择不同的importance score

---

## 5. 输入特征设计

### 5.1 Stage 1输入

| 特征 | 提取方式 | 维度 | 说明 |
|------|---------|------|------|
| Language | Tokenizer + WTE + Mean pooling | (B, d_model) | 从prompt提取 |
| Budget | MLP encoder | (B, hidden_dim) | 从latency budget编码 |

**关键点**：
- ✅ 不需要vision feature（vision还没处理）
- ⚠️ **设计选项**: 可以只用Budget（选项A）或Budget+Language（选项B）

### 5.2 Stage 2输入

| 特征 | 提取方式 | 维度 | 说明 |
|------|---------|------|------|
| Latency Token | LLM after insertion position | (B, d_model) | **最后一个token**（包含budget+vision+language交互） |

**关键点**：
- ✅ **只使用Latency Token**: 已经包含所有必要信息（budget token + vision + language经过attention）
- ✅ **提取位置**: 在插入位置之后的block输出中提取最后一个token
- ✅ **信息完整性**: Latency token已经包含了budget、vision和language的交互信息

---

## 6. 训练方法

### 6.1 方法对比

| 方法 | Overhead | 训练时间 | 样本效率 | 准确性 | 推荐度 |
|------|---------|---------|---------|--------|--------|
| **Lookup Table** | 0 | 0 | N/A | 中等 | ⭐⭐⭐ Baseline |
| **Supervised** | 低 | 快 | 高 | 中等 | ⭐⭐⭐⭐ Baseline |
| **GRPO** | 低 | 中等 | **最高** | 高 | ⭐⭐⭐⭐⭐ **推荐** |

### 6.2 推荐方案：Joint Training（唯一训练方式）

**Joint GRPO Training**:
- **数据来源**: Online execution（实际数据集样本）
- **训练目标**: 学习accuracy-latency trade-off
- **优点**: 高效样本利用，可以学习复杂约束，端到端优化
- **关键特点**: 
  - Stage1和Stage2一起训练，共享reward信号
  - 使用direct latency measurement（hooks）
  - Batch size = 1 per sample（确保准确测量）

**训练流程**:
1. 从实际数据集加载样本（image + prompt）
2. 随机采样latency budget（170-380ms）
3. Stage1预测tier和insertion position
4. 运行vision encoder（基于tier）
5. 运行LLM到insertion position，提取latency token
6. Stage2预测top_k和num_blocks
7. 执行完整模型，测量实际latency（hooks）
8. 计算accuracy和reward
9. Joint GRPO loss更新两个controller

**关键设计**:
- **Direct Measurement**: 使用PyTorch hooks直接测量prefill和decode latency
- **Budget Token**: 编码为d_model维token，在prefill阶段拼接到输入序列
- **Decode Phase**: 使用prefill配置，不重新运行controller

### 6.3 Latency Estimator（独立模块，可选）

**注意**: Latency Estimator作为独立模块保留，但**当前controller训练不使用**。

**设计**: 轻量级MLP（2-3层）

**用途**: 
- 可以用于快速latency预估（不用于controller训练）
- 可以用于configuration搜索和优化
- 可以用于不同硬件的latency预测

**详细信息**: 参见`LATENCY_ESTIMATOR_DESIGN.md`（独立文档）

---

## 7. Overhead分析

### 7.1 Controller开销

**Stage 1 (Knob1)**:
- 参数量: ~10K-50K（取决于是否使用Language feature）
- Latency: ~0.01-0.1ms
- 占比: <0.01-0.1%

**Stage 2 (Knob2 & Knob3)**:
- 参数量: ~50K-200K
- Latency: ~0.1ms
- 占比: <0.1%

**Total Controller Overhead**:
- 参数量: ~60K-250K
- Latency: ~0.11-0.2ms
- **占比: <0.1% of total inference**

### 7.2 节省的计算

- 通过减少top_k: 节省10-30% MoE计算
- 通过跳过blocks: 节省10-25% Transformer计算
- **Net benefit**: 节省的计算 >> controller开销

### 7.3 零Overhead的Knob应用

**Top-K应用**: 直接修改属性（`block.mlp.top_k = new_value`）
- 零overhead
- 不影响计算图

**Block Mask应用**: 使用BlockMaskWrapper（pass-through for skipped blocks）
- 跳过blocks时只做identity pass-through
- Overhead可忽略

---

## 8. 实现细节

### 8.1 动态改变top_k

**可行性**: ✅ 完全可行，零overhead

**实现**:
```python
# 直接修改属性（最简单高效）
for i in range(4, 16):
    block = model.transformer.blocks[i]
    if hasattr(block, 'mlp') and hasattr(block.mlp, 'top_k'):
        block.mlp.top_k = new_top_k  # 直接修改，零overhead
```

**关键点**:
- `top_k`是普通Python属性，不在计算图中
- 可以直接修改，不影响计算图

### 8.2 Importance-Based Block Selection

**实现**: 参见`importance_based_block_selection.py`

**关键函数**:
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

### 8.3 代码结构

```
experiments/controller/
├── minimal_controller.py              # Minimal-overhead controller
├── two_stage_controller.py            # Two-stage controller (当前实现)
├── importance_based_block_selection.py  # Block selection工具
├── feature_extractors.py              # 特征提取
├── latency_estimator.py               # Latency estimator
└── ...
```

---

## 9. 性能指标

### 9.1 Overhead指标

- **Controller Latency**: <0.2ms
- **Controller Memory**: <1MB (parameters + activations)
- **Controller FLOPs**: <100K operations
- **Relative Overhead**: <0.1% of total inference

### 9.2 Effectiveness指标

- **Accuracy Retention**: >95% (compared to full model)
- **Latency Reduction**: 20-50% (depending on budget)
- **Budget Adherence**: >90% (within budget)
- **Pareto Efficiency**: 显著提升accuracy-latency Pareto frontier

### 9.3 Efficiency指标

- **Training Time**: <1 day (on 4 GPUs)
- **Sample Efficiency**: <10K samples for convergence
- **Inference Throughput**: >95% of baseline (minimal overhead)

---

## 10. 关键设计决策

### 10.1 两阶段架构

**决策**: 使用两阶段预测架构

**理由**:
- 符合系统执行流程约束
- Knob1必须在vision encoder之前决定
- Knob2 & Knob3可以在vision encoder之后决定

### 10.2 Knob3: Importance-Based Pruning

**决策**: 使用importance-based pruning，而不是mask预测

**理由**:
- 简化输出空间（2^16 → 5）
- 确定性选择，稳定可靠
- 数据无关（基于预计算的importance）

### 10.3 训练方法: Joint GRPO

**决策**: 使用Joint GRPO同时训练Stage1和Stage2

**理由**:
- Critic-free，训练快
- 高效样本利用
- 可以学习复杂的accuracy-latency trade-off
- 两个阶段共享reward，端到端优化
- 可以协调两个阶段的决策

### 10.4 AdaLoRA-Inspired设计（两种思路）

#### 思路1: 两阶段预测（当前实现）

**架构**:
```
Stage 1: Knob1 (Before Vision Encoder)
  - Input: Budget only 或 Budget + Language
  - Output: Vision Tokens Tier

↓ Vision Encoder + Projector

Stage 2: Knob2 & Knob3 (After Projector)
  - Input: Vision + Language + Budget tokens
  - Method: 借用LLM前3层做attention融合
  - Output: Top-K + Transformer Blocks (后13层)
```

**特点**:
- 符合系统执行流程
- 利用LLM前3层的表示能力
- 两阶段决策，清晰明确

**实现**: 当前`two_stage_controller.py`

#### 思路2: 一阶段预测（备选方案）

**架构**:
```
Single Stage: All Knobs (After Vision Encoder)
  - Input: Budget + Language + Vision (global crop)
  - Method: 融合所有特征，直接预测三个knob
  - Output: Tier + Top-K + Transformer Blocks
```

**特点**:
- 一次性决策，更简洁
- 需要额外的vision encoder pass（global crop）
- 可以利用完整的视觉信息

**实施计划**: 两种方案都保留，分别实现和对比

### 10.5 已确认决策

**Knob1设计选项**:
- ✅ 选项A: Budget-Only（最小overhead）- 优先实现
- ✅ 选项B: Budget + Language - 调研Semantic Router集成
- ✅ 选项C: Budget + Language + Vision - 需要优化建议

**Importance Score理解**:
- ✅ Data-Agnostic: 与数据来源无关
- ✅ Task-Dependent: 与任务类型相关
- ✅ 应用策略: 根据任务类型选择不同的importance score

---

## 附录

### A. 相关文档

- `ANALYSIS.md`: 技术分析文档
- `knob1_predictor_variants.md`: Knob1预测器的不同变体分析
- `controller_implementation_details.md`: 实现细节和可行性分析
- `IMPLEMENTATION_ROADMAP.md`: 实现路线图
- `SEMANTIC_ROUTER_INTEGRATION.md`: Semantic Router集成调研
- `ADALORA_DESIGNS.md`: AdaLoRA-Inspired设计的两种思路

### B. 代码实现

- `experiments/controller/controller.py`: Controller实现（Stage1和Stage2）
- `experiments/controller/joint_grpo_trainer.py`: Joint GRPO训练器
- `experiments/controller/train_joint_controller.py`: 主训练脚本
- `experiments/controller/importance_based_block_selection.py`: Block selection工具
- `experiments/controller/model_forward_with_dynamic_stage2.py`: 动态forward pass

### C. 外部资源

- **Semantic Router**: [https://github.com/aurelio-labs/semantic-router](https://github.com/aurelio-labs/semantic-router)
  - 可用于Knob1选项B的快速语义路由决策
  - 支持多模态路由
  - 超快速决策（<10ms）

### D. 设计历史

- **v1.0**: 原始设计（单阶段）
- **v2.0**: 两阶段架构（当前版本，思路1）
- **v2.1**: SIGMETRICS优化（最小overhead）
- **v2.2**: 添加一阶段方案（思路2）和Semantic Router调研

---

**文档维护**: 本文档整合了所有controller设计文档，基于现有代码实现和SIGMETRICS标准。如有不一致，以代码实现为准。

