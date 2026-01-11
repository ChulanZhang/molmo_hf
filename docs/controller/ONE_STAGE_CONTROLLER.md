# One-Stage Controller 实现文档

## 概述

One-Stage Controller 是一个统一的控制器架构，在模型 forward 之前一次性预测所有配置参数（tier、block mask、per-block top-k），而不是像 two-stage 那样分阶段预测。

## 架构设计

### 核心思想

- **独立特征提取**：在模型 forward 之前提取所有需要的特征（vision、language、budget）
- **一次性预测**：单个控制器同时预测所有 knobs
- **无插入点**：不再有 Stage2 插入位置的概念，所有配置在 forward 前确定
- **完整执行**：每个配置执行完整的模型 forward pass

### 控制器输出

`OneStageControllerPredictor` 输出三个部分：

1. **Tier Logits** `(B, 3)`: 预测 vision tokens tier (low/medium/high)
2. **Block Logits** `(B, 15)`: 预测 blocks 1-15 的激活分数（block0 总是激活）
3. **Block Top-K Logits** `(B, 16, 5)`: 预测每个 block 的 top-k 值（5 个选项：4,5,6,7,8）

## 特征提取

### 1. Vision Features

```python
# 使用 vision_backbone 提取全局 crop 的特征
image_features, cls_embed = model.model.vision_backbone(images, image_masks)
# 使用第一个 crop (global crop) 并 mean pooling patches
vision_feats = image_features[:, 0, :, :].mean(dim=1)  # (B, 768)
```

- **输入**: 图像 `(B, 3, 336, 336)`
- **输出**: Vision features `(B, 768)` - CLIP encoder 输出维度
- **特点**: 使用全局 crop，不需要 tier 信息（tier 是控制器的输出）

### 2. Language Features

```python
# 使用 tokenizer + WTE layer + mean pooling
lang_feat = lang_extractor.extract(prompt)  # (1, 2048)
```

- **输入**: 文本 prompt (字符串)
- **输出**: Language features `(B, 2048)` - d_model 维度
- **实现**: `LanguageFeatureExtractor` 使用模型的 WTE layer

### 3. Budget Features

```python
# 使用 LatencyBudgetEncoder 编码 latency budget
budget_feat = budget_encoder(latency_budget)  # (B, 2048)
```

- **输入**: Latency budget `(B,)` 标量值（单位：ms）
- **输出**: Budget features `(B, 2048)` - d_model 维度
- **实现**: 正弦编码 + MLP（参考 AdaLLaVA 设计）

## 采样逻辑

### 1. Tier 采样

```python
tier_probs = F.softmax(tier_logits, dim=-1)  # (B * group_size, 3)
tier_actions = torch.multinomial(tier_probs, 1).squeeze(-1)
tiers = [tier_options[idx.item()] for idx in tier_actions]  # ["low", "medium", "high"]
```

- **选项**: low (3 crops), medium (6 crops), high (12 crops)
- **训练**: 随机采样（multinomial）
- **验证**: 确定性采样（argmax）

### 2. Block Mask 采样

```python
block_masks, block_mask_log_probs = _sample_block_mask_from_logits(
    block_logits,  # (B * group_size, 15) for blocks 1-15
    min_active=12,  # 最少激活块数（包括 block0）
    max_active=16,  # 最多激活块数（包括 block0）
)
```

**约束条件**:
- Block0 总是激活（`mask[0] = True`）
- 从 blocks 1-15 中选择 `num_to_select` 个块（`min_active-1` 到 `max_active-1`）
- 总激活块数在 12-16 之间（包括 block0）

**采样过程**:
1. 随机选择要激活的块数：`num_to_select = randint(min_active-1, max_active-1)`
2. 根据 `block_logits` 采样 top-N 块（使用 multinomial）
3. 创建 mask：block0=True，选中的 blocks 1-15=True

### 3. Per-Block Top-K 采样

```python
for block_idx in range(16):
    if block_idx == 0:
        block_topk_dict[0] = 8  # Block0 固定为 8
    else:
        # Blocks 1-15: 从 logits 采样
        block_topk_logits_i = block_topk_logits[i, block_idx, :]  # (5,)
        topk_action = torch.multinomial(F.softmax(block_topk_logits_i), 1).item()
        block_topk_dict[block_idx] = topk_choices[topk_action]  # 4,5,6,7,8
```

- **Block0**: 固定 top-k = 8（不采样）
- **Blocks 1-15**: 从 5 个选项（4,5,6,7,8）中采样
- **训练**: 随机采样（multinomial）
- **验证**: 确定性采样（argmax）

## 模型执行

### 配置应用

```python
# 1. 设置 per-block top-k
_set_per_block_top_k(per_block_top_k)  # {block_idx: top_k_value}

# 2. 应用 block mask
_apply_block_mask(block_mask)  # (16,) boolean mask

# 3. 执行模型
result = model.generate(
    input_ids=input_ids,
    images=images,  # 已根据 tier 处理
    latency_budget=latency_budget,
    budget_encoder=budget_encoder,
    ...
)
```

### Block Mask 应用

**注意**: 当前的 `_apply_block_mask` 方法只是存储 mask，实际的 block skipping 需要在模型的 forward 方法中实现。目前所有 blocks 都会执行，但可以通过设置 top_k=0 或其他方式来实现 skipping。

**未来改进**: 可以在模型的 forward 方法中检查 `model._active_block_mask` 并跳过未激活的 blocks。

## 训练流程

### GRPO 实现

1. **特征提取**（一次）:
   - 提取 vision_feat, lang_feat, budget_feat
   - 对每个原始样本重复 `group_size` 次

2. **控制器预测**（一次）:
   - 输入: `(B * group_size, vision_dim)`, `(B * group_size, lang_dim)`, `(B * group_size, budget_dim)`
   - 输出: tier_logits, block_logits, block_topk_logits

3. **采样配置**（一次）:
   - 采样 `group_size` 个不同的配置（tier, block_mask, per_block_topk）
   - 总共 `B * group_size` 个配置

4. **模型执行**（B * group_size 次）:
   - 每个配置执行一次完整的模型 forward pass
   - 测量 latency（使用 hooks）
   - 计算 accuracy

5. **GRPO Loss**:
   - 按 `(sample_id, latency_budget)` 分组
   - 计算 group-normalized advantages
   - 使用 policy gradient: `loss = -log_prob * advantage`

### Log Probability 计算

```python
# 1. Tier log prob
tier_log_probs = F.log_softmax(tier_logits, dim=-1)
tier_log_prob = tier_log_probs.gather(1, tier_actions.unsqueeze(-1)).squeeze(-1)

# 2. Block mask log prob (在 _sample_block_mask_from_logits 中计算)
block_mask_log_prob = block_mask_log_probs

# 3. Per-block top-k log prob
per_block_topk_log_prob = sum of log_probs for all blocks (block0 = 0.0)

# Total log prob
total_log_probs = tier_log_prob + block_mask_log_prob + per_block_topk_log_prob
```

## 与 Two-Stage 的区别

| 特性 | Two-Stage | One-Stage |
|------|-----------|-----------|
| **控制器数量** | 2 个（Stage1 + Stage2） | 1 个 |
| **插入点** | Stage1 预测插入位置（1-5） | 无插入点 |
| **特征提取** | Stage1: lang+budget; Stage2: latency_token | 独立提取 vision+lang+budget |
| **Forward 次数** | 5 次（优化模式）或 10 次（标准模式） | 5 次（每个配置一次） |
| **Block 选择** | 基于插入位置动态调整 | 固定范围（12-16 blocks） |
| **Top-K** | 所有 blocks 使用相同 top-k | 每个 block 独立 top-k |

## 关键参数

### Controller 参数

- `vision_dim=768`: CLIP vision encoder 输出维度
- `lang_dim=2048`: Language features 维度（d_model）
- `budget_dim=2048`: Budget features 维度（d_model）
- `hidden_dim=256`: Controller 内部隐藏层维度
- `total_blocks=16`: Transformer blocks 总数

### 训练参数

- `min_active_blocks=12`: 最少激活块数（包括 block0）
- `max_active_blocks=16`: 最多激活块数（包括 block0）
- `group_size=5`: GRPO 组大小（每个样本采样 5 个配置）
- `topk_choices=[4,5,6,7,8]`: Top-K 选项

### 约束条件

1. **Block0 总是激活**: `block_mask[0] = True`
2. **Block0 top-k 固定**: `per_block_topk[0] = 8`
3. **总激活块数**: 12 ≤ num_active_blocks ≤ 16
4. **Block mask 长度**: 固定为 16（对应 16 个 blocks）

## 文件结构

### 核心文件

1. **`experiments/controller/controller.py`**:
   - `OneStageControllerPredictor`: 控制器模型定义

2. **`experiments/controller/joint_grpo_trainer.py`**:
   - `JointGRPOTrainer`: 训练器（已更新为 one-stage）
   - `_sample_block_mask_from_logits`: Block mask 采样逻辑
   - `_set_per_block_top_k`: Per-block top-k 设置
   - `_apply_block_mask`: Block mask 应用

3. **`experiments/controller/train_joint_controller.py`**:
   - 训练脚本入口
   - 数据集加载和训练循环

4. **`experiments/controller/feature_extractors.py`**:
   - `LanguageFeatureExtractor`: 语言特征提取
   - `LatencyBudgetEncoder`: 预算编码器

## 使用示例

### 初始化控制器

```python
from experiments.controller.controller import OneStageControllerPredictor

controller = OneStageControllerPredictor(
    vision_dim=768,
    lang_dim=2048,
    budget_dim=2048,
    hidden_dim=256,
    dropout=0.1,
    total_blocks=16,
).to(device)
```

### 训练

```python
python experiments/controller/train_joint_controller.py \
    --model_path checkpoints/molmo \
    --output_dir checkpoints/one_stage_controller \
    --dataset_names text_vqa \
    --batch_size 1 \
    --num_epochs 100 \
    --lr 1e-4 \
    --group_size 5 \
    --importance_scores_file results/layer_importance_scores_exp3_recommended.json
```

## 未来改进方向

1. **Block Skipping 实现**: 在模型 forward 中实际跳过未激活的 blocks（目前只是存储 mask）
2. **更灵活的 Block 选择**: 支持更复杂的 block 选择策略（如基于 importance scores）
3. **动态 Top-K**: 根据 block 的重要性动态调整 top-k 范围
4. **多任务支持**: 针对不同任务类型使用不同的控制器配置

## 注意事项

1. **Block Mask 应用**: 当前 `_apply_block_mask` 只是存储 mask，实际的 skipping 需要在模型 forward 中实现
2. **Vision Features**: 使用全局 crop，不考虑 tier（tier 是控制器的输出，用于后续的图像处理）
3. **Batch Size**: 训练时 batch_size=1 per sample，以确保准确的 latency 测量
4. **GRPO Grouping**: 按 `(sample_id, latency_budget)` 分组，确保相同样本和预算的配置在同一组

