# 训练问题修复说明

## 1. Knob3 (Number of Blocks) 修复

### 问题
- 原来的选项是 `[12, 13, 14, 15, 16]`，但第一个block总是保留
- 剩下15个block，从这15个中选择11-15个，加上第一个block，总数是12-16
- 所以应该是 `[12, 13, 14, 15, 16]`

### 修复
- **Knob3选项改为**: `[12, 13, 14, 15, 16]`
  - 12 = 第一个block(1) + 11个选中的blocks
  - 13 = 第一个block(1) + 12个选中的blocks
  - ...
  - 16 = 第一个block(1) + 15个选中的blocks

### 基于Importance Score的选择
- 第一个block (index 0) **总是保留**
- 从剩余的15个blocks中，根据importance score选择top-N
- 使用 `results/layer_importance_scores_exp3_recommended.json` 中的scores

### 实现
```python
def _select_blocks_by_importance(self, num_active_blocks: int, total_blocks: int = 16) -> List[int]:
    # First block (index 0) is always kept
    if num_active_blocks <= 1:
        return [0]
    
    # Number of blocks to select from remaining 15 blocks
    num_to_select_from_remaining = num_active_blocks - 1
    
    if self.importance_scores is not None:
        # Select based on importance scores (excluding first block)
        remaining_scores = {k: v for k, v in self.importance_scores.items() if k != 0}
        sorted_blocks = sorted(remaining_scores.items(), key=lambda x: x[1], reverse=True)
        selected_remaining = [block_idx for block_idx, _ in sorted_blocks[:num_to_select_from_remaining]]
        selected_blocks = [0] + sorted(selected_remaining)
    else:
        # Fallback: use prefix blocks
        selected_blocks = list(range(num_active_blocks))
    
    return selected_blocks
```

---

## 2. Top-K 作用范围

### 问题
- Top-K应该只作用于第一个block之后的blocks
- 第一个block的top_k固定为8

### 修复
- `_set_top_k()` 方法已经正确实现：
  - Block 0: 固定 `top_k=8`
  - Blocks 1-15: 使用预测的 `top_k` 值

```python
def _set_top_k(self, top_k: int, start_layer: int = 1):
    for i, block in enumerate(blocks):
        if i == 0:
            block.mlp.top_k = 8  # Fixed
        elif i >= start_layer:
            block.mlp.top_k = top_k  # Predicted
```

---

## 3. Latency Estimator 使用细节

### 问题
Latency estimator在训练时和实际使用时的配置不一致：
- **训练时**: 所有blocks使用相同的top_k，所有blocks都参与
- **实际使用时**: 第一个block固定top_k=8，只有部分blocks参与

### 关键理解

**Latency Estimator的输入**：
- `top_k`: 这是**blocks after first**的top_k值（第一个block固定为8）
- `num_active_blocks`: **总block数**（包括第一个block）

**例如**：
- 如果 `num_active_blocks=11`, `top_k=6`:
  - 意味着：block 0 (top_k=8) + 10个blocks (top_k=6)
  - Estimator应该用 `top_k=6`, `num_active_blocks=11` 来预测

### 当前实现
```python
def _estimate_latency(self, ..., top_k: int, num_active_blocks: int, ...):
    # top_k: Top-K for blocks after first (first block fixed at 8)
    # num_active_blocks: Total blocks (includes first block)
    
    prefill_latency = self.latency_estimator(
        top_k=torch.tensor([top_k]),  # Top-K for blocks after first
        num_active_blocks=torch.tensor([num_active_blocks]),  # Total blocks
        ...
    )['T_prefill_total'].item()
```

### 注意事项
- Estimator是在profiling数据上训练的，这些数据中：
  - 第一个block的top_k总是8
  - `num_active_blocks` 包括第一个block
  - `top_k` 参数指的是blocks after first的top_k
- 所以当前实现是**正确的**

### 关于Output Tokens
- Estimator需要 `output_tokens` 来预测decode latency
- 当前实现使用**实际生成的tokens数**（从 `result['output_ids']` 中提取）
- 这是合理的，因为：
  - 训练时我们知道实际生成了多少tokens
  - 可以用这个来准确估计decode latency

---

## 4. Accuracy 计算修复

### 问题
- 不同数据集使用不同的evaluation metric
- 当前只支持VQA score，导致其他数据集accuracy为0

### 修复
支持多种metrics：
- **vqa_score**: TextVQA, OK-VQA, COCO-VQA
- **mc**: ScienceQA (multiple choice)
- **em**: TallyQA (exact match)
- **ansl_em**: ST-VQA, DocVQA (ANLS + EM)
- **cider_score**: COCO Caption (CIDEr score)

### 实现
```python
# Get metric from dataset name
from experiments.base_experiment import get_metric_for_dataset
metric_name = get_metric_for_dataset(dataset_name)

# Compute accuracy based on metric
if metric_name == "vqa_score":
    from molmo.eval.vqa import vqa_score
    score = vqa_score(answers, pred_text)
elif metric_name == "mc":
    # Multiple choice
    pred_idx = select_mc_option(pred_text, options)
    score = 1.0 if pred_idx == answer_idx else 0.0
elif metric_name == "em":
    # Exact match
    score = 1.0 if pred_normalized == answer_normalized else 0.0
# ... etc
```

---

## 5. Batch Execution 问题

### 问题
每个batch中的样本可能选择不同的模型配置（不同的tier, top_k, num_blocks），如何batch执行？

### 当前实现

**答案：当前是逐个样本执行，不是batch执行**

```python
for i in range(batch_size):
    # 每个样本单独执行（batch_size=1）
    result = self._execute_model(
        input_ids=batch['input_ids'][i:i+1],  # Single sample (batch_size=1)
        images=images[i:i+1],
        tier=tiers[i],  # Different for each sample
        top_k=int(top_k_values[i]),  # Different for each sample
        num_active_blocks=int(num_active_blocks_values[i]),  # Different for each sample
        ...
    )
```

### 为什么不能batch执行？

1. **不同的配置需要不同的模型设置**：
   - 不同的 `top_k` 需要设置不同blocks的 `mlp.top_k`（全局设置）
   - 不同的 `num_active_blocks` 需要mask不同的blocks（基于importance score选择）
   - 这些设置是**全局的**（影响整个模型），不能per-sample设置

2. **模型架构限制**：
   - PyTorch的模型forward是全局的
   - `model.generate()` 内部调用 `model.forward()`，而block配置是全局的
   - 不能在一个batch中让不同样本使用不同的block配置

3. **Block Masking的复杂性**：
   - 需要 `BlockMaskWrapper` 来跳过不活跃的blocks
   - 当前实现中，`_set_block_mask()` 计算了mask，但 `model.generate()` 仍然执行所有blocks
   - 这是因为 `model.generate()` 是一个复杂的函数，不容易插入block masking
   - **当前状态**: Top-K被正确设置，但block skipping（基于importance score）尚未完全实现

### 可能的优化方案

#### 方案1: 按配置分组（当前未实现）
```python
# Group samples by configuration
config_groups = {}
for i in range(batch_size):
    config = (tiers[i], top_k_values[i], num_active_blocks_values[i])
    if config not in config_groups:
        config_groups[config] = []
    config_groups[config].append(i)

# Execute each group as a batch
for config, indices in config_groups.items():
    tier, top_k, num_blocks = config
    # Set model config
    self._set_top_k(top_k, start_layer=1)
    self._set_block_mask(num_blocks)
    
    # Execute batch
    batch_input_ids = batch['input_ids'][indices]
    batch_images = images[indices] if images is not None else None
    results = self.model.generate(...)
```

**优点**：
- 减少forward pass次数
- 提高GPU利用率

**缺点**：
- 需要实现block masking（当前未实现）
- 代码复杂度增加

#### 方案2: 保持当前实现（推荐）
- 简单、清晰
- 每个样本独立，易于调试
- 虽然慢，但正确性有保障

### 性能影响

- **当前**: 每个样本单独执行，batch_size=64意味着64次forward pass
- **优化后**: 如果配置相同，可以batch执行，减少forward pass次数

**实际影响**：
- 如果batch中配置多样性高，优化收益有限
- 如果batch中配置相似，可以显著加速

---

## 总结

### 已修复
1. ✅ Knob3选项改为 `[12,13,14,15,16]`（第一个block总是保留，从剩余15个中选择11-15个）
2. ✅ 实现基于importance score的block选择
3. ✅ Top-K只作用于第一个block之后的blocks
4. ✅ Latency estimator正确使用（考虑第一个block固定top_k=8）
5. ✅ Accuracy计算支持多种数据集metrics

### 待优化
- [ ] Batch execution优化（按配置分组）
- [ ] Block masking实现（用于跳过不活跃的blocks）

### 关键理解

1. **第一个block总是保留，top_k固定为8**
2. **Knob3 = 总block数**（包括第一个）
3. **Top-K只作用于blocks after first**
4. **Latency estimator的输入**：
   - `top_k`: blocks after first的top_k
   - `num_active_blocks`: 总block数（包括第一个）
5. **Batch execution**: 当前是逐个样本执行，因为不同配置需要不同的模型设置

