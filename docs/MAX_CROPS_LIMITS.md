# max_crops 最大值限制说明

## 概述

`max_crops` 参数在代码中**没有硬编码的最大值限制**，但实际使用会受到以下几个因素的限制：

## 1. 序列长度限制（主要限制）

### 配置中的限制

从 `configs/model/config.json`：
- `max_sequence_length: 4096` - 总序列长度限制（包括vision tokens + text tokens）
- `max_position_embeddings: 32768` - 位置编码的最大值（通常不是限制因素）

### Vision Tokens 计算

根据代码分析，每个 crop 产生的 vision tokens 数量取决于：

```python
# 从 model_preprocessor.py
image_token_length_w = 12  # 默认值
image_token_length_h = 12  # 默认值
tokens_per_crop = image_token_length_w * image_token_length_h  # 12 × 12 = 144 tokens
```

**注意**：实际可能包含额外的特殊tokens（如 `image_start_token`, `image_end_token`, `image_col_token`），所以实际可能略多于144。

### 实际限制计算

假设每个 crop 产生约 **144-150 tokens**（包括特殊tokens）：

| max_crops | Vision Tokens (估计) | 剩余 Text Tokens | 是否可行 |
|-----------|---------------------|------------------|----------|
| 12        | ~1,728              | ~2,368           | ✅ 安全 |
| 20        | ~2,880              | ~1,216           | ⚠️ 紧张 |
| 25        | ~3,600              | ~496             | ⚠️ 很紧张 |
| 28        | ~4,032              | ~64              | ❌ 几乎不可行 |
| 30+       | >4,096              | <0               | ❌ 超出限制 |

**结论**：理论上 `max_crops` 可以达到 **25-28**，但实际建议不超过 **20-24**，以留出足够的空间给文本tokens。

## 2. 内存限制

### Vision Tokens 内存占用

- 每个 crop 的 vision features: `(batch_size, num_crops, 144, hidden_dim)`
- Attention 计算: `O(sequence_length²)`，其中 sequence_length 包括所有 vision tokens

### 估算

对于 `max_crops=12`：
- Vision tokens: ~1,728
- 如果 batch_size=64，内存需求已经很大

对于 `max_crops=20`：
- Vision tokens: ~2,880
- 需要显著降低 batch_size（可能到 16-32）

对于 `max_crops=28`：
- Vision tokens: ~4,032
- batch_size 可能需要降到 8-16

## 3. 计算复杂度限制

### Attention 计算复杂度

Attention 的计算复杂度是 `O(sequence_length²)`：

- `max_crops=12`: sequence_length ≈ 1,728 + text_tokens → attention 复杂度 ≈ 3M
- `max_crops=20`: sequence_length ≈ 2,880 + text_tokens → attention 复杂度 ≈ 8M
- `max_crops=28`: sequence_length ≈ 4,032 + text_tokens → attention 复杂度 ≈ 16M

更大的 `max_crops` 会导致：
- 更慢的推理速度
- 更高的内存占用
- 可能触发 CUDA kernel 限制（如 "invalid configuration argument" 错误）

## 4. 代码层面的限制

### select_tiling 函数

```python
def select_tiling(h, w, patch_size, max_num_crops):
    for i in range(1, max_num_crops + 1):
        for j in range(1, max_num_crops + 1):
            if i*j <= max_num_crops:
                # ...
```

**没有硬编码限制**，理论上可以接受任意大的值，但：
- 循环复杂度是 `O(max_num_crops²)`
- 对于非常大的值（如 >100），会变慢

### 实际使用中的限制

在实验脚本中，默认测试范围是 `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]`，但这只是**实验配置**，不是代码限制。

## 5. 推荐值

### 安全范围

- **推荐**: `max_crops ≤ 20`
- **可行但需谨慎**: `20 < max_crops ≤ 25`
- **不推荐**: `max_crops > 25`

### 根据用途选择

1. **标准使用**（如 VQA v2）:
   - `max_crops = 12`（默认值）
   - 已经足够覆盖大多数图像

2. **高分辨率图像**:
   - `max_crops = 15-18`
   - 需要降低 batch_size

3. **极高分辨率图像**:
   - `max_crops = 20-24`
   - 需要显著降低 batch_size（可能到 8-16）
   - 需要监控内存使用

4. **实验/研究**:
   - 可以尝试 `max_crops > 24`，但需要：
     - 非常小的 batch_size（1-4）
     - 监控序列长度是否超过 4096
     - 接受较慢的推理速度

## 6. 如何测试更大的 max_crops

如果你想测试更大的 `max_crops` 值：

```bash
# 测试 max_crops=20
python experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --max_crops_list 20 \
    --batch_size 16 \
    --auto_adjust_batch_size

# 测试 max_crops=25（需要更小的batch size）
python experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --max_crops_list 25 \
    --batch_size 8 \
    --auto_adjust_batch_size
```

**注意事项**：
1. 从较小的 batch_size 开始（如 8-16）
2. 启用 `--auto_adjust_batch_size` 自动调整
3. 监控内存使用和序列长度
4. 如果遇到 "invalid configuration argument" 错误，说明 batch_size 或 max_crops 太大

## 7. 检查实际限制的方法

### 方法1：检查序列长度

在代码中添加日志：

```python
input_len = batch["input_ids"].shape[1]
if input_len > 4000:
    log.warning(f"Sequence length {input_len} is very close to max_sequence_length=4096!")
```

### 方法2：监控内存

```python
import torch
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9  # GB
    reserved = torch.cuda.memory_reserved() / 1e9     # GB
    log.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

### 方法3：测试不同值

逐步增加 `max_crops`，观察：
- 是否 OOM
- 序列长度是否超过 4096
- 推理速度变化

## 总结

| 限制类型 | 实际限制 | 说明 |
|---------|---------|------|
| **硬编码限制** | ❌ 无 | 代码中没有硬编码的最大值 |
| **序列长度限制** | ~25-28 | 受 `max_sequence_length=4096` 限制 |
| **内存限制** | ~20-24 | 取决于 GPU 内存和 batch_size |
| **计算复杂度** | ~20-24 | Attention 复杂度 O(n²) |
| **推荐值** | **≤ 20** | 平衡性能和实用性 |

**结论**：虽然代码没有硬编码限制，但实际使用中建议 `max_crops ≤ 20`，最大不超过 25-28（需要非常小的 batch_size）。

