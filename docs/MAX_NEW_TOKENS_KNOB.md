# `max_new_tokens` Knob 详解

## 概述

`max_new_tokens` 是控制文本生成时**最大生成token数量**的关键参数。本文档深入解释其在 Molmo 模型中的行为、默认值、提前停止机制，以及对 VQA 任务的影响。

## 1. 参数定义与默认值

### 1.1 参数定义

`max_new_tokens` 指定模型在生成文本时**最多生成多少个新token**（不包括输入prompt的tokens）。

### 1.2 默认值

**Molmo 模型没有内置默认值**，必须显式指定。在代码中：

```python
# modeling_molmoe.py:2449
max_new_tokens = generation_config.max_new_tokens
assert max_new_tokens is not None  # 必须提供，否则报错
```

**Transformers 库的默认行为**：
- 如果未指定 `max_new_tokens`，Transformers 的 `PreTrainedModel.generate()` 会使用 `max_length` 参数
- 默认 `max_length=20`（如果两者都未指定）
- 但 Molmo 的实现要求必须通过 `GenerationConfig` 显式指定 `max_new_tokens`

### 1.3 在代码中的使用

```python
from transformers import GenerationConfig

generation_config = GenerationConfig(
    max_new_tokens=128,  # 必须显式指定
    do_sample=False,
    use_cache=True,  # Molmo 模型要求 use_cache=True
)
outputs = model.generate(
    input_ids=input_ids,
    generation_config=generation_config,
)
```

## 2. 生成行为详解

### 2.1 是否强制生成 `max_new_tokens` 个token？

**答案：否。生成可能会提前停止。**

生成过程会在以下情况提前停止：

1. **EOS Token 生成**：如果模型生成了 EOS (End-of-Sequence) token，生成会立即停止
2. **Stop Strings**：如果指定了 `stop_strings`，遇到这些字符串时会停止
3. **其他停止条件**：如 `max_length` 限制等

### 2.2 提前停止机制

#### Transformers 库的默认行为

`PreTrainedModel.generate()` 方法（Molmo 通过 `super().generate()` 调用）会：

1. **检查 EOS Token**：
   - 默认使用 `tokenizer.eos_token_id`
   - 每次生成新token后，检查是否为 EOS token
   - 如果是，立即停止该序列的生成

2. **批量处理**：
   - 不同序列可能在不同时间停止
   - 已停止的序列不再生成新token
   - 所有序列都停止或达到 `max_new_tokens` 后，返回结果

3. **输出长度**：
   - 实际生成的token数 ≤ `max_new_tokens`
   - 输出形状：`(batch_size, input_length + actual_generated_tokens)`

#### 代码实现位置

```python
# modeling_molmoe.py:2468-2479
out = super().generate(  # 调用 Transformers 的 generate
    input_ids,
    generation_config,
    attention_mask=attention_mask,
    images=images,
    image_masks=image_masks,
    image_input_idx=image_input_idx,
    position_ids=position_ids,
    append_last_valid_logits=append_last_valid_logits,
    **kwargs,
)
```

Transformers 库内部会：
- 在生成循环中检查 `eos_token_id`
- 使用 `StoppingCriteria` 类处理停止条件
- 支持自定义停止条件

### 2.3 如果回答较短，会截断吗？

**答案：不会截断。** 生成会在以下情况自然停止：

1. **EOS Token**：模型生成 EOS token 时停止
2. **自然结束**：模型认为回答已完成（通过 EOS token 表示）
3. **达到上限**：如果未生成 EOS token，会生成到 `max_new_tokens` 个token

**重要**：生成的输出**包含所有生成的tokens**，不会因为回答短而截断。如果回答只有 5 个token，输出就是 5 个token（加上 EOS token，如果生成的话）。

## 3. VQA 任务的特殊考虑

### 3.1 VQA 回答的特点

VQA (Visual Question Answering) 任务的回答通常很短：
- 大多数回答是 1-5 个单词
- 例如："red", "two", "yes", "a dog"
- 对应的token数：通常 1-10 个tokens

### 3.2 推荐的 `max_new_tokens` 设置

对于 VQA 任务，建议设置：

```python
max_new_tokens = 128  # 足够长，但不会浪费计算
```

**理由**：
- VQA 回答通常很短（1-10 tokens）
- 设置 128 可以覆盖绝大多数情况
- 即使回答很短，模型会在生成 EOS token 时提前停止
- 不会因为设置太大而浪费计算（因为会提前停止）

### 3.3 实际生成长度分布

在 VQA v2 validation set 上的典型分布：
- 1-5 tokens: ~60% 的回答
- 6-10 tokens: ~30% 的回答
- 11-20 tokens: ~8% 的回答
- 21+ tokens: ~2% 的回答

因此，`max_new_tokens=128` 是安全的设置，既能覆盖所有情况，又不会因为设置太小而截断长回答。

## 4. 代码中的处理

### 4.1 提取生成的tokens

在 accuracy 计算脚本中：

```python
# exp1_accuracy.py:786-790
input_len = input_ids.shape[1]
if predictions.shape[1] > input_len:
    generated_tokens = predictions[:, input_len:]  # 只取新生成的tokens
else:
    generated_tokens = predictions  # 如果已经是生成的tokens
```

**关键点**：
- `predictions` 包含输入 + 生成的tokens
- 需要从 `input_len` 位置开始提取
- 实际长度可能 < `max_new_tokens`（如果提前停止）

### 4.2 处理答案提取

```python
# exp1_accuracy.py:805-813
pred_text = pred_texts[i]
if "Answer:" in pred_text:
    pred_text = pred_text.split("Answer:")[1].strip()
elif "\n" in pred_text:
    lines = [line.strip() for line in pred_text.split("\n") if line.strip()]
    pred_text = lines[-1] if lines else pred_text.strip()
else:
    pred_text = " ".join(pred_text.strip().split())
```

**处理逻辑**：
- 提取 "Answer:" 后的内容
- 如果有多行，取最后一行（通常是实际答案）
- 去除多余空格

## 5. 性能影响

### 5.1 计算成本

生成时间主要取决于：
1. **实际生成的token数**（不是 `max_new_tokens`）
2. **Prefill 时间**（处理输入，固定）
3. **Decode 时间**（每个新token，线性增长）

**示例**：
- `max_new_tokens=128`，实际生成了 5 个tokens
- 计算时间 ≈ Prefill + 5 × Decode_per_token
- **不会**计算 128 个tokens（因为提前停止）

### 5.2 内存使用

内存分配基于 `max_new_tokens`：

```python
# modeling_molmoe.py:2451
mask_len = seq_len + max_new_tokens if self.config.use_position_ids else seq_len
attention_mask = torch.cat(
    [attention_mask, attention_mask.new_ones((batch_size, max_new_tokens))],
    dim=1,
)
```

**注意**：即使提前停止，attention mask 也会预分配 `max_new_tokens` 的空间。但这是必要的，因为生成过程中无法预知何时停止。

## 6. 最佳实践

### 6.1 对于 VQA 任务

```python
generation_config = GenerationConfig(
    max_new_tokens=128,      # 足够覆盖所有VQA回答
    do_sample=False,         # 确定性生成（greedy decoding）
    use_cache=True,          # 必须启用（Molmo要求）
    # 可选：指定 EOS token（如果tokenizer有设置）
    # eos_token_id=tokenizer.eos_token_id,
)
```

### 6.2 对于其他任务

- **对话任务**：可能需要 `max_new_tokens=512` 或更大
- **摘要任务**：根据目标长度设置
- **代码生成**：可能需要 `max_new_tokens=1024` 或更大

### 6.3 调试建议

如果怀疑生成被截断：

```python
# 检查实际生成了多少个tokens
input_len = input_ids.shape[1]
output_len = outputs.shape[1]
actual_generated = output_len - input_len

print(f"Requested: {max_new_tokens}, Actual: {actual_generated}")

# 检查是否包含 EOS token
if tokenizer.eos_token_id is not None:
    eos_positions = (outputs == tokenizer.eos_token_id).nonzero()
    print(f"EOS positions: {eos_positions}")
```

## 7. 常见问题

### Q1: 如果设置 `max_new_tokens=10`，但回答需要 15 个tokens，会怎样？

**A**: 生成会在 10 个tokens 后停止，回答会被截断。建议设置足够大的值（如 128），让模型自然停止。

### Q2: 模型会强制生成 `max_new_tokens` 个tokens吗？

**A**: 不会。如果模型生成 EOS token，会立即停止。只有未生成 EOS token 时，才会生成到 `max_new_tokens`。

### Q3: 如何知道生成了多少个tokens？

**A**: 
```python
actual_tokens = outputs.shape[1] - input_ids.shape[1]
```

### Q4: 可以设置 `max_new_tokens=None` 吗？

**A**: 不可以。Molmo 的实现要求必须显式指定 `max_new_tokens`，否则会报错。

### Q5: 提前停止会影响accuracy计算吗？

**A**: 不会。Accuracy 计算基于实际生成的文本，无论长度如何。VQA 评估会自动处理不同长度的回答。

## 8. 总结

- **`max_new_tokens` 是上限，不是目标**：生成会在 EOS token 或达到上限时停止
- **VQA 任务推荐 `max_new_tokens=128`**：足够覆盖所有回答，不会浪费计算
- **实际生成长度 ≤ `max_new_tokens`**：提前停止是正常行为
- **不会截断短回答**：如果回答短，生成会自然停止，不会强制生成到上限
- **必须显式指定**：Molmo 模型要求通过 `GenerationConfig` 显式设置

## 9. 相关代码位置

- **模型生成方法**：`molmo/models/modeling_molmoe.py:2428-2481`
- **Accuracy 计算**：`experiments/base_experiment.py:760-855`
- **Token 提取**：`experiments/profiling/knob1_tokens/exp1_accuracy.py:786-813`
- **Transformers 生成**：通过 `super().generate()` 调用 Transformers 库的实现

