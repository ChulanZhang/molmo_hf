# Exp 4: Language Tokens vs Latency - 深入分析

## 数据来源

**Exp 4 使用真实数据**：
- 从 VQA v2 validation set 中选取固定数量的真实图像（默认10张）
- 每张图像使用不同的 `max_new_tokens` 值进行生成
- JSON 结果中包含 `image_id`、`example_id`、`answers` 等真实数据字段

## Log Scale 下的线性关系分析

### 观察到的现象

在 log scale 下，随着 output tokens 的 2 倍增长（8, 16, 32, 64, 128, ...），decode latency 的图像基本是一条直线。

### 数学分析

#### 1. 线性关系假设

假设 decode latency 与 output tokens 呈**线性关系**：
```
T_decode = k × N_tokens + b
```

其中：
- `T_decode`: Decode latency (ms)
- `N_tokens`: Output tokens
- `k`: 每 token 的延迟（约 130 ms/token）
- `b`: 固定开销（约 0 ms，可忽略）

#### 2. Log Scale 变换

在 log scale 下，我们绘制的是：
```
log(T_decode) vs log(N_tokens)
```

将线性关系代入：
```
log(T_decode) = log(k × N_tokens + b)
```

当 `b ≈ 0` 且 `k × N_tokens >> b` 时：
```
log(T_decode) ≈ log(k) + log(N_tokens)
```

这是一个**斜率为 1** 的直线！

#### 3. 为什么看起来是直线？

当 `N_tokens` 以 2 的幂次增长时（8, 16, 32, 64, ...）：
- `log(N_tokens)` 是**等间距**的：`log(8), log(16), log(32), ...` = `3, 4, 5, ...` (以 2 为底)
- 如果 `T_decode` 也以 2 的幂次增长（即 `T_decode ∝ N_tokens`），那么 `log(T_decode)` 也是等间距的
- 因此 `log(T_decode) vs log(N_tokens)` 是一条**斜率为 1** 的直线

#### 4. 实际数据验证

从 Exp 4 的实际数据看：
- 8 tokens: ~1050 ms
- 16 tokens: ~2130 ms (≈ 2×)
- 32 tokens: ~4260 ms (≈ 2×)
- 64 tokens: ~8520 ms (≈ 2×)

这证实了：
1. **Decode latency 与 output tokens 基本呈线性关系**（每 token 约 130 ms）
2. 在 log scale 下，由于两者都按 2 的幂次增长，所以呈现为一条直线

### 结论

**Log scale 下的直线并不意味着 decode latency 与 output tokens 的 2 倍增长呈线性关系**，而是因为：

1. **Decode latency 与 output tokens 本身是线性关系**（`T_decode ∝ N_tokens`）
2. **在 log scale 下，线性关系表现为斜率为 1 的直线**
3. **当 tokens 以 2 的幂次增长时，log(tokens) 是等间距的，所以看起来是一条完美的直线**

### 物理意义

- **Decode 是顺序的**：每个 token 的生成时间大致相同（约 130 ms/token）
- **没有并行化**：无法像 prefill 那样并行处理多个 tokens
- **线性扩展**：增加 N 个 tokens 需要 N 倍的 decode 时间

## Prefill Latency 不可见的问题

### 问题分析

从数据看：
- Prefill (Vision + LLM): ~306 ms (73 ms vision + 233 ms LLM prefill)
- Decode (8 tokens): ~1050 ms
- Decode (4096 tokens): ~532,000 ms (532 秒)

**Prefill 与 Decode 的比例**：
- 8 tokens: 306 ms / 1050 ms ≈ 29%
- 4096 tokens: 306 ms / 532,000 ms ≈ 0.06%

在 log scale 下，当 decode latency 达到数百秒时，prefill 的 0.3 秒几乎不可见。

### 解决方案

1. **使用相对比例图**：显示 prefill 占总延迟的比例
2. **分离图表**：单独绘制 prefill 和 decode 的图表
3. **使用线性 scale 的局部放大**：对较小的 token 数量使用线性 scale

## 实验设计说明

**Exp 4 使用真实数据的原因**：
- 需要真实的图像-文本对来测试实际的生成延迟
- 不同图像可能产生不同长度的输出（虽然我们强制 `max_new_tokens`）
- 真实数据更能反映实际应用场景的性能

**与 Exp 3 的区别**：
- Exp 3 使用 dummy images（不同分辨率）来测试 vision tokens 的影响
- Exp 4 使用真实图像（固定分辨率）来测试 language tokens 的影响

