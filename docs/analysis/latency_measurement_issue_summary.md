# Latency 测量问题总结

## 问题现象

22.85% 的样本出现 `T_LLM_decode = 0.0` 但 `output_tokens > 0`，平均测量误差达到 **21.73 ms**。

## 关键代码位置

### 文件：`experiments/base_experiment.py`

#### 1. Vision Encoder 测量 (行 388-399)
```python
# 单独测量 vision encoder
start = time.perf_counter()
_ = vision_backbone.encode_image(batch["images"])  # 🔑 关键调用
latencies_vit.append((time.perf_counter() - start) * 1000)
```

#### 2. Vision Total 测量 (行 401-413)
```python
# 单独测量 vision (ViT + Projector)
start = time.perf_counter()
_ = vision_backbone(batch["images"], batch.get("image_masks"))  # 🔑 关键调用
latencies_vision_total.append((time.perf_counter() - start) * 1000)
```

#### 3. LLM Prefill 测量 (行 426-500)
```python
# 使用 hooks 测量 prefill
# 注册 hooks 在 transformer blocks 上
start_hook_handle = transformer.blocks[0].register_forward_hook(start_hook)
end_hook_handle = transformer.blocks[-1].register_forward_hook(end_hook)

# 调用 model() - 这会触发 hooks，但内部也会调用 vision_backbone()！
_ = self.model(
    input_ids=batch["input_ids"],
    images=batch.get("images"),  # 🔑 这里会触发 vision_backbone()
    ...
)
```

**⚠️ 注意**：`model.forward()` 内部会调用 `vision_backbone()` (见 `molmo/models/modeling_molmoe.py:1935`)

#### 4. T_total 测量 (行 557-607)
```python
# 测量完整的 generate() 流程
if self.device.type == 'cuda':
    torch.cuda.empty_cache()  # ⚠️ 关键：清空缓存！

start = time.perf_counter()
output = self.model.generate(  # 🔑 关键调用
    input_ids=batch["input_ids"],
    images=batch.get("images"),  # 内部会再次调用 vision_backbone()
    ...
)
results["T_total"] = (time.perf_counter() - start) * 1000

# 计算 decode
results["T_LLM_decode"] = max(0.0, T_total - T_vision_total - T_LLM_prefill)
```

## 🔍 问题根源

### 1. 测量方法的问题

当前代码使用**减法方法**计算 decode latency：

```python
results["T_LLM_decode"] = max(0.0, results["T_total"] - results.get("T_vision_total", 0.0) - results.get("T_LLM_prefill", 0.0))
```

### 2. 分别测量 vs 整体测量

- `T_vision_total` 和 `T_LLM_prefill` 是**分别独立测量**的（在不同的 forward pass 中）
- `T_total` 是在 `model.generate()` 中测量的，这是一个**完整的流程**（vision + prefill + decode 一起）

### 3. Vision 被计算了多次

- **测量 `T_vision_total`**: 单独调用 `vision_backbone()` (第1次)
- **测量 `T_LLM_prefill`**: 调用 `model()`，内部调用 `vision_backbone()` (第2次)
- **测量 `T_total`**: 调用 `model.generate()`，内部调用 `model.forward()` -> `vision_backbone()` (第3次)

### 4. 测量环境不一致

| 测量阶段 | `empty_cache()` | GPU 缓存状态 | 测量时间 |
|---------|----------------|------------|---------|
| `T_vision_total` | ❌ 无 | 可能有缓存 | 较快 |
| `T_LLM_prefill` | ❌ 无 | 受益于缓存 | 更快 |
| `T_total` | ✅ **有** | 缓存被清空 | 较慢 |

### 5. 测量误差的来源

1. **缓存效应**：分别测量时，第二次测量可能受益于 GPU 缓存
2. **GPU 预热**：第一次测量可能包含 GPU 预热时间
3. **时序问题**：分别测量时，`torch.cuda.synchronize()` 的时机可能不同
4. **系统负载**：不同时间点的系统负载可能不同

### 6. `torch.cuda.empty_cache()` 的影响

**只在 `T_total` 测量前调用** (行 567)，导致：
- Vision/Pre fill 测量时：可能有 GPU 缓存，测量较快
- `T_total` 测量时：缓存被清空，内存分配更慢，测量较慢
- 结果：`T_vision_total + T_LLM_prefill` 可能 **大于** `T_total`，导致 `T_LLM_decode` 为负数

### 7. 为什么误差达到几十毫秒？

1. **GPU 缓存效应**:
   - 第一次测量 vision: 冷启动，~180ms
   - 第二次测量 (prefill 中的 vision): 受益于缓存，可能更快
   - 第三次测量 (`generate()` 中的 vision): `empty_cache()` 后，可能更慢

2. **内存分配模式**:
   - `empty_cache()` 后，内存分配需要更多时间
   - 分别测量时，内存可能已经分配好

3. **CUDA kernel 调度**:
   - 不同的内存状态可能导致不同的 kernel 调度策略
   - `empty_cache()` 可能触发不同的优化路径

### 8. 为什么 decode 时间短时更容易出现？

- 当 decode 时间很短（1-2 tokens，约 20-50ms）时
- 测量误差（约 20-30ms）可能超过实际的 decode 时间
- 导致 `T_vision_total + T_LLM_prefill > T_total`
- 从而 `T_LLM_decode` 计算为负数，被 `max(0.0, ...)` 截断为 0

## 📊 数据验证

从 `coco-2014-vqa_imgsizetier-high_crops12_topk8_blocks16.json` 分析：

- **受影响样本**: 457 / 2000 (22.85%)
- **平均误差**: `(T_vision + T_prefill) - T_total = 21.73 ms`
- **误差范围**: -53.13 ms 到 -0.04 ms

**示例 (Sample 0)**:
- `T_total = 335.32 ms` (整体测量，`empty_cache()` 后)
- `T_vision_total = 181.28 ms` (单独测量，可能有缓存)
- `T_LLM_prefill = 207.17 ms` (单独测量，受益于缓存)
- `Sum = 388.45 ms` (分别测量的和)
- `Difference = 53.13 ms` (测量误差)
- `Calculated decode = -53.13 ms` → 被截断为 `0.0 ms`

## 💡 解决方案

### 方案1：统一测量环境（推荐）

在所有测量前都调用 `empty_cache()`，或都不调用：

```python
# 在 measure_inference_latency() 开始处
if self.device.type == 'cuda':
    torch.cuda.empty_cache()  # 统一清空缓存

# 然后进行所有测量
# 1. Measure Vision
# 2. Measure Prefill  
# 3. Measure T_total
```

### 方案2：移除 `empty_cache()`

如果不需要清空缓存，可以移除：

```python
# 行 567: 移除或注释掉
# if self.device.type == 'cuda':
#     torch.cuda.empty_cache()
```

### 方案3：直接测量 decode 时间

使用 hooks 在 `model.generate()` 内部直接测量 decode 阶段，而不是用减法。

### 方案4：记录负值并警告（临时方案）

不要简单地截断为 0，而是记录负值并发出警告：

```python
calculated_decode = results["T_total"] - results.get("T_vision_total", 0.0) - results.get("T_LLM_prefill", 0.0)
if calculated_decode < 0:
    log.warning(f"Negative decode latency calculated: {calculated_decode:.2f} ms. "
                f"This indicates measurement error. Using 0.0.")
    results["T_LLM_decode"] = 0.0
    results["T_LLM_decode_negative"] = calculated_decode  # 记录负值用于分析
else:
    results["T_LLM_decode"] = calculated_decode
```

### 方案5：使用 T_decode_per_token 反推

如果 `output_tokens > 0` 但 `T_LLM_decode = 0`，可以使用 aggregate 的 `T_decode_per_token_mean` 来估算：

```python
if results["T_LLM_decode"] == 0.0 and output_tokens > 0:
    # 使用 aggregate 的 per-token latency 来估算
    decode_per_token = aggregate_stats.get("T_decode_per_token_mean", 0.0)
    if decode_per_token > 0:
        results["T_LLM_decode"] = decode_per_token * output_tokens
```

## 📉 影响

1. **Latency 统计不准确**：22.85% 的样本 decode latency 被错误地设为 0
2. **Per-token latency 计算错误**：`T_decode_per_token = T_LLM_decode / output_tokens` 会得到 0
3. **Latency 模型训练数据有噪声**：这些样本的 decode latency 标签是错误的

## 🎯 建议

**优先实施方案1**（统一测量环境）或**方案3**（直接测量 decode 时间）：

- **方案1**：统一测量环境，在所有测量前都调用 `empty_cache()`，确保测量条件一致
  - 消除缓存效应导致的测量误差
  - 确保所有测量在相同的 GPU 状态下进行
  - 虽然可能整体测量时间稍长，但数据更准确

- **方案3**：直接测量 decode 时间（最准确的解决方案）
  - 使用 hooks 在 `model.generate()` 内部直接测量 decode 阶段
  - 避免减法计算的累积误差
  - 如果暂时无法修改 `model.generate()`，可以先实施**方案4**（记录负值并警告），以便识别和分析问题

