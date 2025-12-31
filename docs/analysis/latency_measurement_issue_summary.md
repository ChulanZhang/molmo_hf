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

### 1. Vision 被计算了多次

- **测量 `T_vision_total`**: 单独调用 `vision_backbone()` (第1次)
- **测量 `T_LLM_prefill`**: 调用 `model()`，内部调用 `vision_backbone()` (第2次)
- **测量 `T_total`**: 调用 `model.generate()`，内部调用 `model.forward()` -> `vision_backbone()` (第3次)

### 2. 测量环境不一致

| 测量阶段 | `empty_cache()` | GPU 缓存状态 | 测量时间 |
|---------|----------------|------------|---------|
| `T_vision_total` | ❌ 无 | 可能有缓存 | 较快 |
| `T_LLM_prefill` | ❌ 无 | 受益于缓存 | 更快 |
| `T_total` | ✅ **有** | 缓存被清空 | 较慢 |

### 3. `torch.cuda.empty_cache()` 的影响

**只在 `T_total` 测量前调用** (行 567)，导致：
- Vision/Pre fill 测量时：可能有 GPU 缓存，测量较快
- `T_total` 测量时：缓存被清空，内存分配更慢，测量较慢
- 结果：`T_vision_total + T_LLM_prefill` 可能 **大于** `T_total`，导致 `T_LLM_decode` 为负数

### 4. 为什么误差达到几十毫秒？

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

### 方案4：记录负值用于分析

不要简单地截断为 0，而是记录负值：

```python
calculated_decode = results["T_total"] - results.get("T_vision_total", 0.0) - results.get("T_LLM_prefill", 0.0)
if calculated_decode < 0:
    log.warning(f"Negative decode latency: {calculated_decode:.2f} ms")
    results["T_LLM_decode"] = 0.0
    results["T_LLM_decode_negative"] = calculated_decode  # 记录负值
else:
    results["T_LLM_decode"] = calculated_decode
```

## 🎯 建议

**优先实施方案1**：统一测量环境，在所有测量前都调用 `empty_cache()`，确保测量条件一致。

这样可以：
- 消除缓存效应导致的测量误差
- 确保所有测量在相同的 GPU 状态下进行
- 虽然可能整体测量时间稍长，但数据更准确

