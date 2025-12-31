# Latency 测量代码位置详解

## 文件位置
`experiments/base_experiment.py` - `BaseExperiment.measure_inference_latency()` 方法

## 测量流程

### 1. Vision Encoder 测量 (行 388-399)

```python
# 位置: experiments/base_experiment.py:388-399
latencies_vit = []
for _ in range(num_runs):
    if self.device.type == 'cuda':
        torch.cuda.synchronize(self.device)  # ⏱️ 同步点 1
    start = time.perf_counter()              # ⏱️ 开始计时
    with torch.inference_mode():
        _ = vision_backbone.encode_image(batch["images"])  # 🔑 关键调用
    if self.device.type == 'cuda':
        torch.cuda.synchronize(self.device)  # ⏱️ 同步点 2
    latencies_vit.append((time.perf_counter() - start) * 1000)  # ⏱️ 结束计时
results["T_vision_encoder"] = np.mean(latencies_vit)
```

**测量内容**: 只测量 ViT encoder，不包括 projector

---

### 2. Vision Total (ViT + Projector) 测量 (行 401-413)

```python
# 位置: experiments/base_experiment.py:401-413
latencies_vision_total = []
for _ in range(num_runs):
    if self.device.type == 'cuda':
        torch.cuda.synchronize(self.device)  # ⏱️ 同步点 1
    start = time.perf_counter()              # ⏱️ 开始计时
    with torch.inference_mode():
        _ = vision_backbone(batch["images"], batch.get("image_masks"))  # 🔑 关键调用
    if self.device.type == 'cuda':
        torch.cuda.synchronize(self.device)  # ⏱️ 同步点 2
    latencies_vision_total.append((time.perf_counter() - start) * 1000)  # ⏱️ 结束计时
results["T_vision_total"] = np.mean(latencies_vision_total)
```

**测量内容**: ViT + Projector 的完整 forward pass

**Projector 时间**: 通过减法计算 (行 416)
```python
results["T_projector"] = max(0.0, results["T_vision_total"] - results["T_vision_encoder"])
```

---

### 3. LLM Prefill 测量 (行 426-555)

#### 方法1: 使用 Hooks (行 434-500) - **当前使用的方法**

```python
# 位置: experiments/base_experiment.py:434-500

# Hook 定义
def start_hook(module, input, output):
    nonlocal llm_start_time
    if self.device.type == 'cuda':
        torch.cuda.synchronize(self.device)  # ⏱️ 同步点
    llm_start_time = time.perf_counter()     # ⏱️ 开始计时

def end_hook(module, input, output):
    nonlocal llm_start_time
    if llm_start_time is not None:
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)  # ⏱️ 同步点
        end_time = time.perf_counter()
        llm_prefill_times.append((end_time - llm_start_time) * 1000)  # ⏱️ 结束计时
        llm_start_time = None

# 注册 hooks
start_hook_handle = transformer.blocks[0].register_forward_hook(start_hook)   # 🔑 第一个 block
end_hook_handle = transformer.blocks[-1].register_forward_hook(end_hook)      # 🔑 最后一个 block

# 测量
for _ in range(num_runs):
    if self.device.type == 'cuda':
        torch.cuda.synchronize(self.device)  # ⏱️ 同步点
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            _ = self.model(  # 🔑 关键调用 - 这会触发 hooks
                input_ids=batch["input_ids"],
                images=batch.get("images"),
                image_masks=batch.get("image_masks"),
                image_input_idx=batch.get("image_input_idx"),
                attention_mask=batch.get("attention_mask"),
                attention_bias=batch.get("attention_bias"),
                position_ids=batch.get("position_ids"),
            )
    if self.device.type == 'cuda':
        torch.cuda.synchronize(self.device)  # ⏱️ 同步点

results["T_LLM_prefill"] = np.mean(llm_prefill_times)
```

**测量内容**: 从第一个 transformer block 到最后一个 block 的时间（只包括 LLM prefill，不包括 vision）

**⚠️ 问题**: 这个测量是在**单独的 forward pass**中进行的，不包括 vision encoder！

#### 方法2: Fallback 减法方法 (行 502-522)

如果 hooks 失败，使用减法：
```python
# 测量整个 prefill step (Vision + Prefill)
T_prefill_step = ...  # 测量 model() 的完整时间
results["T_LLM_prefill"] = max(0.0, T_prefill_step - results.get("T_vision_total", 0.0))
```

---

### 4. T_total (包含 Decode) 测量 (行 557-607)

```python
# 位置: experiments/base_experiment.py:557-607
if max_new_tokens > 0:
    if self.device.type == 'cuda':
        torch.cuda.synchronize(self.device)  # ⏱️ 同步点 1
    start = time.perf_counter()              # ⏱️ 开始计时
    
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()     # ⚠️ 清空缓存！
            
            # ... 准备 generation_config ...
            
            output = self.model.generate(    # 🔑 关键调用 - 包含 Vision + Prefill + Decode
                input_ids=batch["input_ids"],
                images=batch.get("images"),
                image_masks=batch.get("image_masks"),
                image_input_idx=batch.get("image_input_idx"),
                generation_config=generation_config,
            )
    
    if self.device.type == 'cuda':
        torch.cuda.synchronize(self.device)  # ⏱️ 同步点 2
    results["T_total"] = (time.perf_counter() - start) * 1000  # ⏱️ 结束计时
    
    # 计算 Decode Latency
    results["T_LLM_decode"] = max(0.0, results["T_total"] - results.get("T_vision_total", 0.0) - results.get("T_LLM_prefill", 0.0))
```

**测量内容**: `model.generate()` 的完整时间，包括：
- Vision encoder
- Projector
- LLM prefill
- LLM decode (所有生成的 tokens)

---

## ⚠️ 问题分析

### 测量不一致导致的误差

1. **Vision 和 Prefill 是分别测量的**:
   - `T_vision_total`: 单独调用 `vision_backbone()` 测量
   - `T_LLM_prefill`: 单独调用 `model()` 测量（使用 hooks，只测量 transformer blocks）

2. **T_total 是整体测量的**:
   - `T_total`: 调用 `model.generate()`，包含完整的 pipeline

3. **关键差异**:
   - **分别测量时**: 每次测量都是独立的 forward pass，可能有：
     - GPU 缓存效应（第二次测量可能更快）
     - 不同的内存分配模式
     - 不同的 CUDA kernel 调度
   - **整体测量时**: `model.generate()` 内部会：
     - 复用 vision 的输出
     - 使用 KV cache
     - 可能有不同的优化路径

4. **`torch.cuda.empty_cache()` 的影响** (行 567):
   - 在 `T_total` 测量前清空缓存
   - 这可能导致 `T_total` 测量时内存分配更慢
   - 但分别测量 vision/prefill 时没有清空缓存

### 为什么误差达到几十毫秒？

1. **GPU 缓存效应**: 
   - 分别测量 vision 时，GPU 可能已经缓存了某些中间结果
   - 整体测量时，这些缓存可能不存在或不同

2. **内存分配模式**:
   - 分别测量时，每次都是新的内存分配
   - 整体测量时，`generate()` 内部可能复用内存

3. **CUDA kernel 调度**:
   - 分别测量时，kernel 调度可能不同
   - 整体测量时，可能有更优化的调度

4. **`torch.cuda.empty_cache()`**:
   - 在 `T_total` 测量前清空缓存，可能导致更慢的内存分配
   - 但 vision/prefill 测量时没有这个操作

## 🔍 关键发现

### `model.generate()` 内部会重新计算 Vision！

查看 `molmo/models/modeling_molmoe.py`:

1. **`model.generate()`** (行 2475-2528):
   - 调用 `super().generate()` (transformers 库的 generate)
   - 传入 `images` 参数

2. **`model.forward()`** (行 1829-1946):
   - 如果 `images is not None`，会调用 `self.vision_backbone(images, image_masks)` (行 1935)
   - 这意味着 **每次 `model.forward()` 都会重新计算 vision**

3. **`model.generate()` 的工作流程**:
   ```
   model.generate()
     └─> transformers.generate() (内部循环)
         └─> model.forward() (prefill step)
             └─> vision_backbone()  ← 🔑 Vision 被重新计算！
         └─> model.forward() (decode step 1)
         └─> model.forward() (decode step 2)
         └─> ...
   ```

### ⚠️ 问题根源

**Vision 被计算了多次，且每次时间不同！**

1. **测量 `T_vision_total`** (行 401-413):
   - 单独调用 `vision_backbone()` 
   - 第一次计算，可能较慢（GPU 预热、内存分配）

2. **测量 `T_LLM_prefill`** (行 426-500):
   - 调用 `model()`，内部会调用 `vision_backbone()` (行 1935)
   - 第二次计算，可能受益于 GPU 缓存，**更快**

3. **测量 `T_total`** (行 557-607):
   - 调用 `model.generate()`，内部会再次调用 `vision_backbone()`
   - 第三次计算，时间又可能不同
   - **而且 `torch.cuda.empty_cache()` 在测量前被调用** (行 567)，可能导致更慢

### 为什么误差达到几十毫秒？

1. **GPU 缓存效应**:
   - 第一次测量 vision: 冷启动，较慢
   - 第二次测量 (prefill): 受益于缓存，较快
   - 第三次测量 (generate): `empty_cache()` 后，又变慢

2. **内存分配模式**:
   - 分别测量时，每次都是新的内存分配
   - `generate()` 内部可能有内存复用

3. **`torch.cuda.empty_cache()` 的影响** (行 567):
   - **只在 `T_total` 测量前调用**
   - 清空缓存后，内存分配更慢
   - 但 vision/prefill 测量时没有这个操作

4. **测量时机不同**:
   - Vision 和 Prefill 是连续测量的（可能有缓存）
   - `T_total` 是在 `empty_cache()` 后测量的（可能更慢）

## 🔍 验证建议

1. **检查 `torch.cuda.empty_cache()` 的影响**:
   - 在 vision/prefill 测量前也清空缓存，看是否一致
   - 或者移除 `empty_cache()`，看误差是否减小

2. **检查 hooks 在 `model.generate()` 中是否触发**:
   - 确认 hooks 在 `model.generate()` 的 prefill step 中也会触发
   - 如果不会，那么 `T_LLM_prefill` 的测量方法需要调整

3. **统一测量环境**:
   - 所有测量前都调用 `empty_cache()`，或者都不调用
   - 确保测量环境一致

