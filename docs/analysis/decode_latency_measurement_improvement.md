# Decode Latency 测量方法改进方案

> **文档目的**: 深度分析decode latency测量方法，实现最佳测量方案  
> **最后更新**: 2026-01-08

## 📋 目录

1. [当前实现分析](#当前实现分析)
2. [问题分析](#问题分析)
3. [最佳方案](#最佳方案)
4. [实施计划](#实施计划)

---

## 🔍 当前实现分析

### 现状

**两种实现方式并存**:

1. **`base_experiment.py`**: 使用`_measure_with_hooks`方法，**直接测量**decode时间
   - ✅ 在`tracked_forward`中记录第一个decode step的开始时间
   - ✅ 在`generate()`完成后记录decode结束时间
   - ✅ 直接计算：`T_LLM_decode = decode_end_time - decode_start_time`
   - ✅ 然后计算：`T_decode_per_token = T_LLM_decode / output_tokens`

2. **`motivate/base_experiment.py`和`exp6_accuracy.py`**: 使用**减法方法**
   - ❌ `T_LLM_decode = max(0.0, T_total - T_vision_total - T_LLM_prefill)`
   - ❌ 导致测量误差累积
   - ❌ 22.85%的样本出现`T_LLM_decode = 0.0`但`output_tokens > 0`

### 代码位置

**直接测量方法** (`experiments/base_experiment.py:572-640`):
```python
# 在tracked_forward中记录decode开始时间
if not is_prefill and decode_start_time is None:
    if self.device.type == 'cuda':
        torch.cuda.synchronize(self.device)
    decode_start_time = time.perf_counter()

# 在generate()完成后记录decode结束时间
if decode_start_time is not None:
    if self.device.type == 'cuda':
        torch.cuda.synchronize(self.device)
    decode_end_time = time.perf_counter()
    decode_times.append((decode_end_time - decode_start_time) * 1000)
```

**减法方法** (`experiments/motivate/base_experiment.py:538`):
```python
results["T_LLM_decode"] = max(0.0, results["T_total"] - results.get("T_vision_total", 0.0) - results.get("T_LLM_prefill", 0.0))
```

---

## ⚠️ 问题分析

### 为什么减法方法有问题？

1. **测量环境不一致**:
   - Vision和Prefill分别测量时，GPU可能有缓存，测量较快
   - `T_total`测量前调用`empty_cache()`，缓存被清空，测量较慢
   - 导致`T_vision_total + T_LLM_prefill > T_total`，`T_LLM_decode`为负数

2. **误差累积**:
   - 每个组件的测量误差会累积
   - Vision被计算了3次（分别测量vision、prefill、total），每次时间不同
   - 对于短输出，误差占比更大

3. **数据质量问题**:
   - 22.85%的样本`T_LLM_decode = 0.0`但`output_tokens > 0`
   - 导致`T_decode_per_token = 0.0`，数据不可用

### 为什么直接测量方法更好？

1. **准确性**:
   - 直接测量decode阶段的总时间，不依赖其他组件的测量
   - 避免了误差累积
   - 测量环境一致（在同一个`generate()`调用中）

2. **简单性**:
   - 只需要记录两个时间点：decode开始和结束
   - 然后除以`output_tokens`得到per-token latency
   - 逻辑清晰，易于理解和维护

3. **可靠性**:
   - 不依赖减法计算，不会出现负数
   - 即使测量有误差，也是直接误差，不会放大

---

## ✅ 最佳方案

### 方案：统一使用直接测量方法

**核心思想**:
1. **直接测量decode总时间**: 在`generate()`内部，记录第一个decode step的开始时间和最后一个decode step的结束时间
2. **计算per-token latency**: `T_decode_per_token = T_LLM_decode / output_tokens`

**实现要点**:

1. **在`tracked_forward`中记录decode开始时间**:
   ```python
   if not is_prefill and decode_start_time is None:
       # 第一个decode step
       if self.device.type == 'cuda':
           torch.cuda.synchronize(self.device)
       decode_start_time = time.perf_counter()
   ```

2. **在`generate()`完成后记录decode结束时间**:
   ```python
   if decode_start_time is not None:
       if self.device.type == 'cuda':
           torch.cuda.synchronize(self.device)
       decode_end_time = time.perf_counter()
       T_LLM_decode = (decode_end_time - decode_start_time) * 1000
   ```

3. **计算per-token latency**:
   ```python
   output_tokens = output.shape[1] - input_ids.shape[1]
   T_decode_per_token = T_LLM_decode / output_tokens if output_tokens > 0 else 0.0
   ```

### 优势

1. **准确性**: 直接测量，无误差累积
2. **简单性**: 逻辑清晰，易于实现
3. **可靠性**: 不会出现负数或0值（除非真的没有decode）
4. **一致性**: 所有实验使用相同的测量方法

---

## 🚀 实施计划

### 步骤1: 更新`motivate/base_experiment.py`

**当前**: 使用减法方法
**目标**: 使用`_measure_with_hooks`方法（继承自`base_experiment.py`）

**检查**: `motivate/base_experiment.py`是否继承自`base_experiment.py`？

### 步骤2: 更新`exp6_accuracy.py`

**当前**: 使用减法方法
**目标**: 使用直接测量方法（类似`_measure_with_hooks`）

**实现**: 在`exp6_accuracy.py`中实现类似的hook机制，直接测量decode时间

### 步骤3: 验证

1. 运行实验，检查是否还有`T_LLM_decode = 0.0`但`output_tokens > 0`的情况
2. 比较新旧方法的测量结果
3. 确认per-token latency的分布是否更合理

---

## 📊 预期效果

### 改进前（减法方法）

- ❌ 22.85%的样本`T_LLM_decode = 0.0`
- ❌ Decode per-token latency与output_tokens有0.70的相关性（测量误差导致）
- ❌ 短输出的测量误差更大（6.09% vs 2.00%）

### 改进后（直接测量方法）

- ✅ 所有有decode的样本都有有效的`T_LLM_decode`
- ✅ Decode per-token latency应该只与配置相关，与output_tokens无关
- ✅ 测量误差更小，更稳定

---

## 🔗 相关文档

- **[key_insights_latency_measurement.md](key_insights_latency_measurement.md)**: 测量关键洞察
- **[latency_measurement_refactoring.md](latency_measurement_refactoring.md)**: 测量重构方案
- **[decode_measurement_strategy.md](decode_measurement_strategy.md)**: Decode测量策略

---

**最后更新**: 2026-01-08  
**维护者**: Analysis Team



