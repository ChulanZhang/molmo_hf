# Decode Latency 测量方法验证

> **文档目的**: 验证当前decode latency测量方法的正确性  
> **最后更新**: 2026-01-08

## ✅ 当前实现验证

### `acc_lat_profiling.py`使用的测量方法

`acc_lat_profiling.py`调用`self.measure_inference_latency()`，该方法来自`BaseExperiment`（`base_experiment.py`），使用`_measure_with_hooks`方法**直接测量decode时间**。

### 实现细节（`base_experiment.py:572-661`）

**1. 记录decode开始时间**（第一个decode step）:
```python
def tracked_forward(*args, **kwargs):
    nonlocal forward_count, vision_start_time, decode_start_time
    is_prefill = forward_count == 0
    
    # Record decode start time (only on first decode step)
    if not is_prefill and decode_start_time is None:
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        decode_start_time = time.perf_counter()
    
    # Call original forward
    output = original_forward(*args, **kwargs)
    
    # Increment forward count after prefill
    if is_prefill:
        forward_count += 1
    
    return output
```

**2. 记录decode结束时间**（所有decode steps完成后）:
```python
# After model.generate() completes
if decode_start_time is not None:
    if self.device.type == 'cuda':
        torch.cuda.synchronize(self.device)
    decode_end_time = time.perf_counter()
    decode_times.append((decode_end_time - decode_start_time) * 1000)
```

**3. 计算平均值**:
```python
if decode_times:
    results["T_LLM_decode"] = np.mean(decode_times)
```

**4. 计算per-token latency**（在`acc_lat_profiling.py:1240`）:
```python
"T_decode_per_token": latency_results.get("T_LLM_decode", 0.0) / max(num_output_tokens, 1)
```

---

## ✅ 结论

**当前实现已经是正确的！**

1. **直接测量**: 不是减法，而是直接测量decode阶段的总时间
2. **准确计算**: `T_LLM_decode = decode_end_time - decode_start_time`
3. **Per-token计算**: `T_decode_per_token = T_LLM_decode / output_tokens`

这正是用户建议的方法：**直接统计total decode latency，然后除以output token数**。

---

## 📊 为什么这个方法更好？

### 优势

1. **准确性**: 直接测量，无误差累积
2. **简单性**: 逻辑清晰，易于理解
3. **可靠性**: 不会出现负数或0值（除非真的没有decode）
4. **一致性**: 所有实验使用相同的测量方法

### 与减法方法对比

| 方法 | 准确性 | 误差累积 | 可靠性 |
|------|--------|---------|--------|
| **直接测量**（当前） | ✅ 高 | ✅ 无 | ✅ 高 |
| 减法方法 | ❌ 低 | ❌ 有 | ❌ 低 |

---

## 🔍 验证检查

### 检查点1: 是否使用直接测量

✅ **已确认**: `base_experiment.py`使用`_measure_with_hooks`方法，直接测量decode时间

### 检查点2: 是否使用减法

✅ **已确认**: 未发现减法计算方法（`T_LLM_decode = T_total - T_vision - T_prefill`）

### 检查点3: Per-token计算

✅ **已确认**: `acc_lat_profiling.py:1240`正确计算：`T_decode_per_token = T_LLM_decode / output_tokens`

---

## 📝 总结

**当前实现已经完全符合最佳实践**：

1. ✅ 直接测量decode总时间（不是减法）
2. ✅ 除以output_tokens得到per-token latency
3. ✅ 无误差累积
4. ✅ 逻辑清晰

**无需修改**，当前实现已经是正确且最优的。

---

## 🔗 相关文档

- **[DECODE_LATENCY_MEASUREMENT_IMPROVEMENT.md](DECODE_LATENCY_MEASUREMENT_IMPROVEMENT.md)**: 改进方案文档（针对其他使用减法方法的实验）
- **[key_insights_latency_measurement.md](key_insights_latency_measurement.md)**: 测量关键洞察

---

**最后更新**: 2026-01-08  
**维护者**: Analysis Team


