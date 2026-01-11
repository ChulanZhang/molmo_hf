# 直接测量Latency而不是使用Estimator

## 问题

既然当前实现是逐个样本执行（batch_size=1 per sample），那为什么还要用latency estimator呢？我们直接统计prefill latency和decode per-token latency不就行了吗？以prefill latency作为latency主指标。

## 答案

**用户说得对！** 既然batch_size=1，可以直接测量实际latency，不需要estimator。

## 修改方案

### 1. 在`_execute_model`中直接测量latency

使用hooks直接测量prefill和decode latency（参考`BaseExperiment._measure_with_hooks`）：

```python
def _execute_model(self, ...):
    # ... apply knobs ...
    
    # Measure latency directly using hooks
    latency_results = self._measure_latency_with_hooks(
        batch={
            'input_ids': input_ids,
            'images': images,
            'image_masks': image_masks,
            'image_input_idx': image_input_idx,
        },
        max_new_tokens=max_new_tokens,
        generation_config=generation_config,
    )
    
    output_ids = latency_results['output_ids']
    prefill_latency = latency_results.get('T_LLM_prefill', 0.0)
    decode_latency = latency_results.get('T_LLM_decode', 0.0)
    
    # ... compute accuracy ...
    
    return {
        'output_ids': output_ids,
        'accuracy': accuracy,
        'prefill_latency': prefill_latency,  # Primary metric
        'decode_latency': decode_latency,
    }
```

### 2. 实现`_measure_latency_with_hooks`方法

参考`BaseExperiment._measure_with_hooks`的实现，使用hooks测量：

- **Prefill latency**: 使用hooks在第一个和最后一个transformer block上测量
- **Decode latency**: 在tracked_forward中测量decode阶段的总时间
- **Per-token latency**: `decode_latency / output_tokens`

### 3. 在`train_step`中使用测量的latency

```python
# In train_step, after _execute_model
result = self._execute_model(...)

# Use prefill latency as primary metric (as user requested)
prefill_latency = result.get('prefill_latency', 0.0)
decode_latency = result.get('decode_latency', 0.0)

# For reward computation, use prefill_latency as the main latency metric
latency = prefill_latency  # Primary metric
# Optionally: total_latency = prefill_latency + decode_latency
```

### 4. 移除或简化latency estimator的使用

- **Training**: 直接使用测量的latency，不需要estimator
- **Validation**: 也可以直接测量，或者保留estimator作为fallback

## 优势

1. **更准确**: 直接测量实际latency，不依赖estimator的预测误差
2. **更简单**: 不需要维护estimator，减少代码复杂度
3. **更符合实际情况**: 使用实际测量的latency作为reward，更真实
4. **Prefill作为主指标**: 符合用户要求，prefill latency是主要指标

## 注意事项

1. **Hook overhead**: 使用hooks会有少量overhead，但对于batch_size=1的情况，这是可以接受的
2. **CUDA synchronization**: 确保在测量前后进行`torch.cuda.synchronize()`以获得准确时间
3. **Error handling**: 如果hook测量失败，可以fallback到estimator或简单的总时间测量

## 实现细节

参考`experiments/base_experiment.py`中的`_measure_with_hooks`方法：
- 使用hooks在vision_backbone和transformer blocks上测量
- 使用tracked_forward来区分prefill和decode阶段
- 正确清理hooks以避免内存泄漏

