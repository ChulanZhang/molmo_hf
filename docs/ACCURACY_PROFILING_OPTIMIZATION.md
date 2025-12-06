# Accuracy Profiling 性能优化指南

## 当前性能分析

- **Batch size**: 32
- **GPU利用率**: <100% (未充分利用)
- **Memory使用**: 50GB/80GB (62.5%)
- **每个batch时间**: 1.80秒
- **总时间**: 约3.3小时 (6699 batches)

## 优化策略

### 1. 增加 Batch Size（最直接）

**当前**: 32 → **建议**: 64

**优势**:
- 提高GPU利用率
- 更好的并行度
- 减少overhead

**内存估算**:
- 当前: 50GB
- Batch size 64: 约 100GB（可能OOM）
- **建议**: 先试48，如果内存足够再试64

### 2. 优化数据加载（重要）

**当前问题**:
```python
num_workers=0,  # 串行加载，CPU等待
pin_memory=False,  # 数据移动慢
```

**优化**:
```python
num_workers=4,  # 并行加载数据
pin_memory=True,  # 加速CPU->GPU传输
prefetch_factor=2,  # 预取数据
```

**预期提升**: 10-20%

### 3. 减少 max_new_tokens（关键）

**当前**: 32 → **建议**: 16 或 8

**优势**:
- 根据之前的分析，VQA答案通常1-10 tokens
- 16应该覆盖>99%的答案
- 8可能覆盖>95%的答案

**性能提升**:
- max_new_tokens=16: 约2倍加速（相比32）
- max_new_tokens=8: 约4倍加速（相比32）

### 4. 优化数据移动

**当前**:
```python
batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
        for k, v in batch.items()}
```

**优化**:
```python
batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
        for k, v in batch.items()}
```

**优势**: 异步数据传输，减少等待时间

### 5. 使用多卡（最有效）

**当前**: 单卡 → **建议**: 8卡

**优势**:
- 线性加速（理论上8倍）
- 每个GPU处理部分数据
- 总时间: 3.3小时 → 约25分钟

**命令**:
```bash
torchrun --nproc-per-node=8 experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --batch_size 32  # 每个GPU的batch size
```

### 6. 优化生成配置

**当前**: 每次循环都创建GenerationConfig

**优化**: 在循环外创建一次

### 7. 减少不必要的计算

- 移除调试日志（除了第一个batch）
- 优化accuracy计算（批量处理）

## 综合优化方案

### 方案A: 单卡优化（保守）

1. Batch size: 32 → 48
2. max_new_tokens: 32 → 16
3. num_workers: 0 → 4
4. pin_memory: False → True
5. non_blocking: 添加

**预期**: 2-3倍加速，总时间约1-1.5小时

### 方案B: 单卡激进优化

1. Batch size: 32 → 64（如果内存允许）
2. max_new_tokens: 32 → 8
3. 所有数据加载优化

**预期**: 4-5倍加速，总时间约40-50分钟

### 方案C: 多卡优化（推荐）

1. 使用8卡
2. Batch size per GPU: 32
3. max_new_tokens: 16
4. 所有数据加载优化

**预期**: 16倍加速（8卡 × 2倍优化），总时间约12-15分钟

## 实施优先级

1. **立即实施**: 减少max_new_tokens到16
2. **高优先级**: 优化数据加载（num_workers, pin_memory）
3. **中优先级**: 增加batch size到48
4. **最高效**: 使用多卡（torchrun）

