# expanded_batch_size 的作用和多卡训练说明

## expanded_batch_size 的作用

### 定义
```python
expanded_batch_size = batch_size * group_size
```

### 含义
`expanded_batch_size` 表示**扩展后的配置总数**，是GRPO算法的核心概念：

1. **原始batch**：有 `batch_size` 个原始samples
2. **GRPO扩展**：对每个原始sample，采样 `group_size` 次不同的配置
3. **扩展后**：总共有 `batch_size * group_size` 个配置需要执行

### 示例

假设 `batch_size=4`, `group_size=5`：

```
原始samples: [Sample1, Sample2, Sample3, Sample4]
              ↓ (每个sample采样5次)
扩展后configs:
  Sample1 → [Config1-1, Config1-2, Config1-3, Config1-4, Config1-5]
  Sample2 → [Config2-1, Config2-2, Config2-3, Config2-4, Config2-5]
  Sample3 → [Config3-1, Config3-2, Config3-3, Config3-4, Config3-5]
  Sample4 → [Config4-1, Config4-2, Config4-3, Config4-4, Config4-5]

expanded_batch_size = 4 * 5 = 20 个配置
```

### 为什么需要 expanded_batch_size？

1. **GRPO算法要求**：需要在同一组内比较多个配置的相对表现
2. **相对优势计算**：计算组内标准化优势 `adv = (r - mean(r)) / std(r)`
3. **代码组织**：需要知道总共有多少个配置需要处理

---

## 多卡训练的实现方式

### 当前实现（DataParallel）

#### 单GPU情况
```
batch_size = 1
expanded_batch_size = 1 * 5 = 5

处理流程：
GPU0: Sample1 → [Config1-1, Config1-2, Config1-3, Config1-4, Config1-5] (串行执行5次)
```

#### 多GPU情况（例如4个GPU）
```
batch_size = 4 (每个GPU处理1个sample)
expanded_batch_size = 4 * 5 = 20

处理流程：
GPU0: Sample1 → [Config1-1, Config1-2, Config1-3, Config1-4, Config1-5] (串行执行5次)
GPU1: Sample2 → [Config2-1, Config2-2, Config2-3, Config2-4, Config2-5] (串行执行5次)
GPU2: Sample3 → [Config3-1, Config3-2, Config3-3, Config3-4, Config3-5] (串行执行5次)
GPU3: Sample4 → [Config4-1, Config4-2, Config4-3, Config4-4, Config4-5] (串行执行5次)
```

### 关键点

1. **每个GPU处理1个原始sample**
   - `batch_size = num_gpus`（例如4个GPU，batch_size=4）
   - 每个GPU处理1个sample

2. **每个sample的group_size个配置在对应GPU上串行执行**
   - 每个GPU上，对该sample的5个配置**串行执行**
   - 不是并行执行（因为每个配置需要完整的模型forward pass）

3. **DataParallel的作用**
   - 将不同的samples分配到不同的GPU
   - 每个GPU独立处理自己的sample及其group_size个配置
   - 最后在主GPU上聚合结果

### 代码流程

```python
# 1. 扩展batch
input_ids_expanded = batch['input_ids'].repeat_interleave(self.group_size, dim=0)
# 形状: (batch_size, seq_len) → (batch_size * group_size, seq_len)

# 2. 串行执行所有expanded configs
for i in range(expanded_batch_size):  # 例如 20 次循环
    result = self._execute_model(
        input_ids=input_ids_expanded[i:i+1],  # 每次处理1个config
        ...
    )
```

### 当前实现的限制

**问题**：虽然使用了DataParallel，但实际执行是**完全串行**的，没有充分利用多GPU的优势。

**原因**：
1. **串行循环**：代码使用 `for i in range(expanded_batch_size)` 串行处理每个配置
2. **单sample执行**：每次调用 `_execute_model` 时只传入1个sample (`input_ids_expanded[i:i+1]`)
3. **DataParallel无效**：DataParallel需要batch_size > 1才能并行化，但这里每次只有1个sample

**实际执行流程**：
```
主GPU (cuda:0) 串行执行：
  Config1-1 → GPU0 (DataParallel但只有1个sample，无法并行)
  Config1-2 → GPU0
  Config1-3 → GPU0
  ...
  Config4-5 → GPU0
```

**多GPU没有被充分利用**：虽然模型被DataParallel包装，但因为每次只处理1个sample，其他GPU处于空闲状态。

### 理想的并行化方案（未来优化）

#### 方案1：每个GPU并行执行group内的配置
```
GPU0: Sample1 → [Config1-1, Config1-2, Config1-3, Config1-4, Config1-5] (并行执行5次)
GPU1: Sample2 → [Config2-1, Config2-2, Config2-3, Config2-4, Config2-5] (并行执行5次)
...
```

**挑战**：
- 需要支持动态batch（不同配置可能有不同的模型结构）
- 需要处理不同配置的latency差异

#### 方案2：跨GPU并行执行同一sample的配置
```
Sample1的5个配置：
  Config1-1 → GPU0
  Config1-2 → GPU1
  Config1-3 → GPU2
  Config1-4 → GPU3
  Config1-5 → GPU0 (如果GPU数 < group_size)
```

**挑战**：
- 需要更复杂的调度逻辑
- 需要跨GPU聚合结果

---

## 总结

### expanded_batch_size 的作用
- **表示扩展后的配置总数**：`batch_size * group_size`
- **GRPO算法要求**：需要在组内比较多个配置
- **代码组织**：统一管理所有需要执行的配置

### 多卡训练的实现
- **当前**：每个GPU处理1个sample，该sample的group_size个配置在该GPU上串行执行
- **并行化**：不同samples在不同GPU上并行处理
- **限制**：每个sample的group内配置是串行的，没有充分利用多GPU

### 性能影响
- **单GPU**：`expanded_batch_size = 1 * 5 = 5`，串行执行5次，时间 = 5 * T
- **4 GPU（当前实现）**：`expanded_batch_size = 4 * 5 = 20`，但**仍然是串行执行**，时间 ≈ 20 * T（几乎没有加速）
- **4 GPU（理想情况）**：如果真正并行化，时间 ≈ 5 * T（4倍加速）

**当前实现的问题**：多GPU几乎没有带来加速，因为所有配置都在主GPU上串行执行。

### 建议

#### 当前实现的评估
**问题**：多GPU几乎没有带来加速，因为所有配置都在主GPU上串行执行。

**原因**：
1. 每次只处理1个sample，DataParallel无法并行化
2. 不同配置的模型结构可能不同（不同的top_k、不同的active blocks），难以批量执行

#### 优化方案

**方案1：批量执行相同结构的配置**（推荐）
```python
# 按配置结构分组
configs_by_structure = group_configs_by_structure(all_configs)

# 批量执行相同结构的配置
for structure, configs in configs_by_structure.items():
    batch_input_ids = torch.cat([c['input_ids'] for c in configs])
    # DataParallel可以并行处理这个batch
    results = model.generate(batch_input_ids, ...)
```

**方案2：异步执行不同samples的配置**
```python
# 使用多进程/多线程并行执行不同samples的配置
# 每个sample的group_size个配置在一个进程/线程中串行执行
# 不同samples的配置在不同进程/线程中并行执行
```

**方案3：使用DDP（DistributedDataParallel）**
- 每个GPU处理不同的samples
- 每个GPU上，对该sample的group_size个配置串行执行
- 需要修改代码以支持DDP

#### 当前建议
对于controller训练：
1. **单GPU训练**：如果单GPU内存足够，使用单GPU即可（当前实现已经针对单GPU优化）
2. **多GPU训练**：当前实现多GPU收益有限，建议：
   - 如果必须使用多GPU，考虑实现方案1（批量执行相同结构）
   - 或者等待DDP支持（方案3）

