# 自动 Batch Size 调整逻辑详解

## 概述

**重要**：自动调整 batch size **不是**从 `--batch_size` 开始向下测试，而是：
1. 根据 `max_crops` 值**估算**一个初始 batch size
2. 从这个估算值开始，使用**二分搜索**找到最大的可用 batch size
3. **每个 `max_crops` 值独立测试**，互不影响

## 详细流程

### 步骤1：初始估算

对于每个 `max_crops` 值，首先根据经验公式估算初始 batch size：

```python
# 对于 max_crops=12
base_batch_size = 64  # 你传入的 --batch_size
scale_factor = 0.3    # max_crops=12 对应的缩放因子
estimated_batch_size = 64 * 0.3 = 19

# 但不会超过 base_batch_size
current_batch_size = min(19, 64) = 19
```

**缩放因子表**：
| max_crops | scale_factor | 对于 base=64 的估算值 |
|-----------|--------------|---------------------|
| ≤ 2       | 1.0          | 64                  |
| ≤ 4       | 0.8          | 51                  |
| ≤ 6       | 0.6          | 38                  |
| ≤ 8       | 0.5          | 32                  |
| ≤ 10      | 0.4          | 25                  |
| ≤ 12      | 0.3          | 19                  |
| > 12      | 0.25         | 16                  |

### 步骤2：二分搜索找到最优值

从估算值开始，使用二分搜索：

#### 场景A：估算值太小（可以增大）

```
尝试 batch_size=19 → ✅ 成功
  → 尝试增大到 19 * 1.5 = 28
尝试 batch_size=28 → ✅ 成功
  → 尝试增大到 28 * 1.5 = 42
尝试 batch_size=42 → ✅ 成功
  → 尝试增大到 42 * 1.5 = 63 (不超过64)
尝试 batch_size=63 → ✅ 成功
  → 返回 63（最大可用值）
```

#### 场景B：估算值太大（需要减小）

```
尝试 batch_size=19 → ❌ OOM
  → 减小到 19 // 2 = 9
尝试 batch_size=9 → ✅ 成功
  → 尝试增大到 (9 + 19) // 2 = 14
尝试 batch_size=14 → ✅ 成功
  → 尝试增大到 (14 + 19) // 2 = 16
尝试 batch_size=16 → ✅ 成功
  → 返回 16（最大可用值）
```

#### 场景C：估算值刚好

```
尝试 batch_size=19 → ✅ 成功
  → 尝试增大到 19 * 1.5 = 28
尝试 batch_size=28 → ❌ OOM
  → 尝试减小到 (19 + 28) // 2 = 23
尝试 batch_size=23 → ✅ 成功
  → 返回 23（最大可用值）
```

## 实际运行示例

### 你的命令

```bash
torchrun --nproc-per-node=4 experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --batch_size 64 \
    --start_from_max_crops 12 \
    --auto_adjust_batch_size
```

### 执行流程

#### 对于 max_crops=12：

1. **初始估算**：
   ```
   estimated = 64 * 0.3 = 19
   starting = min(19, 64) = 19
   ```

2. **测试过程**（假设）：
   ```
   ✓ Batch size 19 works
     → Trying larger batch size: 28...
   ✓ Batch size 28 works
     → Trying larger batch size: 42...
   ✓ Batch size 42 works
     → Trying larger batch size: 63...
   ✓ Batch size 63 works
   Using batch size: 63 for max_crops=12
   ```

3. **实际运行**：使用 batch_size=63 运行完整的 accuracy 测试

#### 对于 max_crops=13：

1. **初始估算**：
   ```
   estimated = 64 * 0.25 = 16
   starting = min(16, 64) = 16
   ```

2. **测试过程**（假设）：
   ```
   ✓ Batch size 16 works
     → Trying larger batch size: 24...
   ✓ Batch size 24 works
     → Trying larger batch size: 36...
   ✗ Batch size 36 caused OOM
     → Trying 24...
   Using batch size: 24 for max_crops=13
   ```

3. **实际运行**：使用 batch_size=24 运行完整的 accuracy 测试

## 关键点

### 1. 不是从 64 向下测试

❌ **错误理解**：从 batch_size=64 开始，如果 OOM 就减到 32，再 OOM 就减到 16...

✅ **正确理解**：
- 对于 max_crops=12：从估算值 19 开始，**向上**搜索到最大可用值（可能是 63）
- 对于 max_crops=13：从估算值 16 开始，**向上**搜索到最大可用值（可能是 24）

### 2. 每个 max_crops 独立测试

- `max_crops=12` 和 `max_crops=13` 的 batch size **完全独立**
- 不会因为 `max_crops=12` 用了 63，`max_crops=13` 就必须更小
- 每个都会找到自己的最大可用值

### 3. 二分搜索策略

- **如果成功**：尝试增大（×1.5），直到达到 base_batch_size 或 OOM
- **如果失败**：使用二分搜索，在已知的"工作值"和"失败值"之间找中点
- **最多尝试 5 次**：避免无限循环

### 4. 测试内容

每次测试都会：
1. 创建 dataloader
2. 加载一个 batch
3. 执行 forward pass
4. **执行生成**（这是关键，因为生成需要更多内存）

这确保了找到的 batch size 在实际运行中不会 OOM。

## 为什么这样设计？

### 优势

1. **高效**：不需要从很大的值开始向下试，节省时间
2. **智能**：根据 max_crops 自动估算，通常很接近最优值
3. **安全**：测试时包含生成，确保实际运行不会 OOM
4. **最大化性能**：找到每个 max_crops 的最大可用 batch size

### 示例对比

#### 方法A：从 base_batch_size 向下（低效）

```
max_crops=12:
  尝试 64 → OOM
  尝试 32 → OOM
  尝试 16 → ✅ 成功
  使用 16（但可能 20-30 也能工作！）
```

#### 方法B：从估算值向上（当前方法，高效）

```
max_crops=12:
  估算 19 → ✅ 成功
  尝试 28 → ✅ 成功
  尝试 42 → ✅ 成功
  尝试 63 → ✅ 成功
  使用 63（找到最大可用值！）
```

## 日志解读

运行时会看到类似这样的日志：

```
INFO:__main__:Testing max_crops=12
INFO:__main__:Finding optimal batch size for max_crops=12 (estimated: 19, starting with: 19)...
INFO:__main__:✓ Batch size 19 works for max_crops=12
INFO:__main__:  Trying larger batch size: 28...
INFO:__main__:✓ Batch size 28 works for max_crops=12
INFO:__main__:  Trying larger batch size: 42...
INFO:__main__:✓ Batch size 42 works for max_crops=12
INFO:__main__:  Trying larger batch size: 63...
INFO:__main__:✓ Batch size 63 works for max_crops=12
INFO:__main__:Using batch size: 63 for max_crops=12
INFO:__main__:Measuring accuracy for max_crops=12...
```

这表示：
- 估算值 19 可以工作
- 逐步增大到 63 都可以工作
- 最终使用 63 进行完整的 accuracy 测试

## 总结

1. **不是向下测试**：从估算值开始，向上搜索最大可用值
2. **每个 max_crops 独立**：互不影响，各自找到最优值
3. **智能估算**：根据 max_crops 自动估算，通常很准确
4. **二分搜索**：高效找到最大可用 batch size
5. **安全测试**：包含生成测试，确保实际运行不会 OOM

**你的命令**：
- `--batch_size 64`：这是**上限**，不会超过这个值
- `--start_from_max_crops 12`：从 max_crops=12 开始测试
- `--auto_adjust_batch_size`：为每个 max_crops 自动找到最优 batch size


