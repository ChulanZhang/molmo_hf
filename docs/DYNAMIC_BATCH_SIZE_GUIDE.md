# 动态 Batch Size 调整指南

## 功能概述

Accuracy profiling 脚本现在支持：
1. **从指定 max_crops 开始测试**：可以跳过较小的 max_crops，直接从较大的开始
2. **自动调整 batch size**：根据每个 max_crops 自动找到不会 OOM 的最大 batch size

## 使用方法

### 方案1：从 max_crops=12 开始测试（推荐）

```bash
# 从max_crops=12开始，自动调整batch size
python experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --batch_size 64 \
    --start_from_max_crops 12 \
    --auto_adjust_batch_size
```

**优势**：
- 从较大的 max_crops 开始，可以先用较大的 batch size
- 自动调整确保不会 OOM
- 如果某个 max_crops 的 batch size 太小，会自动降低

### 方案2：指定 max_crops 列表，自动调整

```bash
# 只测试较大的max_crops值
python experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --batch_size 64 \
    --max_crops_list 12 13 \
    --auto_adjust_batch_size
```

### 方案3：禁用自动调整（使用固定 batch size）

```bash
# 使用固定batch size，手动控制
python experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --batch_size 32 \
    --no_auto_adjust_batch_size
```

## 自动调整算法

### 1. 初始估计

基于经验公式，根据 `max_crops` 估计初始 batch size：

```python
if max_crops <= 4:
    scale_factor = 1.0      # 使用100%的base batch size
elif max_crops <= 8:
    scale_factor = 0.75     # 使用75%
elif max_crops <= 12:
    scale_factor = 0.5      # 使用50%
else:
    scale_factor = 0.4      # 使用40%
```

### 2. 二分搜索

如果初始估计导致 OOM：
1. 将 batch size 减半
2. 重新测试
3. 重复直到找到工作的 batch size

如果初始估计安全：
1. 尝试增加到 base batch size
2. 找到最大安全值

### 3. 测试方法

对每个候选 batch size：
1. 创建 dataloader
2. 加载一个 batch
3. 执行 forward pass（不生成，节省时间）
4. 检查是否 OOM

## 性能影响

### 自动调整的开销

- **时间开销**：每个 max_crops 需要 1-5 次测试（每次约 1-2 秒）
- **总开销**：约 10-30 秒（取决于尝试次数）

### 收益

- **避免 OOM**：不会因为 batch size 太大而崩溃
- **最大化性能**：自动找到每个 max_crops 的最大可用 batch size
- **自动化**：无需手动调整

## 实际使用建议

### 场景1：从大 max_crops 开始（你的需求）

```bash
# 从max_crops=12开始，使用较大的base batch size
python experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --batch_size 64 \
    --start_from_max_crops 12 \
    --auto_adjust_batch_size
```

**预期行为**：
- max_crops=12: 自动找到合适的 batch size（可能是 32-48）
- max_crops=13: 自动找到合适的 batch size（可能是 24-32）

### 场景2：测试所有 max_crops，自动优化

```bash
# 测试所有max_crops，每个自动优化batch size
python experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --batch_size 64 \
    --auto_adjust_batch_size
```

**预期行为**：
- max_crops=2-4: 使用 batch_size=64
- max_crops=5-8: 使用 batch_size=48
- max_crops=9-12: 使用 batch_size=32
- max_crops=13: 使用 batch_size=24-32

### 场景3：多卡运行

```bash
# 多卡时，每个GPU独立调整batch size
torchrun --nproc-per-node=8 experiments/profiling/knob1_tokens/exp1_accuracy.py \
    --batch_size 64 \
    --start_from_max_crops 12 \
    --auto_adjust_batch_size
```

## 结果记录

结果 JSON 文件中会记录每个 max_crops 实际使用的 batch size：

```json
{
  "summary": [
    {
      "max_crops": 12,
      "accuracy": 0.85,
      "batch_size_used": 32,  // 实际使用的batch size
      ...
    }
  ]
}
```

## 调优建议

如果发现自动调整的 batch size 太保守或太激进，可以修改 `_estimate_batch_size_for_max_crops` 方法中的 scale_factor。

例如，如果发现 max_crops=12 时 batch_size=32 很安全，可以调整：

```python
elif max_crops <= 12:
    scale_factor = 0.6  # 从0.5增加到0.6
```

## 故障排除

### 问题1：自动调整总是失败

**可能原因**：
- Base batch size 太大
- GPU 内存不足

**解决**：
- 降低 `--batch_size` 参数
- 检查 GPU 内存使用情况

### 问题2：自动调整太慢

**可能原因**：
- 测试次数太多（max_attempts=5）

**解决**：
- 可以修改代码中的 `max_attempts` 参数（默认5次）
- 或者禁用自动调整，手动设置较小的 batch size

### 问题3：某些 max_crops 仍然 OOM

**可能原因**：
- 自动调整找到了工作的 batch size，但实际生成时内存需求更大

**解决**：
- 自动调整会保守一些，但生成时可能还需要更多内存
- 如果遇到，可以手动降低该 max_crops 的 batch size

## 总结

- **推荐使用**：`--start_from_max_crops 12 --auto_adjust_batch_size`
- **优势**：自动化、安全、最大化性能
- **开销**：每个 max_crops 约 10-30 秒的测试时间
- **结果**：每个配置都会记录实际使用的 batch size

