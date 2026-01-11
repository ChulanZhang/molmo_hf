# Controller Training FAQ

## 1. Validation Latency 测量

### 当前实现

**Training阶段**：
- 使用 `latency_estimator` 来估计latency（快速，支持batch训练）
- 这是为了加速训练，避免每次都要实际运行模型

**Validation阶段**：
- 默认也使用 `latency_estimator`（与training一致）
- **可选**：可以启用实际测量（`measure_actual_latency_in_val=True`）

### 启用实际Latency测量

如果要让validation使用实际测量（batch_size=1，通过实际运行模型得到end-to-end latency）：

```python
trainer = JointGRPOTrainer(
    # ... other args ...
    use_latency_estimator=True,  # Training仍使用estimator
    measure_actual_latency_in_val=True,  # Validation使用实际测量
)
```

**注意**：
- 实际测量需要 `batch_size=1`（validation时）
- 会显著增加validation时间
- 测量的是完整的 `model.generate()` 时间（包括vision encoder + prefill + decode）

### 使用Hooks进行组件级测量

如果需要像 `BaseExperiment.measure_inference_latency()` 那样测量各个组件（Vision, Prefill, Decode），需要：

1. 创建一个 `BaseExperiment` 实例
2. 在validation中调用 `measure_inference_latency()`

这需要额外的代码修改，目前未实现。

---

## 2. 多卡训练支持

### 当前状态

**当前代码不支持多卡训练**。所有模型和tensor都在单个GPU上。

### 添加多卡训练支持

可以使用 PyTorch 的 `DistributedDataParallel` (DDP) 或 `DataParallel` (DP)。

#### 方案1: DataParallel (DP) - 简单但效率较低

```python
# 在 train_joint_controller.py 中
if torch.cuda.device_count() > 1:
    log.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = torch.nn.DataParallel(model)
    knob1_predictor = torch.nn.DataParallel(knob1_predictor)
    knob2_knob3_predictor = torch.nn.DataParallel(knob2_knob3_predictor)
```

**限制**：
- 只能在一台机器上使用
- 效率较低（GIL限制）
- 不支持模型并行

#### 方案2: DistributedDataParallel (DDP) - 推荐

需要修改代码以支持DDP：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, world_size):
    """Initialize DDP."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """Cleanup DDP."""
    dist.destroy_process_group()

# 在训练脚本中
if args.distributed:
    setup_ddp(rank, world_size)
    model = DDP(model, device_ids=[rank])
    knob1_predictor = DDP(knob1_predictor, device_ids=[rank])
    knob2_knob3_predictor = DDP(knob2_knob3_predictor, device_ids=[rank])
    
    # 使用 DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
```

**启动命令**：
```bash
torchrun --nproc_per_node=4 train_joint_controller.py --distributed ...
```

### 注意事项

1. **模型大小**：
   - Controller模型很小（~1M参数），多卡训练可能收益有限
   - 主要瓶颈是LLM的forward pass（用于计算accuracy和latency）

2. **数据并行 vs 模型并行**：
   - 当前实现是数据并行（每个GPU处理不同的batch）
   - 如果LLM太大，可能需要模型并行

3. **同步点**：
   - GRPO需要group内的samples在同一batch中
   - 需要确保DDP不会打乱group结构

### 推荐方案

对于当前场景（controller训练），**建议**：
- **单卡训练**：如果单卡内存足够，使用单卡即可
- **多卡训练**：如果数据量大，可以：
  1. 使用DDP进行数据并行
  2. 保持 `group_size` 不变，确保group内的samples在同一GPU上

---

## 实现状态

- [x] Validation实际latency测量（可选）
- [ ] 多卡训练支持（需要实现）
- [ ] 使用hooks进行组件级latency测量（需要实现）

