# MoE Top-K Latency Analysis Experiment

## 实验概述

本实验旨在研究MoE（Mixture of Experts）模型中Top-K参数对模型推理延迟（latency）的影响。通过系统性地改变Top-K值，测量Prefill和Decode阶段的延迟变化，以理解专家选择数量对性能的影响。

## 模型配置

根据`configs/model/config.json`，Molmo模型配置如下：

- **MoE Experts数量**: `moe_num_experts = 64`
- **默认Top-K值**: `moe_top_k = 8`
- **模型架构**: 使用纯PyTorch实现的MoE（HF版本，不依赖megablocks）
- **MoE实现**: `MolmoeSparseMoeBlock` - 纯PyTorch实现，位于`molmo/models/modeling_molmoe.py`

## Top-K值选择分析

### 选择的测试值

实验测试以下Top-K值：`[1, 2, 4, 8, 16, 32]`

### 选择合理性

1. **覆盖范围**：
   - 最小值：`k=1`（最少专家选择，最低计算量）
   - 最大值：`k=32`（64个专家的一半，中等计算量）
   - 默认值：`k=8`（模型默认配置）

2. **2的幂次序列**：
   - 所有值都是2的幂次（1, 2, 4, 8, 16, 32）
   - 便于观察指数增长趋势
   - 便于分析和可视化

3. **理论计算量比例**：
   - `k=1`: 基准（1x计算量）
   - `k=2`: 2x计算量
   - `k=4`: 4x计算量
   - `k=8`: 8x计算量（默认配置）
   - `k=16`: 16x计算量
   - `k=32`: 32x计算量

4. **实际意义**：
   - `k=1-8`: 常见配置范围，大多数MoE模型使用
   - `k=16-32`: 高计算量配置，用于研究极端情况下的性能瓶颈

### 为什么不测试k=64？

- `k=64`意味着使用所有专家，这在实际应用中很少见
- 对于64个专家的模型，通常top_k远小于专家总数
- 测试到`k=32`已经覆盖了从最小到中等计算量的范围

## 实验方法

### 代码位置

实验脚本位于：`experiments/profiling/knob2_topk/exp_moe_topk.py`

### 实验流程

1. **模型加载**：
   - 从checkpoint加载Molmo模型
   - 加载processor用于输入处理

2. **Top-K值修改**：
   ```python
   # 1. 更新config
   self.model.config.moe_top_k = k
   
   # 2. 更新所有MoE blocks的top_k（HF版本：直接修改block.mlp.top_k）
   from molmo.models.modeling_molmoe import MolmoeSparseMoeBlock
   
   for block in self.model.model.transformer.blocks:
       if hasattr(block, 'mlp') and isinstance(block.mlp, MolmoeSparseMoeBlock):
           block.mlp.top_k = k
   ```

3. **输入准备**：
   - 固定输入：336x336蓝色图像
   - 固定文本提示："Describe this image."

4. **延迟测量**：
   - **Prefill延迟**：使用forward hooks测量LLM prefill阶段
   - **Decode延迟**：通过总延迟减去Vision和Prefill延迟计算
   - 每个top_k值运行`num_samples`次（默认50次）

5. **统计计算**：
   - 计算P50（中位数）、P95、P99百分位数
   - 计算均值、标准差、最小值、最大值

### 关键技术细节

#### MoE Block结构

Molmo HF版本使用纯PyTorch实现的MoE（不依赖megablocks）：
- `block.mlp`: `MolmoeSparseMoeBlock`类型（位于`molmo/models/modeling_molmoe.py`）
- `block.mlp.top_k`: 直接可访问的整数属性，可动态修改
- `block.mlp.num_experts`: 专家数量
- `block.mlp.experts`: `nn.ModuleList`包含所有专家MLP

**注意**: HF版本与原始molmo仓库（使用megablocks）不同：
- HF版本：`block.mlp.top_k`（直接属性）
- 原始版本：`block.ffn.args.top_k`（通过megablocks的Arguments对象）

#### 验证机制

代码包含验证逻辑：
```python
assert 1 <= k <= self.model.config.moe_num_experts, \
    f"top_k must be between 1 and {self.model.config.moe_num_experts}"
```

这确保了：
- `k >= 1`：至少选择一个专家
- `k <= 64`：不超过专家总数

#### 延迟测量方法

1. **Prefill延迟**：
   - 使用forward hooks在第一个和最后一个transformer block上测量
   - 直接测量LLM prefill时间，避免减法误差

2. **Decode延迟**：
   - `T_LLM_decode = T_total - T_vision_total - T_LLM_prefill`
   - 通过生成10个新token测量总延迟

## 使用方法

### 基本用法

```bash
# 使用本地checkpoint路径（默认）
python experiments/profiling/knob2_topk/exp_moe_topk.py \
    --model_path checkpoints \
    --output_dir ./results/moe_topk \
    --num_samples 50

# 或者指定其他本地路径
python experiments/profiling/knob2_topk/exp_moe_topk.py \
    --model_path /path/to/checkpoint \
    --output_dir ./results/moe_topk \
    --num_samples 50
```

**注意**: `--model_path` 必须是本地checkpoint目录路径，不能使用 `hf:` 前缀。checkpoint目录应包含 `pytorch_model.bin` 或 `model.safetensors` 文件。

### 自定义Top-K值

修改`exp_moe_topk.py`中的`run`方法：

```python
experiment.run(
    num_samples=50,
    top_k_values=[1, 2, 4, 8, 16, 32]  # 自定义值
)
```

### 输出结果

结果保存在`output_dir/exp2_moe_topk_results.json`，格式如下：

```json
[
  {
    "top_k": 1,
    "prefill": {
      "P50": 123.45,
      "P95": 145.67,
      "P99": 156.78,
      "mean": 124.56,
      "std": 5.67,
      "min": 115.23,
      "max": 158.90
    },
    "decode": {
      "P50": 12.34,
      "P95": 14.56,
      "P99": 15.67,
      "mean": 12.45,
      "std": 0.56,
      "min": 11.23,
      "max": 15.89
    }
  },
  ...
]
```

## 预期结果分析

### 理论预期

1. **Prefill延迟**：
   - 应该随top_k线性或接近线性增长
   - 如果完全线性：`latency(k) ≈ latency(1) * k`
   - 实际可能由于内存带宽、并行度等因素而偏离线性

2. **Decode延迟**：
   - 每个token的decode延迟应该随top_k增长
   - 增长趋势可能与prefill类似

3. **性能瓶颈识别**：
   - 如果延迟增长远小于k倍：可能存在内存带宽瓶颈或GPU并行度优势
   - 如果延迟增长接近k倍：计算是主要瓶颈

### 关键指标

- **延迟缩放比**：`latency(k) / latency(1)` vs `k`
- **效率比**：`k / (latency(k) / latency(1))`（理想情况下为1）
- **Prefill vs Decode**：两个阶段的延迟增长趋势对比

## 代码检查结果

### ✅ 代码正确性

1. **Top-K修改逻辑**：
   - ✅ 正确更新config和所有MoE blocks
   - ✅ 包含范围验证（1 <= k <= 64）
   - ✅ 记录修改的block数量

2. **延迟测量**：
   - ✅ 使用forward hooks精确测量Prefill
   - ✅ 正确计算Decode延迟
   - ✅ 包含warmup阶段减少测量误差

3. **统计计算**：
   - ✅ 计算多个百分位数和统计量
   - ✅ 保存完整的统计信息

### ⚠️ 潜在改进点

1. **Top-K值选择**：
   - 当前默认值`[1, 2, 4, 8]`只覆盖到默认配置
   - 建议扩展到`[1, 2, 4, 8, 16, 32]`以更全面分析

2. **验证工具**：
   - 可以使用`experiments/profiling/utils/verify_moe_topk.py`验证top_k修改是否生效

3. **结果可视化**：
   - 建议添加可视化脚本绘制延迟vs top_k曲线

## 相关文件

- **实验脚本**: `experiments/profiling/knob2_topk/exp_moe_topk.py`
- **基础类**: `experiments/motivate/base_experiment.py`
- **验证工具**: `experiments/profiling/utils/verify_moe_topk.py`
- **FLOPs测量**: `experiments/profiling/utils/measure_flops_scaling.py`
- **模型配置**: `configs/model/config.json`
- **MoE实现**: `molmo/models/modeling_molmoe.py`

## 总结

### Top-K值选择评估

**结论：`[1, 2, 4, 8, 16, 32]`是合适的选择**

理由：
1. ✅ 覆盖了从最小到中等计算量的完整范围
2. ✅ 所有值都在有效范围内（1 <= k <= 64）
3. ✅ 2的幂次序列便于分析和可视化
4. ✅ 包含了模型默认值（k=8）
5. ✅ 避免了极端值（k=64），这在实践中很少使用

### 实验价值

1. **性能优化**：帮助确定最优的top_k值，平衡准确性和延迟
2. **瓶颈识别**：识别计算vs内存带宽瓶颈
3. **架构理解**：理解MoE模型在不同配置下的行为
4. **部署指导**：为实际部署提供性能参考

## 下一步建议

1. **运行实验**：使用建议的top_k值`[1, 2, 4, 8, 16, 32]`运行完整实验
2. **结果分析**：分析延迟缩放比和效率比
3. **可视化**：创建延迟vs top_k的可视化图表
4. **对比分析**：对比Prefill和Decode阶段的延迟增长趋势
5. **扩展实验**：如果需要，可以测试更多中间值（如k=6, 12, 24）

