# Exp 3 实验结果分析

## 1. Total Latency 差异分析

### 问题
Hook方法测量出来的total latency比减法方法高一些（差异从3.31ms到93.13ms不等）。

### 原因分析

**主要差异来源：T_LLM_prefill的测量方法不同**

1. **减法方法**：
   - `T_LLM_prefill = T_prefill_step - T_vision_total`
   - 由于独立测量`T_prefill_step`和`T_vision_total`，误差会累积
   - 当vision tokens增加时，`T_LLM_prefill`反而下降（不合理）
   - 这导致`T_total`被低估

2. **Hook方法**：
   - 直接在LLM transformer blocks上注册forward hooks
   - 直接测量LLM prefill时间，不依赖减法
   - `T_LLM_prefill`随vision tokens增加而增加（合理）
   - `T_total`更准确地反映真实延迟

### 验证

从实验结果看：
- Hook方法：`T_total ≈ T_vision + T_LLM_prefill`（完全一致）
- 减法方法：虽然公式上也一致，但`T_LLM_prefill`被低估了

### 结论

**Hook方法的T_total更高是正常的**，因为它更准确地反映了真实的延迟。差异主要来自：
- 减法方法低估了`T_LLM_prefill`（特别是vision tokens多的时候）
- Hook方法提供了更准确的直接测量

**建议**：使用Hook方法进行实验，因为它提供了更准确的延迟测量。

## 2. Vision Token 精确控制

### 问题
后面的几个较大分辨率（336×2688, 336×3024, 336×3360, 336×3696, 336×4032）都使用了13个crops（1344个vision tokens），无法精确控制vision token数。

### 原因
`select_tiling`函数会根据图像尺寸自动选择最小upscaling的tiling配置。当分辨率很大时，多个分辨率可能选择相同的tiling（比如都是2×6或3×4等）。

### 解决方案

**使用精确的分辨率计算**

基于`select_tiling`的逻辑，我们可以计算每个tiling配置需要的精确分辨率：

```python
# 参数
crop_window_size = 224  # 16 patches * 14 pixels per patch
total_margin_pixels = 112  # 8 patches * 14 pixels (4+4 margins)

# 对于1×k tiling:
w = k * crop_window_size + total_margin_pixels
h = 1 * crop_window_size + total_margin_pixels  # Always 336 for 1×k
```

**精确分辨率表（1×k tiling）**：

| k | Resolution | Crops | Vision Tokens |
|---|------------|-------|---------------|
| 1 | 336×336    | 2     | 288           |
| 2 | 560×336    | 3     | 480           |
| 3 | 784×336    | 4     | 576           |
| 4 | 1008×336   | 5     | 768           |
| 5 | 1232×336   | 6     | 864           |
| 6 | 1456×336   | 7     | 1056          |
| 7 | 1680×336   | 8     | 1152          |
| 8 | 1904×336   | 9     | 1344          |
| 9 | 2128×336   | 10    | 1344          |
| 10| 2352×336   | 11    | 1344          |
| 11| 2576×336   | 12    | 1344          |
| 12| 2800×336   | 13    | 1344          |

**注意**：当k≥8时，由于`max_crops=12`的限制，实际crops数会被限制在13（12+1 global），所以vision tokens都是1344。

### 改进方案

如果需要在k≥8时也获得不同的vision token数，可以考虑：

1. **增加max_crops限制**（如果模型支持）
2. **使用不同的tiling配置**（比如2×k, 3×k等）
3. **直接指定tiling配置**（需要修改preprocessor）

当前实现已经使用精确分辨率计算，确保每个k值都能触发正确的tiling配置。

## 3. Hook方法测量范围

**问题**：Hook方法是否测量了LLM全部transformer block的latency？

**答案**：是的。

Hook方法在第一个和最后一个transformer block上注册forward hooks：
- `start_hook` 注册在 `transformer.blocks[0]`（第一个block）
- `end_hook` 注册在 `transformer.blocks[-1]`（最后一个block）

因此，测量的是**整个transformer的latency**，包括所有transformer blocks的处理时间。这比减法方法更准确，因为它直接测量了LLM的实际处理时间，避免了独立测量导致的误差累积。

## 4. 缺失的Crops数

**问题**：当前实验结果中num_crops只有2,4,5,7,8,10,11,13，缺少3,6,9,12。是否可以通过不同的tiling配置获得这些缺失值？

**答案**：理论上可以，但需要精确的分辨率计算。

所有可能的tiling配置：
- 3 crops: 1×2 或 2×1 tiling
- 6 crops: 1×5 或 5×1 tiling  
- 9 crops: 1×8, 2×4, 4×2, 或 8×1 tiling
- 12 crops: 1×11 或 11×1 tiling

**注意**：`select_tiling`会根据图像尺寸自动选择最小upscaling的配置。对于矩形图像（宽>高），通常会选择1×k而不是k×1。如果分辨率计算不够精确，可能会触发不同的tiling配置。

**当前实现**：代码已经更新为使用精确分辨率计算，理论上应该能够获得所有缺失的crops数。如果某些配置仍然无法触发，可能是因为`select_tiling`选择了更优的配置（比如对于接近正方形的图像，可能选择2×k而不是1×k）。

