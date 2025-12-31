# Tier Fallback 机制深度分析

## 当前实现

### 1. Tier Relaxation 机制（第452-519行）

当 tier 范围内的最佳匹配的 `mismatch > tier_relaxation_threshold` (默认 0.5) 时：
- 会搜索 tier 范围外的所有 crops (1 到 max_num_crops)
- 如果 tier 外的匹配明显更好（improvement > 0.2 且 mismatch < 0.3），就使用 tier 外的 crops
- 这会导致实际使用的 crops 数超出 tier 范围

### 2. 图像缩放机制（resize_and_pad）

```python
image_scale = min(image_scale_x, image_scale_y)  # 使用最小缩放比例
```

**关键特性**：
- 图像按最小缩放比例缩放，保持宽高比
- 然后通过 padding 填充到目标尺寸
- **不会造成严重失真**，因为：
  - 如果图像很大：会 downscale（缩小），不会丢失信息
  - 如果图像很小：会 upscale（放大），但使用 bilinear 插值，质量可接受
  - 如果 aspect ratio 不匹配：会 padding，不会扭曲图像

## 用户观点分析

### ✅ 观点1：Fallback 会造成 latency 预测不准

**完全正确！**

**原因**：
1. Latency 预测模型基于 tier 范围训练，假设 crops 数在 tier 范围内
2. 如果使用了 tier 外的 crops（例如 high tier 使用了 6 crops），实际 latency 会与 tier 预期不符
3. 这会导致：
   - Controller 的 latency 预测不准确
   - 无法正确满足 latency budget
   - 实验结果与预期偏差

**证据**：
- `acc_lat_profiling.py` 中记录的是 `actual_num_crops`，可能与 tier 范围不一致
- Latency 模型训练时假设 tier 内的 crops 分布，但实际可能超出范围

### ✅ 观点2：即使没有匹配，图像也会按最小放大比例缩放，不会有失真

**基本正确！**

**原因**：
1. `resize_and_pad` 使用 `min(scale_x, scale_y)`，保持宽高比
2. 即使 aspect ratio 不匹配，也只是 padding，不会扭曲图像
3. 对于大图像：downscale 不会丢失信息（只是分辨率降低）
4. 对于小图像：upscale 使用 bilinear 插值，质量可接受

**潜在问题**：
- 如果 mismatch 很大（> 0.5），padding 会很多，可能浪费计算资源
- 但这不是"失真"问题，而是效率问题

### ✅ 观点3：是否不需要 fallback 到 tier 外？

**建议：移除 tier 外的 fallback 机制**

**理由**：

#### 1. **Latency 预测准确性优先**
- Tier 机制的核心目的是控制 latency
- 如果允许 tier 外选择，latency 预测就失去了意义
- 实验设计需要可预测的 latency 范围

#### 2. **图像质量不是主要问题**
- `resize_and_pad` 已经保证了不会严重失真
- 即使 aspect ratio 不匹配，也只是 padding，不影响图像内容
- 大 mismatch 只是效率问题，不是质量问题

#### 3. **Tier 设计应该覆盖常见情况**
- 如果某个 tier 经常需要 fallback，说明 tier 设计不合理
- 应该调整 tier 范围，而不是允许 fallback

#### 4. **简化逻辑，提高可预测性**
- 移除 fallback 后，逻辑更简单
- 每个 tier 的行为更可预测
- 更容易调试和分析

## 建议的修改方案

### 方案1：完全移除 tier 外搜索（推荐）

```python
# 移除第452-519行的 tier relaxation 逻辑
# 只保留 tier 范围内的选择
if best_tiling is not None:
    # 直接使用 tier 范围内的最佳匹配，即使 mismatch 较大
    return best_tiling
else:
    # Fallback: 使用 tier 内的最小 crops
    fallback_crops = min(min_crops, max_num_crops)
    ...
```

**优点**：
- 保证 latency 预测准确性
- 逻辑简单清晰
- 行为可预测

**缺点**：
- 某些极端 aspect ratio 的图像可能 padding 较多
- 但这不是质量问题，只是效率问题

### 方案2：保留 tier 外搜索，但仅用于警告（不推荐）

```python
# 搜索 tier 外，但只用于记录警告，不实际使用
if best_mismatch > tier_relaxation_threshold:
    # 搜索 tier 外，但只用于日志
    # 仍然使用 tier 内的最佳匹配
    log.warning(f"Tier {tier_name} mismatch is large ({best_mismatch:.4f}), "
                f"but using tier range to maintain latency predictability")
    return best_tiling
```

**优点**：
- 可以监控 tier 设计是否合理

**缺点**：
- 增加了代码复杂度
- 没有实际作用

## 结论

**建议移除 tier 外的 fallback 机制**，原因：

1. ✅ **Latency 预测准确性是核心需求**：Tier 机制的目的是控制 latency，如果允许 tier 外选择，就失去了这个意义
2. ✅ **图像质量不是问题**：`resize_and_pad` 已经保证了不会严重失真
3. ✅ **简化逻辑**：移除 fallback 后，代码更简单，行为更可预测
4. ✅ **Tier 设计应该合理**：如果经常需要 fallback，应该调整 tier 范围，而不是允许 fallback

**如果发现某些 tier 经常出现大 mismatch，应该**：
- 分析原因（aspect ratio 分布？）
- 调整 tier 的 `preferred_crops` 或范围
- 而不是允许 fallback 到 tier 外

