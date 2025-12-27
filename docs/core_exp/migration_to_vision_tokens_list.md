# Migration to vision_tokens_list: 修改总结

## 概述

已将实验脚本从使用 `image_size_list` 改为使用 `vision_tokens_list`，以充分利用 `select_tiling` 的自适应机制，根据每个图像的 aspect ratio 自动选择最合适的 tiling，避免图像变形。

## 修改的文件

### 1. `experiments/core_exp/run_multi_datasets_h100.sh`

**修改内容**：
- 将 `IMAGE_SIZE_LIST` 改为 `VISION_TOKENS_LIST`
- 默认值：`432 720 1008 1440` (对应 2, 4, 6, 9 crops)
- 更新所有相关的日志输出和注释
- 更新命令行参数：`--image_size_list` → `--vision_tokens_list`

**关键变化**：
```bash
# 之前
IMAGE_SIZE_LIST="${IMAGE_SIZE_LIST:-560x784 784x784}"
--image_size_list ${IMAGE_SIZE_LIST}

# 现在
VISION_TOKENS_LIST="${VISION_TOKENS_LIST:-432 720 1008 1440}"
--vision_tokens_list ${VISION_TOKENS_LIST}
```

### 2. `experiments/core_exp/run_multi_datasets_a100.sh`

**修改内容**：
- 与 `run_multi_datasets_h100.sh` 相同的修改
- 保持 A100 特定的内存优化设置

### 3. `experiments/core_exp/combined_profiling.py`

**修改内容**：

#### a. `_generate_config_filename` 函数
- 支持 `vision_tokens_list` 模式的文件名生成
- 格式：`<task_name>_visiontoken<T>_topk<k>_blocks<n>.json`
- 保持向后兼容 `image_size_list` 模式

**逻辑**：
```python
# vision_tokens_list 模式：使用 tokens<T>
if target_vision_tokens is not None:
    img_size_str = f"tokens{target_vision_tokens}"

# image_size_list 模式：使用 <H>x<W>
elif target_image_size:
    img_size_str = f"{img_h}x{img_w}"
```

#### b. 日志输出更新
- 将 `vision_tokens_list` 标记为 "recommended knob"
- 将 `image_size_list` 标记为 "legacy knob"
- 添加说明：`select_tiling` 会根据每个图像的 aspect ratio 自适应选择

#### c. 参数帮助信息更新
- 更新 `--vision_tokens_list` 的帮助信息，说明其优势
- 更新 `--image_size_list` 的帮助信息，标记为 legacy

#### d. 类文档字符串更新
- 更新 `CombinedProfilingExperiment` 的文档字符串
- 说明两种模式的区别和推荐使用 `vision_tokens_list`

#### e. Glob 模式更新
- 更新文件清理逻辑，支持新的文件名格式
- 更新日志输出，说明两种文件名格式

## 工作原理

### vision_tokens_list 模式（推荐）

1. **指定目标 vision tokens**：例如 `1008` (6 crops)
2. **计算 num_crops**：`(1008 // 144) - 1 = 6`
3. **对于每个图像**：
   - `select_tiling` 根据原始图像的 aspect ratio 自动选择最合适的 tiling
   - 例如：tall image (aspect 0.75) → 选择 (3,2) tiling
   - 例如：wide image (aspect 1.33) → 选择 (2,3) tiling
4. **自适应 resize**：根据选择的 tiling resize 图像，最小化变形

### image_size_list 模式（legacy）

1. **指定固定尺寸**：例如 `560x784`
2. **推断固定 tiling**：`2×3` (6 crops)
3. **所有图像**都被强制 resize 到这个固定尺寸
4. **问题**：可能导致 aspect ratio 不匹配，图像变形

## 文件名格式

### vision_tokens_list 模式
```
coco-2014-vqa_visiontoken1008_topk8_blocks14.json
```

### image_size_list 模式（向后兼容）
```
coco-2014-vqa_imgsize560x784_topk8_blocks14.json
```

## 实验结果保存

### 结果文件结构

每个配置的结果保存在单独的文件中：
- **文件名**：包含所有控制参数（vision_tokens/image_size, top_k, num_active_blocks）
- **内容**：包含所有样本的详细结果和聚合统计

### 关键字段

```json
{
  "target_vision_tokens": 1008,
  "target_image_size": null,  // vision_tokens_list 模式下为 null
  "theoretical_image_size": [784, 560],  // 根据第一个样本计算（可能变化）
  "num_crops": 6,
  "top_k": 8,
  "num_active_blocks": 14,
  "per_sample_results": [...],  // 每个样本的详细结果
  "aggregate_stats": {...}  // 聚合统计
}
```

**注意**：在 `vision_tokens_list` 模式下，`theoretical_image_size` 可能因样本而异（因为每个图像的 aspect ratio 不同），这是正常的。文件名使用 `target_vision_tokens` 来标识配置。

## 向后兼容性

- `image_size_list` 仍然支持（标记为 legacy）
- 如果同时提供两个参数，`image_size_list` 优先
- 文件名生成支持两种模式

## 验证

### 检查点

1. ✅ 脚本可以正常运行
2. ✅ 文件名格式正确
3. ✅ 日志输出清晰
4. ✅ 结果保存完整
5. ✅ 向后兼容 `image_size_list`

### 测试建议

```bash
# 测试 vision_tokens_list 模式
VISION_TOKENS_LIST="432 720" NUM_SAMPLES=10 \
  bash experiments/core_exp/run_multi_datasets_h100.sh coco_2014_vqa

# 验证结果文件
ls -la results/core_exp_h100/coco-2014-vqa/*.json
# 应该看到：coco-2014-vqa_visiontoken432_*.json
#         coco-2014-vqa_visiontoken720_*.json
```

## 优势总结

1. **自适应 aspect ratio**：每个图像根据其原始 aspect ratio 选择最合适的 tiling
2. **最小化图像变形**：避免强制 resize 导致的图像变形
3. **充分利用 select_tiling**：发挥其自动优化机制
4. **配置更简单**：只需指定 vision tokens 数量，无需手动选择多个尺寸
5. **实验设计更灵活**：自动适应不同数据集的图像尺寸分布

## 相关文档

- `docs/core_exp/vision_tokens_list_vs_image_size_list.md`：详细的工作原理对比
- `docs/core_exp/image_size_selection_analysis.md`：问题分析和推荐方案
- `docs/knobs/vision_tokens_knob.md`：Vision tokens 控制 knob 的详细文档

