# Image Size Selection Analysis and Recommendations

## 问题分析

### 当前实现的问题

当前脚本使用固定的 `image_size_list` (如 `560x784 784x784`)，存在以下问题：

1. **Aspect Ratio 不匹配**：
   - 固定尺寸会强制所有图像 resize 到这些尺寸
   - 但 `resize_and_pad` 函数会 preserve aspect ratio（通过 padding）
   - 这导致实际使用的 tiling 可能与预期不符

2. **覆盖不全面**：
   - 当前只有 2 个尺寸：`560x784` (aspect 1.4) 和 `784x784` (aspect 1.0)
   - 缺少 tall images (aspect < 0.8) 的覆盖
   - COCO 数据集中有很多 tall images (aspect ~0.75)

3. **与 `exact_num_crops` 的冲突**：
   - `exact_num_crops` 会强制选择特定的 tiling
   - 但 `select_tiling` 会根据原始图像的 aspect ratio 来选择最接近的 tiling
   - 这可能导致 aspect ratio 不匹配，影响模型性能

### 代码行为分析

#### 1. `resize_and_pad` 函数（preserve aspect ratio）

```python
# molmo/data/model_preprocessor.py:54-112
def resize_and_pad(image, desired_output_size, ...):
    """Resize an image while padding to preserve its aspect ratio."""
    image_scale = min(image_scale_x, image_scale_y)  # 保持 aspect ratio
    # ... resize with padding
```

**关键点**：即使指定了 `desired_output_size`，实际 resize 的尺寸也会根据原始图像的 aspect ratio 进行调整（通过 padding）。

#### 2. `select_tiling` 函数（考虑 aspect ratio）

```python
# molmo/data/model_preprocessor.py:202-276
def select_tiling(h, w, patch_size, max_num_crops, exact_num_crops=None):
    if exact_num_crops is not None:
        # 在所有可能的 tiling 中，选择最接近原始图像 aspect ratio 的
        aspect_ratio = w / h if h > 0 else 1.0
        # ... 选择 best_tiling
```

**关键点**：当 `exact_num_crops` 设置时，`select_tiling` 会在所有可能的 tiling 中选择最接近原始图像 aspect ratio 的配置。

#### 3. 实际 resize 尺寸计算

```python
# molmo/data/model_preprocessor.py:454-459
tiling = select_tiling(...)
src, img_mask = self.resize_image(
    image,
    [tiling[0]*crop_window_size+total_margin_pixels, 
     tiling[1]*crop_window_size+total_margin_pixels],
    ...
)
```

**关键点**：实际 resize 的尺寸是 `tiling[0]*crop_window_size+total_margin_pixels`，而不是 `image_size_list` 中指定的尺寸。

## 推荐方案

### 方案 1: 使用 `vision_tokens_list`（推荐）

**优点**：
- 让 `select_tiling` 根据原始图像的 aspect ratio 自动选择最合适的 tiling
- 避免 aspect ratio 不匹配的问题
- 更符合 `select_tiling` 的设计初衷（最小化 upscaling）

**实现**：
```bash
# 不使用 image_size_list，改用 vision_tokens_list
VISION_TOKENS_LIST="${VISION_TOKENS_LIST:-432 720 1008 1440}"
```

**对应关系**：
- 432 tokens → 2 crops (适合 wide/tall images)
- 720 tokens → 4 crops (适合 square images)
- 1008 tokens → 6 crops (适合 wide images)
- 1440 tokens → 9 crops (适合 square images)

### 方案 2: 改进 `image_size_list`（如果必须使用）

**原则**：
1. 覆盖常见的 aspect ratio 范围
2. 确保每个 aspect ratio 都有对应的尺寸
3. 选择合理的 tiling 配置

**推荐的尺寸列表**：
```bash
# 覆盖 wide, square, tall 三种 aspect ratio
IMAGE_SIZE_LIST="${IMAGE_SIZE_LIST:-560x336 560x560 560x784 784x560 784x784}"
```

**对应关系**：
- `560x336` (aspect 0.6, tall) → tiling 2x1 → 2 crops → 432 tokens
- `560x560` (aspect 1.0, square) → tiling 2x2 → 4 crops → 720 tokens
- `560x784` (aspect 1.4, wide) → tiling 2x3 → 6 crops → 1008 tokens
- `784x560` (aspect 0.71, tall) → tiling 3x2 → 6 crops → 1008 tokens
- `784x784` (aspect 1.0, square) → tiling 3x3 → 9 crops → 1440 tokens

### 方案 3: 自适应选择（最理想，但需要代码修改）

**思路**：
- 根据原始图像的 aspect ratio，自动选择最接近的 `image_size_list` 中的尺寸
- 或者，根据 aspect ratio 范围，动态选择 tiling

**实现**（需要修改代码）：
```python
def select_best_image_size(original_aspect_ratio, image_size_list):
    """根据原始图像的 aspect ratio，选择最接近的 image_size"""
    best_size = None
    best_match = float('inf')
    
    for size in image_size_list:
        h, w = map(int, size.split('x'))
        target_aspect = w / h
        aspect_diff = abs(target_aspect - original_aspect_ratio)
        if aspect_diff < best_match:
            best_match = aspect_diff
            best_size = size
    
    return best_size
```

## 当前配置的问题

### 当前配置
```bash
IMAGE_SIZE_LIST="${IMAGE_SIZE_LIST:-560x784 784x784}"
```

### 问题分析

1. **缺少 tall images 覆盖**：
   - 只有 `560x784` (aspect 1.4) 和 `784x784` (aspect 1.0)
   - 缺少 aspect < 0.8 的尺寸（如 `784x560`）

2. **Aspect ratio 分布不均**：
   - COCO 数据集中有很多 images 的 aspect ratio 在 0.7-0.8 之间
   - 当前配置无法很好地处理这些 images

3. **Tiling 选择可能不优**：
   - 对于 tall images，`select_tiling` 可能选择 `3x2` tiling（6 crops）
   - 但 `exact_num_crops` 可能强制选择其他 tiling，导致 aspect ratio 不匹配

## 具体建议

### 短期方案（最小改动）

**修改 `run_multi_datasets_h100.sh`**：
```bash
# 添加更多尺寸以覆盖不同的 aspect ratio
IMAGE_SIZE_LIST="${IMAGE_SIZE_LIST:-560x336 560x560 560x784 784x560 784x784}"
```

**理由**：
- `560x336` (aspect 0.6): 覆盖 tall images
- `560x560` (aspect 1.0): 覆盖 square images（较小）
- `560x784` (aspect 1.4): 覆盖 wide images
- `784x560` (aspect 0.71): 覆盖 tall images（较大）
- `784x784` (aspect 1.0): 覆盖 square images（较大）

### 长期方案（推荐）

**改用 `vision_tokens_list`**：
```bash
# 不使用 image_size_list，改用 vision_tokens_list
VISION_TOKENS_LIST="${VISION_TOKENS_LIST:-432 720 1008 1440}"
```

**理由**：
- 让 `select_tiling` 根据原始图像的 aspect ratio 自动选择最合适的 tiling
- 避免 aspect ratio 不匹配的问题
- 更符合 `select_tiling` 的设计初衷

## 实验验证

### 验证方法

1. **统计数据集中的 aspect ratio 分布**：
   ```python
   # 分析 COCO 数据集的 aspect ratio 分布
   aspect_ratios = []
   for image in dataset:
       w, h = image.size
       aspect_ratios.append(w / h)
   
   # 绘制分布图
   import matplotlib.pyplot as plt
   plt.hist(aspect_ratios, bins=50)
   plt.xlabel('Aspect Ratio')
   plt.ylabel('Frequency')
   plt.show()
   ```

2. **比较不同配置的性能**：
   - 使用 `image_size_list` vs `vision_tokens_list`
   - 比较 accuracy 和 latency
   - 分析 aspect ratio 不匹配的影响

3. **检查实际使用的 tiling**：
   - 记录每个样本的实际 tiling
   - 分析是否与预期一致

## 总结

### 核心问题
当前使用固定的 `image_size_list` 会导致：
1. Aspect ratio 不匹配（特别是 tall images）
2. 覆盖不全面（缺少某些 aspect ratio 范围）
3. 与 `select_tiling` 的设计理念冲突

### 推荐方案
1. **短期**：扩展 `image_size_list` 以覆盖更多 aspect ratio 范围
2. **长期**：改用 `vision_tokens_list`，让 `select_tiling` 自动选择最合适的 tiling

### 关键原则
- **Preserve aspect ratio**：避免图像变形
- **Minimize upscaling**：减少图像质量损失
- **Cover common ranges**：确保常见 aspect ratio 都有对应配置

