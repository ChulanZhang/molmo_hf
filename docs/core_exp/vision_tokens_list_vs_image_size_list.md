# Vision Tokens List vs Image Size List: 工作原理对比

## 核心区别

### `image_size_list` 方法（当前使用）

**工作流程**：
```
固定尺寸 (H×W) 
  → 固定 tiling (rows×cols) 
  → 固定 num_crops 
  → 固定 vision_tokens
  → 强制所有图像 resize 到这个尺寸
```

**问题**：
- 所有图像都被强制 resize 到固定尺寸，不管原始 aspect ratio
- 可能导致图像变形（特别是 tall/wide images）
- `select_tiling` 的自动选择机制被 `exact_num_crops` 限制

### `vision_tokens_list` 方法（推荐）

**工作流程**：
```
目标 vision_tokens 
  → 计算 num_crops 
  → 根据原始图像 aspect ratio 自动选择 tiling
  → 自适应 resize 到最佳尺寸
```

**优点**：
- 根据原始图像的 aspect ratio 自动选择最合适的 tiling
- 最小化图像变形（preserve aspect ratio）
- 充分利用 `select_tiling` 的自动优化机制

## 详细工作流程对比

### 方法 1: `image_size_list` 工作流程

#### 步骤 1: 解析 image_size_list
```python
# experiments/core_exp/combined_profiling.py:807-828
if image_size_list:
    image_specs = []
    for sz in image_size_list:  # e.g., "560x784"
        target_h, target_w = int(h_str), int(w_str)  # 560, 784
        
        # 从固定尺寸推断 tiling
        rows, cols = image_size_to_tiling(target_h, target_w)
        # rows = round((560 - 112) / 224) = 2
        # cols = round((784 - 112) / 224) = 3
        
        num_crops = rows * cols  # 2 * 3 = 6
        target_tokens = (num_crops + 1) * 144  # (6 + 1) * 144 = 1008
```

**关键点**：tiling 是**固定的**，由 `image_size_list` 中的尺寸决定。

#### 步骤 2: 设置 exact_num_crops
```python
# experiments/core_exp/combined_profiling.py:977
mm_preprocessor = MultiModalPreprocessor(
    exact_num_crops=num_crops,  # 强制使用 6 crops
    ...
)
```

#### 步骤 3: 图像预处理（每个样本）
```python
# molmo/data/model_preprocessor.py:447-459
# 对于每个图像：
original_image_h, original_image_w = image.shape[:2]  # 例如: 640×480

# select_tiling 被 exact_num_crops 限制
tiling = select_tiling(
    original_image_h - total_margin_pixels,  # 640 - 112 = 528
    original_image_w - total_margin_pixels,  # 480 - 112 = 368
    crop_window_size=224,
    max_crops=6,
    exact_num_crops=6  # 强制使用 6 crops
)

# select_tiling 内部逻辑（当 exact_num_crops 设置时）：
# 1. 找到所有可能的 6-crop tilings: [(1,6), (2,3), (3,2), (6,1)]
# 2. 选择最接近原始图像 aspect ratio 的: (2,3) 或 (3,2)
# 3. 但 aspect ratio 可能仍然不匹配！

# 然后 resize 到：
resize_size = [
    tiling[0] * crop_window_size + total_margin_pixels,  # 2 * 224 + 112 = 560
    tiling[1] * crop_window_size + total_margin_pixels   # 3 * 224 + 112 = 784
]
# 实际 resize: 640×480 → 560×784 (aspect ratio 从 0.75 变成 1.4！)
```

**问题**：
- 原始图像 640×480 (aspect 0.75, tall)
- 目标尺寸 560×784 (aspect 1.4, wide)
- **严重不匹配**！即使 `select_tiling` 选择了 (3,2) tiling，resize 后的尺寸仍然是 560×784

#### 步骤 4: resize_and_pad
```python
# molmo/data/model_preprocessor.py:54-112
def resize_and_pad(image, desired_output_size, ...):
    # desired_output_size = (560, 784)
    # 原始图像: 640×480 (aspect 0.75)
    
    image_scale = min(image_scale_x, image_scale_y)  # preserve aspect ratio
    # scale_x = 784 / 480 = 1.63
    # scale_y = 560 / 640 = 0.875
    # image_scale = 0.875 (使用较小的 scale)
    
    scaled_height = 640 * 0.875 = 560  # ✓ 匹配
    scaled_width = 480 * 0.875 = 420  # ✗ 不匹配！应该是 784
    
    # 然后 padding: (0, 0, 182, 0)  # 左右各 padding 182 pixels
    # 最终: 560×784，但图像被严重拉伸/压缩
```

**结果**：图像被强制 resize 到 560×784，但原始 aspect ratio 是 0.75，导致：
- 图像变形（stretching/squashing）
- 大量 padding（182 pixels on each side）
- 实际使用的 vision tokens 可能少于预期（padding 区域被标记为无效）

### 方法 2: `vision_tokens_list` 工作流程

#### 步骤 1: 解析 vision_tokens_list
```python
# experiments/core_exp/combined_profiling.py:839-845
else:
    # 使用 vision_tokens_list
    if vision_tokens_list is None:
        vision_tokens_list = [432, 720, 1008, 1296, 1584]
    
    # 对于每个 target_tokens (e.g., 1008):
    num_crops = tokens_to_crops(target_tokens)  # (1008 // 144) - 1 = 6
```

**关键点**：只指定目标 vision tokens，不指定具体尺寸。

#### 步骤 2: 设置 exact_num_crops
```python
# experiments/core_exp/combined_profiling.py:977
mm_preprocessor = MultiModalPreprocessor(
    exact_num_crops=num_crops,  # 强制使用 6 crops
    ...
)
```

#### 步骤 3: 图像预处理（每个样本，自适应）
```python
# molmo/data/model_preprocessor.py:447-459
# 对于每个图像：
original_image_h, original_image_w = image.shape[:2]  # 例如: 640×480

# select_tiling 根据原始图像 aspect ratio 自动选择
tiling = select_tiling(
    original_image_h - total_margin_pixels,  # 640 - 112 = 528
    original_image_w - total_margin_pixels,  # 480 - 112 = 368
    crop_window_size=224,
    max_crops=6,
    exact_num_crops=6  # 强制使用 6 crops，但可以选择 tiling
)

# select_tiling 内部逻辑（当 exact_num_crops 设置时）：
original_aspect_ratio = 368 / 528 = 0.697  # tall image

# 1. 找到所有可能的 6-crop tilings: [(1,6), (2,3), (3,2), (6,1)]
# 2. 计算每个 tiling 的 aspect ratio:
#    - (1,6): 6*224 / 1*224 = 6.0 (太宽)
#    - (2,3): 3*224 / 2*224 = 1.5 (太宽)
#    - (3,2): 2*224 / 3*224 = 0.67 (最接近 0.697！) ✓
#    - (6,1): 1*224 / 6*224 = 0.17 (太窄)
# 3. 选择 (3,2) tiling（最接近原始 aspect ratio）

# 然后 resize 到：
resize_size = [
    tiling[0] * crop_window_size + total_margin_pixels,  # 3 * 224 + 112 = 784
    tiling[1] * crop_window_size + total_margin_pixels   # 2 * 224 + 112 = 560
]
# 实际 resize: 640×480 → 784×560 (aspect ratio 从 0.75 变成 0.71，很接近！)
```

**优点**：
- 原始图像 640×480 (aspect 0.75, tall)
- 选择的 tiling: (3,2)
- 目标尺寸 784×560 (aspect 0.71, tall)
- **Aspect ratio 匹配良好**！图像变形最小。

#### 步骤 4: resize_and_pad
```python
# molmo/data/model_preprocessor.py:54-112
def resize_and_pad(image, desired_output_size, ...):
    # desired_output_size = (784, 560)
    # 原始图像: 640×480 (aspect 0.75)
    
    image_scale = min(image_scale_x, image_scale_y)  # preserve aspect ratio
    # scale_x = 560 / 480 = 1.167
    # scale_y = 784 / 640 = 1.225
    # image_scale = 1.167 (使用较小的 scale)
    
    scaled_height = 640 * 1.167 = 747  # 接近 784
    scaled_width = 480 * 1.167 = 560   # ✓ 匹配
    
    # 然后 padding: (18, 19, 0, 0)  # 上下各 padding ~18 pixels
    # 最终: 784×560，图像变形很小
```

**结果**：图像被 resize 到 784×560，aspect ratio 从 0.75 变成 0.71，匹配良好：
- 图像变形最小
- Padding 很少（只有 ~18 pixels）
- 实际使用的 vision tokens 接近预期

## 关键代码位置

### 1. `select_tiling` 函数（核心逻辑）

```python
# molmo/data/model_preprocessor.py:202-276
def select_tiling(h, w, patch_size, max_num_crops, exact_num_crops=None):
    if exact_num_crops is not None:
        # 找到所有可能的 exact_num_crops tilings
        exact_tilings = []
        for i in range(1, exact_num_crops + 1):
            if exact_num_crops % i == 0:
                j = exact_num_crops // i
                exact_tilings.append((i, j))
        
        # 关键：根据原始图像 aspect ratio 选择最接近的 tiling
        aspect_ratio = w / h if h > 0 else 1.0
        best_tiling = None
        best_match = float('inf')
        
        for i, j in exact_tilings:
            tiling_aspect = (j * patch_size) / (i * patch_size)
            aspect_diff = abs(tiling_aspect - aspect_ratio)
            if aspect_diff < best_match:
                best_match = aspect_diff
                best_tiling = (i, j)
        
        return best_tiling  # 返回最匹配的 tiling
```

**关键点**：即使设置了 `exact_num_crops`，`select_tiling` 仍然会根据原始图像的 aspect ratio 选择最合适的 tiling。

### 2. `image_size_list` vs `vision_tokens_list` 的区别

#### `image_size_list` 路径
```python
# experiments/core_exp/combined_profiling.py:807-828
if image_size_list:
    # 从固定尺寸推断 tiling
    rows, cols = image_size_to_tiling(target_h, target_w)
    # 这个 tiling 是固定的，不管原始图像是什么
    num_crops = rows * cols
    target_tokens = (num_crops + 1) * 144
```

**问题**：tiling 是固定的，但实际 resize 时，`select_tiling` 可能会选择不同的 tiling（如果 `exact_num_crops` 允许），导致不一致。

#### `vision_tokens_list` 路径
```python
# experiments/core_exp/combined_profiling.py:839-845
else:
    # 从 vision tokens 计算 num_crops
    num_crops = tokens_to_crops(target_tokens)
    # tiling 会在预处理时根据原始图像自动选择
```

**优点**：tiling 是根据原始图像的 aspect ratio 自动选择的，确保最佳匹配。

## 实际例子对比

### 例子 1: Tall Image (640×480, aspect 0.75)

#### 使用 `image_size_list = ["560x784"]`
```
目标尺寸: 560×784 (aspect 1.4)
  → 固定 tiling: 2×3 (6 crops)
  → exact_num_crops = 6
  → select_tiling 可能选择 (3,2) 或 (2,3)
  → 如果选择 (3,2): resize 到 784×560
  → 如果选择 (2,3): resize 到 560×784
  → 结果：不一致！取决于 select_tiling 的选择
```

#### 使用 `vision_tokens_list = [1008]`
```
目标 tokens: 1008 (6 crops)
  → num_crops = 6
  → exact_num_crops = 6
  → select_tiling 根据原始图像 (640×480, aspect 0.75) 选择
  → 选择 (3,2) tiling (aspect 0.67，最接近 0.75)
  → resize 到 784×560 (aspect 0.71)
  → 结果：一致！aspect ratio 匹配良好
```

### 例子 2: Wide Image (1024×768, aspect 1.33)

#### 使用 `image_size_list = ["560x784"]`
```
目标尺寸: 560×784 (aspect 1.4)
  → 固定 tiling: 2×3 (6 crops)
  → exact_num_crops = 6
  → select_tiling 根据原始图像 (1024×768, aspect 1.33) 选择
  → 选择 (2,3) tiling (aspect 1.5，最接近 1.33)
  → resize 到 560×784 (aspect 1.4)
  → 结果：还可以，但 aspect ratio 仍有差异
```

#### 使用 `vision_tokens_list = [1008]`
```
目标 tokens: 1008 (6 crops)
  → num_crops = 6
  → exact_num_crops = 6
  → select_tiling 根据原始图像 (1024×768, aspect 1.33) 选择
  → 选择 (2,3) tiling (aspect 1.5，最接近 1.33)
  → resize 到 560×784 (aspect 1.4)
  → 结果：与 image_size_list 相同，但逻辑更清晰
```

### 例子 3: Square Image (512×512, aspect 1.0)

#### 使用 `image_size_list = ["560x784"]`
```
目标尺寸: 560×784 (aspect 1.4)
  → 固定 tiling: 2×3 (6 crops)
  → exact_num_crops = 6
  → select_tiling 根据原始图像 (512×512, aspect 1.0) 选择
  → 选择 (2,3) 或 (3,2) tiling（都不匹配 square）
  → resize 到 560×784 或 784×560
  → 结果：严重不匹配！square image 被强制变成 wide/tall
```

#### 使用 `vision_tokens_list = [1008]`
```
目标 tokens: 1008 (6 crops)
  → num_crops = 6
  → exact_num_crops = 6
  → select_tiling 根据原始图像 (512×512, aspect 1.0) 选择
  → 选择 (2,3) 或 (3,2) tiling（都不匹配 square）
  → resize 到 560×784 或 784×560
  → 结果：同样不匹配，但这是 6 crops 的限制
  → 如果使用 720 tokens (4 crops)，可以选择 (2,2) tiling，完美匹配！
```

## 为什么 `vision_tokens_list` 更好？

### 1. **自适应 Aspect Ratio**
- `vision_tokens_list` 让 `select_tiling` 根据原始图像的 aspect ratio 自动选择最合适的 tiling
- `image_size_list` 固定了 tiling，可能导致 aspect ratio 不匹配

### 2. **更符合设计理念**
- `select_tiling` 的设计目标就是"最小化 upscaling，保持 aspect ratio"
- `vision_tokens_list` 充分利用了这个机制
- `image_size_list` 限制了 `select_tiling` 的灵活性

### 3. **更灵活的实验设计**
- 可以只关注 vision tokens 数量（这是影响性能的关键因素）
- 不需要手动选择多个 image sizes 来覆盖不同的 aspect ratio
- 自动适应不同数据集的图像尺寸分布

### 4. **减少配置复杂性**
- `vision_tokens_list = [432, 720, 1008, 1440]` 简单明了
- `image_size_list` 需要选择多个尺寸来覆盖不同 aspect ratio

## 推荐配置

### 使用 `vision_tokens_list`（推荐）
```bash
# 不使用 image_size_list
# IMAGE_SIZE_LIST=""

# 使用 vision_tokens_list
VISION_TOKENS_LIST="${VISION_TOKENS_LIST:-432 720 1008 1440}"
```

**对应关系**：
- 432 tokens → 2 crops (适合 small images)
- 720 tokens → 4 crops (适合 medium images)
- 1008 tokens → 6 crops (适合 large images)
- 1440 tokens → 9 crops (适合 very large images)

**优点**：
- 自动适应不同 aspect ratio
- 减少配置复杂性
- 更符合 `select_tiling` 的设计理念

## 总结

### 核心原理

**`vision_tokens_list` 的工作原理**：
1. 指定目标 vision tokens 数量
2. 计算所需的 num_crops
3. 对于每个图像，`select_tiling` 根据原始图像的 aspect ratio 自动选择最合适的 tiling
4. 自适应 resize 到最佳尺寸，最小化图像变形

**`image_size_list` 的问题**：
1. 固定了目标尺寸和 tiling
2. 所有图像都被强制 resize 到这些固定尺寸
3. 可能导致 aspect ratio 不匹配，图像变形
4. 限制了 `select_tiling` 的自动优化能力

### 推荐

**改用 `vision_tokens_list`**，因为：
- ✅ 自动适应不同 aspect ratio
- ✅ 最小化图像变形
- ✅ 充分利用 `select_tiling` 的优化机制
- ✅ 配置更简单，实验设计更灵活

