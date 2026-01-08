# MMMU Dataset Exclusion

## 原因

MMMU 数据集已从 core experiment 脚本中排除，原因如下：

### 1. 低相关性
- **Train-Validation 相关性**: 0.2558（远低于其他数据集的 0.83+）
- **统计显著性不足**: 相关性过低，结果不可靠
- **分布偏移**: 使用不同的 split (dev vs validation)，可能存在分布偏移

### 2. API 限流问题
- **HuggingFace API 429 错误**: 加载 MMMU 时经常遇到 "Too Many Requests" 错误
- **需要 HF Token**: 即使有 token，也容易触发限流
- **影响实验稳定性**: 导致实验中断和失败

### 3. 样本量小
- **样本数**: 仅 900 样本
- **统计显著性**: 样本量过小，不足以提供可靠的重要性估计

## 当前使用的数据集（8个）

1. coco_2014_vqa (VQA)
2. text_vqa (VQA)
3. okvqa (VQA)
4. science_qa_img (Multiple Choice)
5. st_qa (Scene Text QA)
6. doc_qa (Document QA)
7. tally_qa (Exact Match)
8. coco_caption (Captioning)

所有数据集的相关性 ≥ 0.83，统计显著性充足。

## 如果必须使用 MMMU

如果将来需要使用 MMMU，可以：

1. **设置 HuggingFace Token**:
   ```bash
   export HF_TOKEN=your_token_here
   # 或
   huggingface-cli login
   ```

2. **手动下载数据集**:
   ```bash
   python scripts/download_data.py mmmu --n_procs 1
   ```

3. **在脚本中取消注释**:
   ```python
   datasets = [
       # ... other datasets ...
       ("mmmu", "validation", 16),  # Uncomment if needed
   ]
   ```

## 更新

- **run_multi_datasets_h100.py**: MMMU 已注释掉
- **run_multi_datasets_a100.py**: MMMU 已注释掉

