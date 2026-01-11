# Latency Estimator 训练过程和效果查看指南

> **文档目的**: 详细说明如何查看Latency Estimator的训练过程和评估模型效果

## 📋 目录

1. [查看训练过程](#查看训练过程)
2. [查看训练日志](#查看训练日志)
3. [查看Checkpoint信息](#查看checkpoint信息)
4. [评估模型效果](#评估模型效果)
5. [可视化训练曲线](#可视化训练曲线)

---

## 🔍 查看训练过程

### 1. 实时查看训练日志

训练过程中，日志会实时输出到控制台，包括：

- **每个batch的进度条**：显示当前loss、MAE等指标
- **每个epoch的训练指标**：平均loss、MAE等
- **每个epoch的验证指标**：验证集上的loss、MAE等
- **最佳模型保存提示**：当验证loss降低时保存模型

**示例输出**:
```
Epoch 1/50: 100%|████████| 1350/1350 [00:03<00:00, 435.70it/s, loss=376.72, mae_prefill=14.34ms, mae_decode=4.928ms]
2026-01-08 02:05:03,503 - __main__ - INFO - Train Epoch 1: {'loss': 5536.66, 'mae_prefill': 53.86, 'mae_decode': 6.62, ...}
2026-01-08 02:05:03,503 - __main__ - INFO - Val Epoch 1: {'loss': 1234.56, 'mae_prefill': 12.34, 'mae_decode': 2.34, ...}
2026-01-08 02:05:03,503 - __main__ - INFO - Saved best model (val_loss=1234.56)
```

### 2. 保存训练日志到文件

如果训练时没有重定向日志，可以重新运行并保存：

```bash
python experiments/controller/train_latency_estimator.py \
    --results_dir results/core_exp_h100/4run_2000samples \
    --dataset_names text_vqa coco_2014_vqa \
    --output_dir checkpoints/latency_estimator \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 1e-3 \
    --device cuda \
    --seed 3407 \
    2>&1 | tee training.log
```

这样日志会同时显示在控制台和保存到`training.log`文件中。

---

## 📊 查看训练日志

### 从日志文件中提取关键信息

```bash
# 查看所有epoch的训练指标
grep "Train Epoch" training.log

# 查看所有epoch的验证指标
grep "Val Epoch" training.log

# 查看最佳模型保存记录
grep "Saved best model" training.log

# 查看训练完成信息
grep "Training completed" training.log
```

### 关键指标说明

- **loss**: 总损失（prefill loss + decode loss）
- **mae_prefill**: Prefill latency的平均绝对误差（ms）
- **mae_decode**: Decode per-token latency的平均绝对误差（ms/token）
- **rel_error_prefill**: Prefill latency的相对误差（%）
- **rel_error_decode**: Decode per-token latency的相对误差（%）

---

## 💾 查看Checkpoint信息

### 1. 查看保存的Checkpoint文件

训练完成后，checkpoint目录下会有：

```
checkpoints/latency_estimator/
├── best_latency_estimator.pt          # 最佳模型（验证loss最低）
├── latency_estimator_epoch_10.pt      # 每10个epoch的checkpoint
├── latency_estimator_epoch_20.pt
├── latency_estimator_epoch_30.pt
├── latency_estimator_epoch_40.pt
└── latency_estimator_epoch_50.pt
```

### 2. 查看Checkpoint中的验证指标

```python
import torch

checkpoint_path = "checkpoints/latency_estimator/best_latency_estimator.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Epoch:", checkpoint['epoch'])
print("Validation Metrics:")
for key, value in checkpoint['val_metrics'].items():
    print(f"  {key}: {value:.4f}")
```

**输出示例**:
```
Epoch: 45
Validation Metrics:
  loss: 1234.5678
  mae_prefill: 12.3456
  mae_decode: 2.3456
  rel_error_prefill: 0.1234
  rel_error_decode: 0.2345
```

---

## 🧪 评估模型效果

### 使用评估脚本

我们提供了专门的评估脚本来详细评估模型效果：

```bash
# 使用所有可用数据集（推荐）
python experiments/controller/evaluate_latency_estimator.py \
    --checkpoint_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --results_dir results/core_exp_h100/4run_2000samples \
    --use_all_datasets \
    --batch_size 64 \
    --device cuda \
    --output_file evaluation_results.json

# 或者不指定dataset_names，会自动检测所有数据集
python experiments/controller/evaluate_latency_estimator.py \
    --checkpoint_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --results_dir results/core_exp_h100/4run_2000samples \
    --batch_size 64 \
    --device cuda \
    --output_file evaluation_results.json

# 指定特定数据集
python experiments/controller/evaluate_latency_estimator.py \
    --checkpoint_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --results_dir results/core_exp_h100/4run_2000samples \
    --dataset_names text_vqa coco_2014_vqa coco_caption \
    --batch_size 64 \
    --device cuda \
    --output_file evaluation_results.json
```

### 评估输出说明

评估脚本会输出：

1. **Checkpoint验证指标**：训练时保存的验证集指标
2. **整体评估指标**：
   - **Prefill Latency**:
     - MAE (Mean Absolute Error): 平均绝对误差
     - RMSE (Root Mean Square Error): 均方根误差
     - MAPE (Mean Absolute Percentage Error): 平均绝对百分比误差
     - R²: 决定系数（越接近1越好）
     - Mean/Std Target: 目标值的均值和标准差
   - **Decode Per-Token Latency**: 同样的指标
3. **按配置分组的指标**：不同configuration下的误差分析

**输出示例**:
```
================================================================================
Evaluation Results
================================================================================

Overall Metrics:
Prefill Latency:
  MAE: 12.34ms
  RMSE: 15.67ms
  MAPE: 8.90%
  R²: 0.9234
  Mean Target: 138.56ms
  Std Target: 25.34ms

Decode Per-Token Latency:
  MAE: 2.345ms/token
  RMSE: 3.456ms/token
  MAPE: 12.34%
  R²: 0.8765
  Mean Target: 19.05ms/token
  Std Target: 2.67ms/token

Per-Configuration Metrics (Top 10 by sample count):
  low_topk4_blocks12:
    Count: 5000
    Prefill MAE: 10.23ms
    Decode MAE: 2.12ms/token
  ...
```

### 评估结果JSON文件

如果指定了`--output_file`，评估结果会保存为JSON格式，包含：
- Checkpoint路径和指标
- 整体评估指标
- 按配置分组的详细指标

可以用于后续分析和可视化。

---

## 📊 直观理解模型表现

### 快速查看方法

1. **查看散点图**: 预测值 vs 真实值，点越接近对角线越好
2. **查看误差分布**: 误差集中在0附近，分布越窄越好
3. **查看配置分组**: 不同配置下的误差是否均匀

### 使用可视化脚本（推荐）

我们提供了专门的可视化脚本来生成各种图表，直观展示模型性能：

```bash
# 使用所有可用数据集（推荐）
python experiments/controller/visualize_latency_estimator.py \
    --checkpoint_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --results_dir results/core_exp_h100/4run_2000samples \
    --use_all_datasets \
    --device cuda \
    --output_dir visualizations/latency_estimator \
    --max_samples 10000

# 或者不指定dataset_names，会自动检测所有数据集
python experiments/controller/visualize_latency_estimator.py \
    --checkpoint_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --results_dir results/core_exp_h100/4run_2000samples \
    --device cuda \
    --output_dir visualizations/latency_estimator \
    --max_samples 10000

# 指定特定数据集
python experiments/controller/visualize_latency_estimator.py \
    --checkpoint_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --results_dir results/core_exp_h100/4run_2000samples \
    --dataset_names text_vqa coco_2014_vqa coco_caption \
    --device cuda \
    --output_dir visualizations/latency_estimator \
    --max_samples 10000
```

### 生成的可视化图表

脚本会生成以下图表，保存在`visualizations/latency_estimator/`目录：

#### 1. **散点图 (scatter_plots.png)**
   - **Prefill Latency散点图**: 预测值 vs 真实值
   - **Decode Per-Token Latency散点图**: 预测值 vs 真实值
   - **解读**:
     - 点越接近红色对角线（y=x），预测越准确
     - 显示MAE、RMSE、R²等指标
     - 如果点分布在对角线附近，说明模型预测准确

#### 2. **误差分布图 (error_distributions.png)**
   - **Prefill Latency误差分布**: 绝对误差的直方图
   - **Decode Per-Token Latency误差分布**: 绝对误差的直方图
   - **解读**:
     - 误差分布越集中在0附近越好
     - 红色虚线表示平均误差，绿色虚线表示中位数误差
     - 如果大部分误差都很小，说明模型性能好

#### 3. **按配置分组的误差分析**
   - **errors_by_tier.png**: 按tier（low/medium/high）分组的误差
   - **errors_by_topk.png**: 按top-K值分组的误差
   - **errors_by_blocks.png**: 按block数量分组的误差
   - **解读**:
     - 可以看出模型在不同配置下的表现
     - 如果某些配置误差较大，可能需要更多训练数据
     - 误差柱状图显示均值和标准差

#### 4. **摘要统计 (summary.json)**
   - 保存整体MAE和RMSE指标到JSON文件

### 图表解读指南

#### 散点图解读
- **理想情况**: 所有点都在红色对角线上
- **良好情况**: 点大致分布在对角线附近，R² > 0.9
- **需要改进**: 点分散，远离对角线，R² < 0.8

#### 误差分布解读
- **理想情况**: 误差集中在0附近，大部分误差 < 5ms（prefill）或 < 1ms（decode）
- **良好情况**: 平均误差 < 10ms（prefill）或 < 2ms（decode）
- **需要改进**: 误差分布很宽，平均误差很大

#### 配置分组解读
- **均匀表现**: 所有配置的误差都接近，说明模型泛化好
- **差异较大**: 某些配置误差明显更大，可能需要针对性改进

### 可视化训练曲线（从日志）

如果想可视化训练过程，可以从日志文件中提取数据：

```python
import re
import matplotlib.pyplot as plt

# 从日志文件中提取训练指标
train_losses = []
val_losses = []
val_mae_prefill = []
val_mae_decode = []

with open('training.log', 'r') as f:
    for line in f:
        if 'Train Epoch' in line:
            match = re.search(r"'loss': ([\d.]+)", line)
            if match:
                train_losses.append(float(match.group(1)))
        elif 'Val Epoch' in line:
            match_loss = re.search(r"'loss': ([\d.]+)", line)
            match_prefill = re.search(r"'mae_prefill': ([\d.]+)", line)
            match_decode = re.search(r"'mae_decode': ([\d.]+)", line)
            if match_loss:
                val_losses.append(float(match_loss.group(1)))
            if match_prefill:
                val_mae_prefill.append(float(match_prefill.group(1)))
            if match_decode:
                val_mae_decode.append(float(match_decode.group(1)))

# 绘制训练曲线
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 3, 2)
plt.plot(val_mae_prefill, label='Prefill MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE (ms)')
plt.legend()
plt.title('Prefill Latency MAE')

plt.subplot(1, 3, 3)
plt.plot(val_mae_decode, label='Decode MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE (ms/token)')
plt.legend()
plt.title('Decode Per-Token Latency MAE')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
plt.show()
```

---

## ✅ 预期性能指标

根据设计文档，预期性能指标为：

### Prefill Latency
- **MAE < 5ms**: 平均绝对误差
- **RMSE < 10ms**: 均方根误差
- **MAPE < 5%**: 平均绝对百分比误差
- **R² > 0.9**: 决定系数

### Decode Per-Token Latency
- **MAE < 1ms**: 平均绝对误差
- **RMSE < 2ms**: 均方根误差
- **MAPE < 10%**: 平均绝对百分比误差
- **R² > 0.85**: 决定系数

如果实际指标接近或优于这些值，说明模型训练良好。

---

## 🔧 快速测试模型

### 使用test_adaptive_inference.py

```bash
python experiments/controller/test_adaptive_inference.py \
    --latency_estimator_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --test_latency_estimator \
    --device cuda
```

这会运行一些测试用例，展示模型在不同配置下的预测结果。

---

## 📚 相关文档

- **[LATENCY_ESTIMATOR_DESIGN.md](LATENCY_ESTIMATOR_DESIGN.md)**: Latency Estimator设计文档
- **[EXPERIMENTS.md](EXPERIMENTS.md)**: 实验说明（Exp 1）

---

**最后更新**: 2026-01-08  
**维护者**: Controller Team

