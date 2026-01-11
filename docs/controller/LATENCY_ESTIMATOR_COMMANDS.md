# Latency Estimator 训练、评估和测试命令

> **文档目的**: 提供使用所有数据集的完整命令  
> **最后更新**: 2026-01-08  
> **版本**: 2.0 (支持Positioned Decode Latency)

## 📋 目录

1. [训练命令](#训练命令)
2. [评估命令](#评估命令)
3. [可视化命令](#可视化命令)
4. [测试命令](#测试命令)
5. [完整流程示例](#完整流程示例)

---

## 🚀 训练命令

### 使用所有数据集训练

```bash
python experiments/controller/train_latency_estimator.py \
    --results_dir results/core_exp_h100/4run_2000samples_w_importance_score_on_vqav2 \
    --use_all_datasets \
    --output_dir checkpoints/latency_estimator \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 1e-3 \
    --device cuda \
    --seed 3407
```

### 参数说明

- `--results_dir`: Core experiment结果目录
- `--use_all_datasets`: 自动检测并使用所有可用数据集
- `--output_dir`: 模型保存目录
- `--batch_size`: 批次大小（默认64）
- `--num_epochs`: 训练轮数（默认50）
- `--lr`: 学习率（默认1e-3）
- `--device`: 设备（cuda/cpu）
- `--seed`: 随机种子（默认3407）

### 训练输出

训练过程中会显示：
- 每个epoch的训练损失和验证损失
- Prefill latency的MAE（主要指标）
- Decode total latency的MAE（次要指标，sum of positioned latencies）
- 相对误差（Relative Error）

**训练策略**：
- 使用total decode latency作为训练目标（sum of positioned latencies）
- 损失函数：`loss = 2.0 * loss_prefill + 1.0 * loss_decode_total`
- 模型学习position依赖（后期tokens更慢）

最佳模型会保存到：`checkpoints/latency_estimator/best_latency_estimator.pt`

---

## 📊 评估命令

### 使用所有数据集评估

```bash
python experiments/controller/evaluate_latency_estimator.py \
    --checkpoint_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --results_dir results/core_exp_h100/4run_2000samples_w_importance_score_on_vqav2 \
    --use_all_datasets \
    --batch_size 64 \
    --device cuda \
    --output_file evaluation_results.json
```

### 参数说明

- `--checkpoint_path`: 训练好的模型路径
- `--results_dir`: Core experiment结果目录
- `--use_all_datasets`: 自动检测并使用所有可用数据集
- `--batch_size`: 批次大小（默认64）
- `--device`: 设备（cuda/cpu）
- `--output_file`: 评估结果保存路径（JSON格式）

### 评估输出

评估结果包括：

1. **整体指标**:
   - **Prefill Latency** (Primary): MAE, RMSE, MAPE, R², Relative Error
   - **Decode Total Latency** (Sum of Positioned): MAE, RMSE, MAPE, R²
   - **Decode Average Per-Token** (Reference): MAE, RMSE, MAPE, R²

2. **性能评估**:
   - Prefill Latency: ✓ Excellent (Relative error < 5%) 或 ⚠ Needs improvement
   - Decode Total Latency: ✓ Good (Relative error < 20%) 或 ⚠ Acceptable

3. **按配置的指标**:
   - 每个配置（tier_topk_blocks）的MAE和样本数

结果会保存到指定的JSON文件。

---

## 📈 可视化命令

### 使用所有数据集可视化

```bash
python experiments/controller/visualize_latency_estimator.py \
    --checkpoint_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --results_dir results/core_exp_h100/4run_2000samples_w_importance_score_on_vqav2 \
    --use_all_datasets \
    --batch_size 64 \
    --device cuda \
    --output_dir visualizations/latency_estimator \
    --max_samples 10000
```

### 参数说明

- `--checkpoint_path`: 训练好的模型路径
- `--results_dir`: Core experiment结果目录
- `--use_all_datasets`: 自动检测并使用所有可用数据集
- `--batch_size`: 批次大小（默认64）
- `--device`: 设备（cuda/cpu）
- `--output_dir`: 可视化结果保存目录
- `--max_samples`: 最大样本数（默认10000，用于加速可视化）

### 可视化输出

会生成以下可视化图表：

1. **scatter_plots.png**: 
   - Prefill Latency: Predicted vs Actual
   - Decode Average Per-Token Latency: Predicted vs Actual

2. **error_distributions.png**:
   - Prefill Latency Error Distribution
   - Decode Per-Token Latency Error Distribution

3. **errors_by_tier.png**:
   - Error by Tier (low/medium/high)

4. **errors_by_topk.png**:
   - Error by Top-K (4/6/8/10/12)

5. **errors_by_blocks.png**:
   - Error by Number of Blocks (8/10/12/14/16)

所有图表保存在指定的`output_dir`目录中。

---

## 🧪 测试命令

### 测试Latency Estimator

```bash
python experiments/controller/test_adaptive_inference.py \
    --model_path /path/to/model \
    --latency_estimator_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --device cuda \
    --test_latency_estimator
```

### 参数说明

- `--model_path`: Molmo模型路径
- `--latency_estimator_path`: 训练好的Latency Estimator路径
- `--device`: 设备（cuda/cpu）
- `--test_latency_estimator`: 测试Latency Estimator功能

### 测试输出

测试会验证：
1. Prefill latency预测
2. Positioned decode per-token latency预测
3. Total decode latency计算（sum of positioned latencies）
4. 不同配置下的预测准确性

---

## 🔄 完整流程示例

### 步骤1: 训练模型

```bash
# 训练Latency Estimator（使用所有数据集）
python experiments/controller/train_latency_estimator.py \
    --results_dir results/core_exp_h100/4run_2000samples_w_importance_score_on_vqav2 \
    --use_all_datasets \
    --output_dir checkpoints/latency_estimator \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 1e-3 \
    --device cuda \
    --seed 3407
```

**预期输出**:
```
2026-01-08 XX:XX:XX - __main__ - INFO - Auto-detecting available datasets...
2026-01-08 XX:XX:XX - __main__ - INFO - Found 9 datasets: coco_2014_vqa, coco_caption, doc_qa, mmmu, okvqa, science_qa_img, st_qa, tally_qa, text_vqa
2026-01-08 XX:XX:XX - __main__ - INFO - Loading training data...
2026-01-08 XX:XX:XX - __main__ - INFO - Filtered out X outliers (decode per-token latency > 60ms/token)
2026-01-08 XX:XX:XX - __main__ - INFO - Training samples: XXXX, Validation samples: XXXX
...
Epoch 1/50: loss=XX.XXXX, prefill_mae=XX.XXms, decode_total_mae=XX.XXms
...
2026-01-08 XX:XX:XX - __main__ - INFO - Training completed! Best model saved to checkpoints/latency_estimator/best_latency_estimator.pt
```

### 步骤2: 评估模型

```bash
# 评估训练好的模型
python experiments/controller/evaluate_latency_estimator.py \
    --checkpoint_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --results_dir results/core_exp_h100/4run_2000samples_w_importance_score_on_vqav2 \
    --use_all_datasets \
    --batch_size 64 \
    --device cuda \
    --output_file evaluation_results.json
```

**预期输出**:
```
2026-01-08 XX:XX:XX - __main__ - INFO - Loading model from checkpoints/latency_estimator/best_latency_estimator.pt
2026-01-08 XX:XX:XX - __main__ - INFO - Auto-detecting available datasets...
2026-01-08 XX:XX:XX - __main__ - INFO - Found 9 datasets: ...
2026-01-08 XX:XX:XX - __main__ - INFO - Loading evaluation data...
...
2026-01-08 XX:XX:XX - __main__ - INFO - ================================================================================
2026-01-08 XX:XX:XX - __main__ - INFO - Evaluation Results
2026-01-08 XX:XX:XX - __main__ - INFO - ================================================================================
2026-01-08 XX:XX:XX - __main__ - INFO - Prefill Latency (Primary Metric):
2026-01-08 XX:XX:XX - __main__ - INFO -   MAE: X.XXms
2026-01-08 XX:XX:XX - __main__ - INFO -   R²: 0.XXXX
...
2026-01-08 XX:XX:XX - __main__ - INFO - Decode Total Latency (Sum of Positioned Latencies):
2026-01-08 XX:XX:XX - __main__ - INFO -   MAE: XX.XXms
2026-01-08 XX:XX:XX - __main__ - INFO -   R²: 0.XXXX
...
2026-01-08 XX:XX:XX - __main__ - INFO - Results saved to evaluation_results.json
```

### 步骤3: 可视化结果

```bash
# 生成可视化图表
python experiments/controller/visualize_latency_estimator.py \
    --checkpoint_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --results_dir results/core_exp_h100/4run_2000samples_w_importance_score_on_vqav2 \
    --use_all_datasets \
    --batch_size 64 \
    --device cuda \
    --output_dir visualizations/latency_estimator \
    --max_samples 10000
```

**预期输出**:
```
2026-01-08 XX:XX:XX - __main__ - INFO - Loading model from checkpoints/latency_estimator/best_latency_estimator.pt
2026-01-08 XX:XX:XX - __main__ - INFO - Auto-detecting available datasets...
2026-01-08 XX:XX:XX - __main__ - INFO - Found 9 datasets: ...
2026-01-08 XX:XX:XX - __main__ - INFO - Loading visualization data...
...
2026-01-08 XX:XX:XX - __main__ - INFO - Generating scatter plots...
2026-01-08 XX:XX:XX - __main__ - INFO - Saved scatter_plots.png
2026-01-08 XX:XX:XX - __main__ - INFO - Generating error distribution plots...
2026-01-08 XX:XX:XX - __main__ - INFO - Saved error_distributions.png
...
2026-01-08 XX:XX:XX - __main__ - INFO - All visualizations saved to visualizations/latency_estimator
```

### 步骤4: 测试功能

```bash
# 测试Latency Estimator功能
python experiments/controller/test_adaptive_inference.py \
    --model_path /path/to/molmo/model \
    --latency_estimator_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --device cuda \
    --test_latency_estimator
```

**预期输出**:
```
2026-01-08 XX:XX:XX - __main__ - INFO - ================================================================================
2026-01-08 XX:XX:XX - __main__ - INFO - Testing Latency Estimator
2026-01-08 XX:XX:XX - __main__ - INFO - ================================================================================
Test 1:
  Config: tier=medium, top_k=8, blocks=12
  T_prefill_total: XX.XXms
  T_decode_total: XX.XXms (sum of positioned latencies)
  T_decode_per_token_avg: XX.XXms/token (average)
  T_total (estimated): XX.XXms
...
```

---

## 📝 注意事项

### 1. 数据集自动检测

使用`--use_all_datasets`时，脚本会自动检测`results_dir`下所有包含JSON文件的目录（排除`logs`目录）。

检测到的数据集会显示在日志中，例如：
```
Found 9 datasets: coco_2014_vqa, coco_caption, doc_qa, mmmu, okvqa, science_qa_img, st_qa, tally_qa, text_vqa
```

### 2. 异常值过滤

所有脚本都会自动过滤异常值：
- Decode per-token latency > 60ms/token 的样本会被过滤
- 过滤的样本数量会显示在日志中

### 3. 训练损失权重

训练时使用加权损失：
- Prefill loss权重: 2.0（主要指标）
- Decode total loss权重: 1.0（次要指标）

这反映了prefill latency是主要指标，decode latency是次要指标的设计理念。

### 4. Positioned Decode Latency

**训练策略**：
- 预测所有位置的decode latency `[1, 2, ..., output_tokens]`
- 求和得到total decode latency
- 训练目标：`MSE(sum(predicted_latencies), T_LLM_decode)`

**推理使用**：
- 可以预测任意位置的decode latency
- 根据实际output_tokens计算total decode latency
- 或者使用平均位置估算

### 5. 性能预期

基于当前数据：
- **Prefill Latency**: R² > 0.9, Relative Error < 5% ✓
- **Decode Total Latency**: R² > 0.7, Relative Error < 15% ✓
- **Decode Average Per-Token**: R² ~0.5, Relative Error ~20% ⚠️

---

## 🔗 相关文档

- **[LATENCY_ESTIMATOR_DESIGN.md](LATENCY_ESTIMATOR_DESIGN.md)**: Latency Estimator设计文档
- **[POSITIONED_DECODE_LATENCY_TRAINING.md](POSITIONED_DECODE_LATENCY_TRAINING.md)**: Positioned Decode Latency训练策略
- **[LATENCY_ESTIMATOR_IMPROVEMENT.md](LATENCY_ESTIMATOR_IMPROVEMENT.md)**: 改进方案文档
- **[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)**: 评估指南

---

**最后更新**: 2026-01-08  
**版本**: 2.0 (支持Positioned Decode Latency)  
**维护者**: Controller Team
