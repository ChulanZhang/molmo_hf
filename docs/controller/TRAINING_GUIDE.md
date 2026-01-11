# One-Stage Controller 训练指南

## 快速开始

### 基本训练命令

**注意**: 训练使用 **online training**，直接从 HuggingFace 数据集加载真实样本，不需要 `--results_dir` 参数（但脚本中仍需要提供，可以设置为任意值）。

```bash
python experiments/controller/train_joint_controller.py \
    --results_dir results/core_exp_h100 \
    --dataset_names text_vqa \
    --model_path checkpoints/molmo \
    --output_dir checkpoints/one_stage_controller \
    --batch_size 1 \
    --num_epochs 100 \
    --lr 1e-4 \
    --group_size 5 \
    --importance_scores_file results/layer_importance_scores_exp3_recommended.json
```

**重要说明**:
- `--results_dir`: 虽然需要提供，但实际上不会被使用（online training 直接从数据集加载）
- `--dataset_names`: 指定要使用的数据集（从 HuggingFace 加载）
- 当前实现默认只使用第一个数据集进行调试（代码中 `dataset_name = dataset_names[0]`）
- 训练样本数: 默认 train=200, val=50（可在代码中修改）

## 参数说明

### 必需参数

- `--results_dir`: 包含 core experiment 结果的目录
  - 示例: `results/core_exp_h100` 或 `results/core_exp_h100/5run_2000samples_w_new_importance_score`
  
- `--dataset_names`: 要加载的数据集名称（可以多个）
  - 可用选项: `text_vqa`, `coco_2014_vqa`, `okvqa`, `coco_caption` 等
  - 示例: `--dataset_names text_vqa coco_2014_vqa`
  
- `--model_path`: 模型 checkpoint 路径
  - 示例: `checkpoints/molmo`

### 训练参数

- `--output_dir`: 保存 checkpoints 和日志的目录（默认: `./checkpoints/joint_controller`）
- `--batch_size`: Batch size（默认: 64，但会自动调整为 1 for single GPU）
- `--num_epochs`: 训练轮数（默认: 100）
- `--lr`: 学习率（默认: 1e-4）
- `--group_size`: GRPO 组大小（默认: 5，每个样本采样 5 个配置）
- `--seed`: 随机种子（默认: 42）

### 可选参数

- `--importance_scores_file`: Importance scores JSON 文件路径
  - 默认: `results/layer_importance_scores_exp3_recommended.json`
  - 用于 block 选择（fallback，如果未提供则使用 prefix blocks）

- `--load_stage1_checkpoint`: 预训练 checkpoint 路径（用于 fine-tuning）
  - 示例: `checkpoints/one_stage_controller/best_model.pt`

- `--use_multi_gpu`: 启用多 GPU 训练（使用 DataParallel）
  - 如果启用，batch_size 会自动调整为 GPU 数量

- `--use_wandb`: 启用 Weights & Biases 日志记录
  - 需要安装: `pip install wandb`
  - 相关参数:
    - `--wandb_project`: W&B 项目名称（默认: `molmo-controller`）
    - `--wandb_entity`: W&B 实体/团队名称（可选）
    - `--wandb_name`: W&B run 名称（默认: 自动生成）

## 训练流程

### 1. 数据准备

训练使用 **online training**，直接从数据集加载真实样本（图像 + 提示），而不是使用 profiling 结果。

数据集会自动从 HuggingFace 加载：
- `text_vqa`: TextVQA 数据集
- `coco_2014_vqa`: COCO VQA 数据集
- `okvqa`: OK-VQA 数据集

### 2. 训练过程

1. **特征提取**:
   - Vision features: 从 vision_backbone 提取（全局 crop）
   - Language features: 从 tokenizer + WTE layer 提取
   - Budget features: 从 LatencyBudgetEncoder 编码

2. **控制器预测**:
   - 输入: `(B * group_size, vision_dim)`, `(B * group_size, lang_dim)`, `(B * group_size, budget_dim)`
   - 输出: tier_logits, block_logits, block_topk_logits

3. **配置采样**:
   - 每个原始样本采样 `group_size` 个不同配置
   - 总共 `B * group_size` 个配置

4. **模型执行**:
   - 每个配置执行一次完整的模型 forward pass
   - 使用 hooks 测量实际 latency
   - 计算 accuracy（使用与 core_exp 相同的评估方法）

5. **GRPO Loss**:
   - 按 `(sample_id, latency_budget)` 分组
   - 计算 group-normalized advantages
   - Policy gradient: `loss = -log_prob * advantage`

### 3. 输出文件

训练过程中会生成：

- **Checkpoints**:
  - `checkpoints/one_stage_controller/best_model.pt`: 最佳模型
  - `checkpoints/one_stage_controller/checkpoint_epoch_{N}.pt`: 每个 epoch 的 checkpoint

- **日志文件**:
  - `results/logs/training/joint_controller_training.log`: 训练日志

- **CSV 文件**:
  - `checkpoints/one_stage_controller/training_history.csv`: 训练历史
  - `checkpoints/one_stage_controller/validation_history.csv`: 验证历史

- **TensorBoard**:
  - `checkpoints/one_stage_controller/tensorboard/`: TensorBoard 日志
  - 查看: `tensorboard --logdir=checkpoints/one_stage_controller/tensorboard`

- **Weights & Biases** (如果启用):
  - 在线查看: https://wandb.ai

## 训练示例

### 示例 1: 单 GPU 训练（调试）

```bash
python experiments/controller/train_joint_controller.py \
    --results_dir results/core_exp_h100 \
    --dataset_names text_vqa \
    --model_path checkpoints/molmo \
    --output_dir checkpoints/one_stage_controller \
    --batch_size 1 \
    --num_epochs 10 \
    --lr 1e-4 \
    --group_size 5 \
    --seed 42
```

### 示例 2: 多 GPU 训练（生产）

```bash
python experiments/controller/train_joint_controller.py \
    --results_dir results/core_exp_h100 \
    --dataset_names text_vqa coco_2014_vqa okvqa \
    --model_path checkpoints/molmo \
    --output_dir checkpoints/one_stage_controller \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 1e-4 \
    --group_size 5 \
    --use_multi_gpu \
    --use_wandb \
    --wandb_project molmo-one-stage-controller \
    --seed 42
```

### 示例 3: 从预训练 checkpoint 继续训练

```bash
python experiments/controller/train_joint_controller.py \
    --results_dir results/core_exp_h100 \
    --dataset_names text_vqa coco_2014_vqa \
    --model_path checkpoints/molmo \
    --output_dir checkpoints/one_stage_controller \
    --load_stage1_checkpoint checkpoints/one_stage_controller/best_model.pt \
    --batch_size 1 \
    --num_epochs 50 \
    --lr 5e-5 \
    --group_size 5
```

## 训练监控

### 实时监控

训练过程中会显示：
- Progress bar: 显示当前 epoch 的进度
- Metrics: loss, reward, accuracy, latency
- 日志: 保存到 `results/logs/training/joint_controller_training.log`

### TensorBoard

启动 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir=checkpoints/one_stage_controller/tensorboard
```

然后在浏览器中打开: http://localhost:6006

### Weights & Biases

如果启用了 wandb，可以在线查看：
- 项目页面: https://wandb.ai/{entity}/molmo-controller
- 实时指标更新
- 系统资源监控

## 训练技巧

### 1. Batch Size 设置

- **单 GPU**: `batch_size=1`（自动设置）
- **多 GPU**: `batch_size=num_gpus`（每个 GPU 处理 1 个样本）
- **注意**: 由于每个样本会采样 `group_size` 个配置，实际 forward 次数 = `batch_size * group_size`

### 2. 学习率调整

- **初始训练**: `lr=1e-4`
- **Fine-tuning**: `lr=5e-5` 或 `1e-5`
- **学习率调度**: 使用 ReduceLROnPlateau（基于验证集 reward）

### 3. Group Size

- **默认**: `group_size=5`（每个样本采样 5 个配置）
- **更大值**: 可能提高 GRPO 效果，但会增加训练时间
- **更小值**: 训练更快，但可能影响 GRPO 效果

### 4. 数据集选择

- **单个数据集**: 用于调试和快速迭代
- **多个数据集**: 用于生产训练，提高泛化能力
- **建议**: 从单个数据集开始，验证流程后再扩展到多个数据集

### 5. 内存优化

如果遇到 OOM（Out of Memory）错误：

- 减少 `batch_size`（虽然默认是 1）
- 减少 `group_size`（从 5 降到 3）
- 使用梯度累积（需要修改代码）
- 使用更少的 datasets

## 常见问题

### Q: 训练很慢怎么办？

A: One-stage controller 训练需要执行完整的模型 forward pass，每个配置一次。训练时间主要取决于：
- 数据集大小
- `group_size`（每个样本的配置数）
- 模型大小
- GPU 数量

**优化建议**:
- 使用多 GPU（`--use_multi_gpu`）
- 减少数据集大小（用于调试）
- 减少 `group_size`（从 5 降到 3）

### Q: Loss 一直是 0 或很小？

A: 检查：
1. GRPO grouping 是否正常工作（查看日志中的 `num_groups`）
2. Rewards 是否有变化（查看 `reward_mean`, `reward_std`）
3. 是否有足够的配置差异（检查 `num_configs`）

### Q: Accuracy 一直是 0？

A: 检查：
1. 数据集是否正确加载（查看日志中的 "Loaded X samples"）
2. Metadata 中是否包含 `answers`（查看 debug 日志）
3. Tokenizer 是否正确（查看 `pred_text` 和 `answers` 的格式）

### Q: 如何恢复训练？

A: 使用 `--load_stage1_checkpoint` 参数加载 checkpoint：

```bash
--load_stage1_checkpoint checkpoints/one_stage_controller/checkpoint_epoch_50.pt
```

注意：当前实现会加载 controller 权重，但不会恢复 optimizer 和 scheduler 状态（需要修改代码支持完整恢复）。

## 训练输出示例

```
2026-01-11 17:00:03 - __main__ - INFO - Training One-Stage Controller with GRPO
2026-01-11 17:00:03 - __main__ - INFO - Loaded 200 training samples
2026-01-11 17:00:03 - __main__ - INFO - ================================================================================
2026-01-11 17:00:03 - __main__ - INFO - Epoch 1/100
2026-01-11 17:00:03 - __main__ - INFO - ================================================================================
Training Epoch 1: 100%|████████████| 200/200 [45:30<00:00, 13.65s/it, loss=-0.1234, reward=-0.5678, acc=0.2000]
2026-01-11 17:45:33 - __main__ - INFO - Train Epoch 1 Metrics:
2026-01-11 17:45:33 - __main__ - INFO -   Loss: -0.1234
2026-01-11 17:45:33 - __main__ - INFO -   Reward: -0.5678 (std: 0.1234)
2026-01-11 17:45:33 - __main__ - INFO -   Accuracy: 0.2000 (std: 0.1234)
2026-01-11 17:45:33 - __main__ - INFO -   Latency: 234.56ms (std: 45.67ms)
2026-01-11 17:45:33 - __main__ - INFO -   Budget Violation Rate: 0.1500
```

## 下一步

训练完成后，可以：
1. 评估训练好的 controller（使用 `evaluate_adaptive_inference.py`）
2. 在真实任务上测试（使用 `run_lmms_eval.py`）
3. 分析训练曲线（使用 TensorBoard 或 wandb）
4. 调整超参数并重新训练

