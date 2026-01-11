# Joint Controller训练指南

## 快速开始

### 1. 准备数据

确保以下路径存在：
- **Model checkpoint**: `checkpoints/molmo/`
- **Latency estimator**: `checkpoints/latency_estimator/best_latency_estimator.pt`
- **Profiling results** (可选，用于数据加载): `results/core_exp_h100/5run_2000samples_w_new_importance_score/`

### 2. 运行训练

#### 方法1: 使用Shell脚本（推荐）

```bash
chmod +x experiments/controller/run_training.sh
./experiments/controller/run_training.sh
```

#### 方法2: 直接运行Python脚本

```bash
python experiments/controller/train_joint_controller.py \
    --results_dir results/core_exp_h100/5run_2000samples_w_new_importance_score \
    --dataset_names text_vqa coco_2014_vqa okvqa \
    --model_path checkpoints/molmo \
    --latency_estimator_path checkpoints/latency_estimator/best_latency_estimator.pt \
    --output_dir checkpoints/joint_controller \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 1e-4 \
    --stage1_lr_ratio 1.0 \
    --group_size 5 \
    --device cuda \
    --seed 42
```

### 3. 参数说明

#### 必需参数
- `--results_dir`: Profiling结果目录（用于数据加载，但实际训练使用online dataset）
- `--dataset_names`: 数据集名称列表（例如：`text_vqa coco_2014_vqa okvqa`）
- `--model_path`: 模型checkpoint路径
- `--latency_estimator_path`: Latency estimator路径

#### 训练参数
- `--output_dir`: 输出目录（默认：`./checkpoints/joint_controller`）
- `--batch_size`: Batch size（默认：32，建议从8开始测试）
- `--num_epochs`: 训练轮数（默认：100）
- `--lr`: 学习率（默认：1e-4）
- `--stage1_lr_ratio`: Stage1学习率比例（默认：1.0，如果Stage1已预训练可设为0.1）
- `--group_size`: GRPO组大小（默认：5）
- `--device`: 设备（默认：`cuda`）
- `--seed`: 随机种子（默认：42）

#### 可选参数
- `--load_stage1_checkpoint`: 预训练的Stage1 checkpoint路径（可选）

### 4. 训练输出

训练过程中会输出：
- **训练指标**：loss, reward, accuracy, latency, budget violation rate
- **验证指标**：reward, accuracy, latency, budget violation rate
- **Checkpoints**：
  - `best_joint_checkpoint.pt`: 最佳模型（基于验证reward）
  - `joint_checkpoint_epoch_{N}.pt`: 每10个epoch的checkpoint

### 5. 监控训练

训练日志会保存到：
- **控制台输出**：实时显示训练进度
- **日志文件**：`joint_controller_training.log`

### 6. 训练流程

1. **加载数据**：从指定数据集加载images和prompts
2. **特征提取**：提取language features和budget features
3. **Stage1预测**：预测Knob1 (tier: low/medium/high)
4. **Vision处理**：根据tier处理images
5. **Stage2预测**：预测Knob2 (top_k) 和 Knob3 (num_blocks)
6. **模型执行**：使用预测的配置执行模型生成文本
7. **计算Reward**：基于accuracy和latency计算reward
8. **GRPO更新**：使用GRPO算法更新controller参数

### 7. 常见问题

#### Q: 训练很慢怎么办？
A: 
- 减小batch_size（例如：8或4）
- 限制validation samples数量
- 使用更少的datasets

#### Q: OOM错误？
A:
- 减小batch_size
- 使用gradient accumulation
- 限制max_new_tokens

#### Q: Accuracy一直很低？
A:
- 检查metadata是否正确加载
- 检查ground truth answers格式
- 增加训练轮数

#### Q: Budget violation rate很高？
A:
- 检查latency estimator是否准确
- 调整reward function中的gamma参数
- 检查latency budget范围是否合理

### 8. 训练技巧

1. **从小batch开始**：先用batch_size=4或8测试，确保代码运行正常
2. **监控关键指标**：
   - Reward应该逐渐增加
   - Budget violation rate应该逐渐降低
   - Accuracy应该逐渐提高
3. **调整学习率**：如果训练不稳定，可以降低学习率（例如：5e-5）
4. **使用预训练Stage1**：如果Stage1已预训练，设置`--stage1_lr_ratio 0.1`进行fine-tuning

### 9. 下一步

训练完成后：
1. 使用`best_joint_checkpoint.pt`进行推理
2. 评估在不同latency budget下的性能
3. 分析controller的决策模式

