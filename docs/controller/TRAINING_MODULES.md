# Training Modules Status

## 当前训练状态

### ✅ 会被训练的模块

1. **Stage1 Controller (`knob1_predictor`)**
   - 预测：Knob1 (tier: low/medium/high) + Stage2插入位置 (1-5)
   - 学习率：`lr * stage1_lr_ratio` (默认 `1e-4 * 1.0 = 1e-4`)
   - 参数数量：~50K-100K

2. **Stage2 Controller (`knob2_knob3_predictor`)**
   - 预测：Knob2 (top_k: 4,5,6,7,8) + Knob3 (num_blocks: 12-16)
   - 学习率：`lr` (默认 `1e-4`)
   - 参数数量：~10K-30K

### ❌ 不会被训练的模块（Frozen）

1. **Latency Budget Encoder (`budget_encoder`)**
   - 状态：**Frozen**（未加入optimizer）
   - 原因：目前只用于编码budget到token，不参与梯度更新
   - 参数数量：~4M-8M（取决于d_model）

2. **LLM Model (`model`)**
   - 状态：**Frozen**（未加入optimizer）
   - 原因：只训练controller，不训练LLM本身
   - 参数数量：~7B

3. **Language Feature Extractor (`lang_extractor`)**
   - 状态：**Frozen**（wte_layer被freeze）
   - 原因：只使用预训练的word embedding
   - 参数数量：0（只是wrapper）

## Optimizer配置

```python
self.optimizer = optim.Adam(
    [
        {'params': self.knob1_predictor.parameters(), 'lr': stage1_lr},
        {'params': self.knob2_knob3_predictor.parameters(), 'lr': lr},
    ],
    weight_decay=weight_decay,
)
```

**注意**：`budget_encoder` 没有被加入到optimizer中，所以它不会被训练。

## 是否需要训练 Budget Encoder？

### 当前设计（不训练）

**优点**：
- Budget encoder的参数是固定的，确保budget编码的一致性
- 减少训练参数，加快训练速度
- Budget encoder的sinusoidal encoding是确定性的，不需要学习

**缺点**：
- 无法根据实际训练数据优化budget编码方式
- 可能无法充分利用budget信息

### 如果要训练 Budget Encoder

需要修改 `JointGRPOTrainer.__init__`：

```python
self.optimizer = optim.Adam(
    [
        {'params': self.knob1_predictor.parameters(), 'lr': stage1_lr},
        {'params': self.knob2_knob3_predictor.parameters(), 'lr': lr},
        {'params': self.budget_encoder.parameters(), 'lr': lr},  # 添加这一行
    ],
    weight_decay=weight_decay,
)
```

## 训练流程

1. **初始化**：
   - 加载LLM模型（frozen）
   - 初始化budget_encoder（frozen）
   - 初始化knob1_predictor和knob2_knob3_predictor（trainable）

2. **训练循环**：
   - 对每个batch：
     - Stage1预测tier和insertion_position
     - 运行模型到insertion_position，提取latency_token
     - Stage2预测top_k和num_blocks
     - 执行模型，计算accuracy和latency
     - 计算reward
     - GRPO loss反向传播（只更新controller参数）

3. **验证**：
   - 使用deterministic策略（argmax）进行验证
   - 计算validation metrics

## 运行训练

```bash
./experiments/controller/run_training.sh
```

或者：

```bash
python experiments/controller/train_joint_controller.py \
    --results_dir results/core_exp_h100/5run_2000samples_w_new_importance_score \
    --dataset_names text_vqa coco_2014_vqa okvqa \
    --model_path checkpoints \
    --output_dir checkpoints/joint_controller \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 1e-4 \
    --stage1_lr_ratio 1.0 \
    --group_size 5 \
    --device cuda \
    --seed 42 \
    --use_multi_gpu
```

## 总结

- ✅ **可以运行**：`./experiments/controller/run_training.sh` 可以直接运行
- ✅ **Joint Training**：Stage1和Stage2会被jointly训练
- ❌ **Budget Encoder**：目前**不会**被训练（frozen）
- ❌ **LLM Model**：不会被训练（frozen）

如果需要训练budget_encoder，需要将其参数加入到optimizer中。

