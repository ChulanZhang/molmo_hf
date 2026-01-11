# Joint Training for Stage1 and Stage2 Controller

> **注意**: 这是Controller的唯一训练方式。分阶段训练已被移除。

## 为什么使用Joint Training？

**核心问题**：Stage1和Stage2都会影响latency
- **Stage1 (Knob1)**: 决定vision tokens数量（tier）和Stage2插入位置，影响vision encoding latency和LLM执行
- **Stage2 (Knob2 & Knob3)**: 决定MoE top-K和transformer blocks，影响LLM latency

**Joint Training的优势**：
- 两个阶段共享同一个reward信号（accuracy + latency constraint）
- 可以全局优化end-to-end性能
- Stage1可以学习到"为了满足budget，应该选择哪个tier和插入位置"
- Stage2可以学习到"基于latency token，应该选择哪个top_k和num_blocks"

## 架构设计

### 执行流程

```
Input: Image + Prompt + Latency Budget
    ↓
Stage 1: Predict Knob1 (tier + insertion position)
    - Input: Language Feature + Budget Token (encoded, concatenated)
    - Output: Tier (low/medium/high) + Insertion Position (1-5)
    ↓
Image Preprocessing (based on Knob1 tier)
    ↓
Vision Encoding
    ↓
LLM Forward to Insertion Position
    - Run blocks up to insertion position
    - Extract: Latency Token (last token)
    ↓
Stage 2: Predict Knob2 & Knob3
    - Input: Latency Token (contains budget + vision + language interaction)
    - Output: Top-K (4/5/6/7/8) + Total Blocks (12/13/14/15/16)
    ↓
Apply Knobs to Remaining LLM Blocks
    - Set top_k for blocks after insertion position
    - Select blocks by importance
    - First block fixed: top_k=8, always included
    ↓
Execute Model with all knobs
    - Prefill: Generate with all knobs applied
    - Decode: Use prefill configuration (no controller re-run)
    ↓
Compute Reward:
    - Accuracy (model output quality)
    - Latency constraint (budget violation penalty)
    - Direct latency measurement (hooks)
    ↓
Joint GRPO Loss (both stages contribute)
```

### 关键设计

1. **Budget Token**: 编码为d_model维token，在prefill阶段拼接到输入序列
2. **Latency Token**: 从LLM插入位置之后提取，包含所有必要信息
3. **Dynamic Insertion**: Stage1决定Stage2的插入位置（1-5）
4. **Decode Phase**: 使用prefill配置，不重新运行controller

## Reward Function

```python
reward = accuracy_reward 
       - latency_penalty 
       - budget_violation_penalty  # Hard constraint
       - complexity_penalty 
       + efficiency_bonus
```

**关键点**：
- **accuracy_reward**: 模型输出质量（不是tier预测准确率）
- **budget_violation_penalty**: 硬约束，latency超过budget时大幅惩罚
- 两个阶段共享同一个reward信号

## 训练流程

### 1. 初始化

```python
# 创建两个predictor
knob1_predictor = Knob1PredictorBudgetLanguage(...)
knob2_knob3_predictor = Knob2Knob3Predictor(...)

# 创建budget encoder
budget_encoder = LatencyBudgetEncoder(d_model=2048, use_sinusoidal=True)

# 创建joint trainer
trainer = JointGRPOTrainer(
    knob1_predictor=knob1_predictor,
    knob2_knob3_predictor=knob2_knob3_predictor,
    model=model,
    reward_fn=reward_fn,
    budget_encoder=budget_encoder,
    ...
)
```

### 2. 训练步骤

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # Stage 1: Predict Knob1
        knob1_output = knob1_predictor(lang_feat, budget_feat)
        tier_logits = knob1_output['tier_logits']
        insertion_logits = knob1_output['insertion_logits']
        tier_action = sample(tier_logits)  # tier
        insertion_action = sample(insertion_logits)  # insertion position
        
        # Process images with tier
        # ... vision encoding ...
        
        # Run LLM to insertion position, extract latency token
        latency_token = run_llm_to_position(model, insertion_action)
        
        # Stage 2: Predict Knob2 & Knob3
        knob2_knob3_output = knob2_knob3_predictor(
            latency_token, insertion_action
        )
        knob2_logits = knob2_knob3_output['knob2_logits']
        knob3_logits = knob2_knob3_output['knob3_logits']
        knob2_action = sample(knob2_logits)  # top_k
        knob3_action = sample(knob3_logits)  # num_blocks
        
        # Execute model with all knobs (direct latency measurement)
        result = execute_model(
            tier=tier_action,
            insertion_position=insertion_action,
            top_k=knob2_action,
            num_blocks=knob3_action,
            latency_budget=batch['latency_budget'],
            budget_encoder=budget_encoder,
        )
        
        # Compute reward (direct latency measurement)
        reward = reward_fn(
            accuracy=result['accuracy'],
            latency=result['total_latency'],  # Measured using hooks
            latency_budget=batch['latency_budget'],
            config={
                'tier': tier_action,
                'insertion_position': insertion_action,
                'top_k': knob2_action,
                'num_blocks': knob3_action,
            },
        )
        
        # Joint GRPO loss
        loss = compute_joint_grpo_loss(
            tier_logits, insertion_logits, knob2_logits, knob3_logits,
            actions, rewards, groups
        )
        
        # Backward (both stages + budget encoder MLP)
        loss.backward()
        optimizer.step()
```

### 3. 验证指标

**关键指标**：
1. **Reward mean**: 平均reward（越高越好）
2. **Accuracy mean**: 平均模型输出accuracy（越高越好）
3. **Latency mean**: 平均latency（越低越好，但要满足budget）
4. **Budget violation rate**: Budget违反率（越低越好，目标<5%）

**不是**：
- ❌ Tier预测准确率（这是中间指标，不是最终目标）
- ❌ 单独的Stage1或Stage2准确率

## 使用方式

### 训练命令

```bash
./experiments/controller/run_training.sh
```

或者手动运行：

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

**关键参数**：
- `--batch_size 8`: 每个样本单独处理（batch_size=1 per sample）以确保准确测量
- `--lr 1e-4`: 学习率
- `--stage1_lr_ratio 1.0`: Stage1学习率比例（相对于Stage2）
- `--group_size 5`: GRPO group size
- `--use_multi_gpu`: 可选，使用多GPU训练

**注意**：
- **不使用latency estimator**: 使用direct latency measurement（hooks）
- `--stage1_lr_ratio 1.0` 表示两个阶段使用相同学习率（从头训练）
- 如果要从预训练的Stage1开始，可以使用 `--load_stage1_checkpoint` 和 `--stage1_lr_ratio 0.1`

### 参数说明

- `--stage1_lr_ratio`: Stage1学习率相对于Stage2的比例
  - `1.0`: 两个阶段使用相同学习率（从头训练）
  - `0.1`: Stage1使用较小学习率（fine-tuning预训练的Stage1）
  - `0.0`: 固定Stage1，只训练Stage2（不推荐，失去joint training优势）

## Direct Latency Measurement

**当前实现**: 使用PyTorch hooks直接测量latency

**方法**:
- 在prefill和decode阶段注册hooks
- 使用`time.perf_counter()`测量实际执行时间
- Batch size = 1 per sample（确保准确测量）

**优势**:
- 更准确（实际测量而非估计）
- 不需要额外的estimator模型
- 可以捕获硬件特定的latency特性

**劣势**:
- 训练速度较慢（需要实际运行模型）
- 不能使用大batch size

## Budget Token Integration

**设计**: Budget编码为d_model维token，拼接到输入序列

**实现**:
1. Sinusoidal encoding: scalar budget → 256-D vector
2. MLP: 256-D → d_model (2048-D)
3. 在prefill阶段拼接到输入序列（只在prefill，不在decode）

**训练**: Budget encoder的MLP可训练，sinusoidal encoding固定

## Decode Phase

**关键设计**: Decode阶段使用prefill配置

**流程**:
1. Prefill阶段：运行controller，决定配置
2. Decode阶段：使用prefill配置，不重新运行controller
3. Budget token：只在prefill阶段添加

**优势**:
- 减少controller开销
- 保持配置一致性
- 更快的decode速度

## 关键设计决策

### 1. 为什么使用GRPO而不是监督学习？

- **监督学习**：需要ground truth labels（tier, top_k, num_blocks）
- **GRPO**：只需要reward信号（accuracy + latency），更灵活
- **Joint training**：两个阶段共享reward，无法用监督学习

### 2. 为什么两个阶段共享reward？

- Stage1和Stage2都会影响最终latency
- 需要全局优化accuracy-latency trade-off
- 共享reward可以让两个阶段协调工作

### 3. 为什么使用direct measurement而不是estimator？

- **更准确**: 实际测量而非估计
- **硬件特定**: 可以捕获硬件特定的latency特性
- **简化设计**: 不需要额外的estimator模型

### 4. 为什么Budget Token只在prefill阶段添加？

- Decode阶段不需要重新编码budget
- 减少计算开销
- 保持配置一致性

## 总结

1. **Joint Training**: 两个阶段共享reward信号，全局优化end-to-end性能
2. **Direct Measurement**: 使用hooks直接测量latency，不使用estimator
3. **Budget Token**: 编码为token拼接到输入序列，只在prefill阶段添加
4. **Decode Phase**: 使用prefill配置，不重新运行controller
5. **Reward设计**: 综合考虑accuracy和latency constraint

---

**最后更新**: 2026-01-10  
**维护者**: Controller Team
