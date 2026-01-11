# 训练监控指南

## 日志位置

### 1. CSV日志文件（推荐查看）

训练过程中会自动保存CSV文件，方便后续分析：

**训练历史**：
```
checkpoints/joint_controller/training_history.csv
```

**验证历史**：
```
checkpoints/joint_controller/validation_history.csv
```

**CSV格式**：
```csv
epoch,loss,reward_mean,reward_std,accuracy_mean,accuracy_std,latency_mean,latency_std,budget_violation_rate
1,0.1234,0.5678,0.1234,0.1234,0.0567,234.56,12.34,0.1234
2,0.2345,0.6789,0.2345,0.2345,0.0678,245.67,13.45,0.2345
...
```

### 2. TensorBoard日志（可视化）

**位置**：
```
checkpoints/joint_controller/tensorboard/
```

**查看方法**：
```bash
# 方法1：命令行
tensorboard --logdir=checkpoints/joint_controller/tensorboard

# 方法2：VSCode/Cursor
# 1. 安装TensorBoard扩展
# 2. Ctrl+Shift+P -> "TensorBoard: Launch"
# 3. 选择 checkpoints/joint_controller/tensorboard
```

**记录的指标**：
- `Train/Loss` - 训练损失
- `Train/Reward_Mean` - 平均奖励
- `Train/Accuracy_Mean` - 平均准确率
- `Train/Latency_Mean` - 平均延迟
- `Train/Budget_Violation_Rate` - Budget违反率
- `Val/...` - 验证指标（同上）

### 3. Weights & Biases（如果启用）

**查看方法**：
- 训练开始时会输出W&B URL
- 或访问 https://wandb.ai 查看项目

**启用方法**：
```bash
# 在 run_training.sh 中设置
USE_WANDB="true"
WANDB_PROJECT="molmo-controller"
```

### 4. 控制台输出

实时显示在终端，包括：
- 每个batch的loss、reward、accuracy
- 每个epoch的汇总统计
- 错误信息（如果有）

---

## 如何判断训练效果

### 关键指标解读

#### 1. Loss（损失）

**含义**：GRPO策略梯度损失，表示controller预测的配置与高reward配置的对齐程度。

**正常行为**：
- ✅ **初期**：可能较大（正数或负数），因为controller还在学习
- ✅ **中期**：应该逐渐稳定，波动减小
- ✅ **后期**：应该稳定在较小值附近

**异常行为**：
- ❌ **一直为0**：可能没有形成有效的groups
- ❌ **剧烈波动**：学习率可能太大，或batch_size太小
- ❌ **持续增大**：训练可能发散

**注意**：GRPO的loss可能是负数（因为使用 `-log_prob * advantage`），这是正常的。

#### 2. Reward（奖励）

**含义**：综合指标，结合accuracy和latency约束。

**计算公式**：
```
reward = accuracy - λ * latency_penalty - budget_violation_penalty
```

**正常行为**：
- ✅ **初期**：可能较低（甚至负数），因为accuracy低且可能违反budget
- ✅ **中期**：应该逐渐提升，波动减小
- ✅ **后期**：应该稳定在较高值

**异常行为**：
- ❌ **一直为负数**：可能accuracy太低或budget violation太多
- ❌ **剧烈波动**：可能group_size太小（方差大）
- ❌ **不提升**：可能需要调整reward function的权重

**目标**：reward应该逐渐提升并稳定在较高值。

#### 3. Accuracy（准确率）

**含义**：模型生成的答案与标准答案的匹配程度。

**正常行为**：
- ✅ **初期**：可能很低（0.0-0.1），因为模型还在学习
- ✅ **中期**：应该逐渐提升（0.2-0.5，取决于数据集）
- ✅ **后期**：应该稳定在合理水平

**异常行为**：
- ❌ **一直是0**：可能是metadata中没有answers（已修复）
- ❌ **不提升**：可能需要更多训练或调整超参数

**目标**：accuracy应该逐渐提升，最终达到合理水平（取决于数据集，text_vqa通常在0.3-0.5）。

#### 4. Latency（延迟）

**含义**：模型执行的实际延迟（ms）。

**正常行为**：
- ✅ **初期**：可能较高，因为controller可能选择高配置
- ✅ **中期**：应该逐渐降低（在满足budget的前提下）
- ✅ **后期**：应该稳定在budget附近

**目标**：latency应该接近但不超过budget。

#### 5. Budget Violation Rate（Budget违反率）

**含义**：实际延迟超过budget的配置比例。

**正常行为**：
- ✅ **初期**：可能较高（0.3-0.5），因为controller还在学习
- ✅ **中期**：应该逐渐降低
- ✅ **后期**：应该很低（<0.1，理想<0.05）

**目标**：budget violation rate应该逐渐降低，最终<5%。

---

## Loss和Reward抖动的原因

### 为什么Loss和Reward会抖动？

这是**正常的**，原因如下：

#### 1. GRPO的随机性

- 每个sample采样`group_size`次不同的配置
- 不同配置的reward可能差异很大
- 导致loss和reward的波动

#### 2. 小batch_size

- 当前`batch_size=1`（单GPU）
- 每个batch只有一个sample，统计量不稳定
- 导致loss和reward波动大

#### 3. 训练初期

- Controller还在学习，预测不稳定
- 生成的配置质量差异大
- 导致reward波动大

#### 4. 不同samples的难度差异

- 不同samples的难度不同
- 简单sample的reward可能高，困难sample的reward可能低
- 导致batch间的波动

### 如何判断抖动是否正常？

#### ✅ 正常抖动

- **短期波动大，长期趋势向上**：
  - Loss：虽然每个batch波动，但epoch平均loss逐渐降低
  - Reward：虽然每个batch波动，但epoch平均reward逐渐提升
  - Accuracy：虽然每个batch波动，但epoch平均accuracy逐渐提升

- **波动范围合理**：
  - Loss：在合理范围内波动（例如-2到2）
  - Reward：在合理范围内波动（例如-1到1）
  - 不会出现极端值（例如loss=1000）

#### ❌ 异常抖动

- **持续剧烈波动**：
  - Loss或Reward的波动范围非常大
  - 没有明显的改善趋势

- **发散**：
  - Loss持续增大
  - Reward持续降低
  - Accuracy持续降低

- **不学习**：
  - 所有指标都接近0或常数
  - 没有改善趋势

---

## 如何判断训练效果

### 1. 查看Epoch级别的趋势（最重要）

**不要只看单个batch的loss/reward**，要看**epoch平均值的趋势**：

```python
# 查看 training_history.csv
import pandas as pd
df = pd.read_csv('checkpoints/joint_controller/training_history.csv')

# 绘制趋势图
import matplotlib.pyplot as plt
plt.plot(df['epoch'], df['reward_mean'], label='Reward')
plt.plot(df['epoch'], df['accuracy_mean'], label='Accuracy')
plt.plot(df['epoch'], df['budget_violation_rate'], label='Budget Violation')
plt.legend()
plt.show()
```

**好的趋势**：
- ✅ Reward逐渐提升
- ✅ Accuracy逐渐提升
- ✅ Budget Violation Rate逐渐降低
- ✅ Latency稳定在budget附近

### 2. 查看TensorBoard曲线

**启动TensorBoard**：
```bash
tensorboard --logdir=checkpoints/joint_controller/tensorboard
```

**观察要点**：
- **平滑曲线**：使用TensorBoard的smoothing功能（例如0.6）
- **长期趋势**：不要只看短期波动，看整体趋势
- **验证指标**：验证集的指标更重要（不受训练噪声影响）

### 3. 关键判断标准

#### ✅ 训练效果好

1. **Reward趋势向上**：
   - Epoch 1-10: reward_mean ≈ -0.5
   - Epoch 11-20: reward_mean ≈ 0.0
   - Epoch 21-30: reward_mean ≈ 0.3
   - 持续提升

2. **Accuracy逐渐提升**：
   - Epoch 1-10: accuracy_mean ≈ 0.0-0.1
   - Epoch 11-20: accuracy_mean ≈ 0.1-0.2
   - Epoch 21-30: accuracy_mean ≈ 0.2-0.3
   - 持续提升

3. **Budget Violation Rate降低**：
   - Epoch 1-10: violation_rate ≈ 0.3-0.5
   - Epoch 11-20: violation_rate ≈ 0.2-0.3
   - Epoch 21-30: violation_rate ≈ 0.1-0.2
   - 持续降低

4. **Loss稳定**：
   - 虽然每个batch波动，但epoch平均loss逐渐稳定
   - 不会持续增大

#### ❌ 训练效果差

1. **所有指标不提升**：
   - Reward一直很低（<0）
   - Accuracy一直是0
   - Budget Violation Rate一直很高（>0.5）

2. **指标发散**：
   - Loss持续增大
   - Reward持续降低
   - Accuracy持续降低

3. **指标不学习**：
   - 所有指标接近常数
   - 没有改善趋势

---

## 训练监控最佳实践

### 1. 实时监控

**使用TensorBoard**（推荐）：
```bash
# 在另一个终端运行
tensorboard --logdir=checkpoints/joint_controller/tensorboard --port=6006
```

**观察指标**：
- 每5-10分钟查看一次
- 关注epoch级别的趋势，不要被batch级别的波动干扰

### 2. 定期检查CSV文件

```bash
# 查看最新的训练指标
tail -20 checkpoints/joint_controller/training_history.csv

# 或使用pandas分析
python -c "
import pandas as pd
df = pd.read_csv('checkpoints/joint_controller/training_history.csv')
print(df.tail(10))
print(f'\nReward trend: {df[\"reward_mean\"].tail(10).mean():.4f} (last 10 epochs)')
print(f'Accuracy trend: {df[\"accuracy_mean\"].tail(10).mean():.4f} (last 10 epochs)')
"
```

### 3. 设置检查点

**每10个epoch**：
- 检查epoch平均指标的趋势
- 如果趋势不好，考虑调整超参数

**每50个epoch**：
- 运行完整验证集评估
- 检查验证指标是否与训练指标一致

### 4. 早期停止条件

如果出现以下情况，考虑停止训练或调整超参数：

1. **Loss持续增大**（超过10个epoch）
2. **Reward持续降低**（超过10个epoch）
3. **Accuracy一直是0**（超过20个epoch）
4. **Budget Violation Rate一直很高**（>0.5，超过20个epoch）

---

## 常见问题

### Q: Loss和Reward波动很大，正常吗？

**A**: 是的，这是正常的，特别是：
- 训练初期（前10-20个epoch）
- batch_size=1时（统计量不稳定）
- GRPO的随机采样导致波动

**判断标准**：看**epoch平均值的趋势**，不要看单个batch的值。

### Q: 如何减少波动？

**A**: 
1. **增加batch_size**（如果内存允许）
2. **增加group_size**（例如从5增加到8或10）
3. **使用更大的学习率衰减**
4. **增加训练轮数**（让模型充分学习）

### Q: Reward一直是负数，正常吗？

**A**: 训练初期是正常的，因为：
- Accuracy可能很低（0.0-0.1）
- 可能违反budget（latency penalty）
- 导致reward为负

**判断标准**：看**趋势**，如果reward逐渐提升（从-1.0到-0.5到0.0），就是正常的。

### Q: Accuracy一直是0，怎么办？

**A**: 
1. **检查metadata**：确认answers字段是否正确保存
2. **检查生成文本**：查看模型实际生成的文本
3. **检查metric**：确认使用的metric是否正确
4. **耐心等待**：训练初期accuracy为0是正常的

### Q: 如何快速判断训练是否正常？

**A**: 检查以下3个指标的趋势（epoch级别）：
1. **Reward**：应该逐渐提升
2. **Accuracy**：应该逐渐提升（即使很慢）
3. **Budget Violation Rate**：应该逐渐降低

如果这3个指标都有改善趋势，训练就是正常的。

---

## 总结

### 日志位置总结

| 类型 | 位置 | 用途 |
|------|------|------|
| **CSV日志** | `checkpoints/joint_controller/training_history.csv` | 数据分析、趋势查看 |
| **TensorBoard** | `checkpoints/joint_controller/tensorboard/` | 可视化、实时监控 |
| **W&B** | https://wandb.ai | 实验管理、团队协作 |
| **控制台** | 终端输出 | 实时查看、错误诊断 |

### 判断训练效果的关键

1. **看趋势，不看绝对值**：关注epoch平均值的趋势，不要被batch级别的波动干扰
2. **多指标综合判断**：Reward、Accuracy、Budget Violation Rate都要看
3. **长期观察**：至少观察20-30个epoch才能判断趋势
4. **验证集更重要**：验证集的指标比训练集更可靠（不受训练噪声影响）

### 正常训练的特征

- ✅ Loss：短期波动，长期稳定或降低
- ✅ Reward：短期波动，长期提升
- ✅ Accuracy：逐渐提升（即使很慢）
- ✅ Budget Violation Rate：逐渐降低
- ✅ Latency：稳定在budget附近

如果满足以上特征，训练就是正常的，继续训练即可！

