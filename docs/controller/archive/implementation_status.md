# 实现状态和待完成工作

## ✅ 已完成

1. **修复BaseExperiment问题**
   - 创建了`model_loader.py`，直接加载模型和tokenizer

2. **更新Knob选项值**
   - Knob2 (Top-K): 4, 5, 6, 7, 8 ✅
   - Knob3 (Blocks): 12, 13, 14, 15, 16 ✅

3. **第一层固定Top-K**
   - 更新了`_set_top_k`方法，第一个block固定top_k=8 ✅

4. **Reward设计文档**
   - 创建了详细的reward设计原理文档 ✅

5. **Latency Budget分析**
   - 分析了profiling结果，确定了budget范围：170ms - 380ms ✅

---

## ⏳ 待实现

### 1. Stage2插入位置：第一个transformer block之后

**当前状态**：
- Stage2在vision encoder + projector之后预测
- 需要改为：在第一个transformer block之后预测

**需要修改**：
1. 修改`joint_grpo_trainer.py`中的`train_step`方法
2. 实现以下流程：
   ```
   Input → Vision Encoder → Projector → First Block (fixed top_k=8) 
   → Extract features from first block output 
   → Stage2 Controller predicts knob2 & knob3 
   → Apply knob2 & knob3 to subsequent blocks 
   → Continue forward pass
   ```

3. 需要修改模型的前向传播，支持：
   - 运行到第一个block
   - 提取中间特征
   - 应用配置
   - 继续运行

**实现方案**：
- 方案A：修改模型forward，支持返回中间特征
- 方案B：手动实现前向传播（更灵活，但更复杂）

### 2. 完整的Online Training流程

**当前状态**：
- `train_step`中有placeholder代码
- 需要实现完整的：
  - 数据加载（images + prompts）
  - 实际运行模型获取accuracy
  - 使用latency estimator估计latency
  - 计算reward
  - 更新controller

**需要实现**：
1. 创建proper dataset和dataloader
2. 实现模型执行流程（支持Stage2在第一个block之后）
3. 实现accuracy计算
4. 集成latency estimator
5. 完整的训练循环

### 3. Latency Budget采样

**当前状态**：
- 确定了budget范围：170ms - 380ms
- 需要在训练时采样budget值

**需要实现**：
- 在训练时为每个样本分配latency budget
- 可以从范围内均匀采样，或使用预定义列表

---

## 📋 实现优先级

### 高优先级（必须实现）
1. ✅ Knob选项值更新（已完成）
2. ✅ 第一层固定top_k=8（已完成）
3. ⏳ Stage2在第一个block之后插入
4. ⏳ 完整的online training流程

### 中优先级（重要但可以后续优化）
5. ⏳ Latency budget采样策略
6. ⏳ 训练数据加载和预处理

### 低优先级（可以后续添加）
7. ⏳ 验证和评估脚本
8. ⏳ 可视化训练过程

---

## 🔧 技术难点

### 1. Stage2插入位置实现

**挑战**：
- 需要修改模型前向传播
- 需要提取中间特征
- 需要支持动态配置应用

**解决方案**：
- 使用模型的`forward`方法，但需要支持返回中间特征
- 或者手动实现前向传播步骤

### 2. Accuracy计算

**挑战**：
- 需要ground truth答案
- 需要根据数据集类型选择合适的metric

**解决方案**：
- 使用BaseExperiment中的accuracy计算逻辑
- 或者直接使用数据集提供的evaluation函数

---

## 📝 下一步行动

1. **实现Stage2插入位置**（最优先）
   - 修改`joint_grpo_trainer.py`
   - 实现第一个block之后的特征提取
   - 实现动态配置应用

2. **实现完整的训练流程**
   - 创建dataset和dataloader
   - 实现模型执行和accuracy计算
   - 集成所有组件

3. **测试和验证**
   - 在小规模数据上测试
   - 验证reward计算正确性
   - 验证训练收敛

