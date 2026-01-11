# 实验设计核对文档

## 实验目标

**目标**：使用GRPO进行online training，给定latency budget，寻找满足latency budget并且最大化accuracy的模型配置。

## 实验思路核对

### 1. 数据来源

**问题**：这个实验不是用的accuracy latency profiling实验的数据对吧？

**回答**：
- ❌ **不是**用profiling实验的数据（那是用于监督学习的）
- ✅ **是**用**online training**，即：
  - 给定一个latency budget
  - Controller预测配置（knob1, knob2, knob3）
  - **实际运行模型**得到accuracy
  - 使用latency estimator估计latency（因为batch size不是1，无法准确测量）
  - 计算reward（accuracy + latency constraint）
  - 使用GRPO更新controller

### 2. Knob配置

#### Knob1 (Stage1): Vision Tokens Tier
- **档位**: `low`, `medium`, `high` (3个选择)
- **决策时机**: Before vision encoder
- ✅ **正确**

#### Knob2 (Stage2): MoE Top-K
- **用户描述**: 4, 5, 6, 7, 8 (5个选择)
- **当前代码**: 4, 6, 8, 10, 12 (5个选择)
- **需要确认**: 用户想要的是 4, 5, 6, 7, 8 还是 4, 6, 8, 10, 12？
- **建议**: 使用 4, 5, 6, 7, 8（更细粒度）

#### Knob3 (Stage2): Transformer Blocks
- **用户描述**: 12, 13, 14, 15, 16 (5个选择)
- **当前代码**: 8, 10, 12, 14, 16 (5个选择)
- **需要确认**: 用户想要的是 12, 13, 14, 15, 16 还是 8, 10, 12, 14, 16？
- **建议**: 使用 12, 13, 14, 15, 16（更细粒度，且更接近全模型）

### 3. Stage2 Controller插入位置

**用户描述**: Stage2 controller插入位置在**第一个transformer block之后**

**当前实现**: 需要确认

**设计**:
- Stage2在vision encoder + projector之后
- Controller在第一个transformer block之后预测knob2和knob3
- 这样latency token会跟vision token和language token做attention得到一些信息

**需要实现**:
1. Vision encoder + projector → 得到vision features
2. 第一个transformer block处理（固定topk=8）
3. **Stage2 Controller预测**（基于第一个block的输出）
4. 应用knob2和knob3到后续blocks

### 4. 第一层固定Top-K

**用户描述**: 第一层固定topk=8，后面动态改变

**设计**:
- 第一个transformer block: 固定 `top_k=8`
- 后续blocks: 使用controller预测的top_k值（knob2）
- 这样可以控制变量，只改变后面的blocks

### 5. Latency Estimator

**用户描述**: 因为batch size不是1，所以latency测的是不准的，所以要用latency estimator

**设计**:
- ✅ 使用latency estimator估计latency
- ✅ 可以用大的batch size加速训练
- ✅ 不需要实际测量latency（batch size > 1时测量不准确）

### 6. Reward设计

**用户描述**: Reward会包含latency是否满足，和LLM decode的accuracy

**当前实现**:
```python
reward = accuracy_reward 
       - latency_penalty 
       - budget_violation_penalty  # 硬约束
       - complexity_penalty 
       + efficiency_bonus
```

**需要确认**:
- ✅ Accuracy: LLM decode的accuracy（模型输出质量）
- ✅ Latency constraint: 是否满足budget（硬约束）
- 是否需要其他项（complexity_penalty, efficiency_bonus）？

## 需要确认的问题

1. **Knob2选项**: 4,5,6,7,8 还是 4,6,8,10,12？
2. **Knob3选项**: 12,13,14,15,16 还是 8,10,12,14,16？
3. **Stage2插入位置**: 是否在第一个transformer block之后？
4. **第一层固定topk**: 是否固定为8？
5. **Reward设计**: 是否只需要accuracy和latency constraint？

## 待实现的功能

1. ✅ 修复BaseExperiment问题（已完成）
2. ⏳ 更新Knob2和Knob3的选项值
3. ⏳ 实现Stage2在第一个transformer block之后插入
4. ⏳ 实现第一层固定topk=8
5. ⏳ 确认reward设计
6. ⏳ 实现完整的online training流程（实际运行模型获取accuracy）

