# Controller Training Improvements

## 改进总结

### 1. LanguageFeatureExtractor 优化
**问题**：为什么不用模型本身的tokenizer？

**回答**：`LanguageFeatureExtractor` 已经在使用模型的tokenizer和wte_layer了（见`feature_extractors.py`第81-128行）。它直接使用：
- `self.tokenizer`：模型的tokenizer
- `self.wte_layer`：模型的word embedding layer (`transformer.wte`)

所以没有重复计算，已经复用了模型模块。

### 2. Latency Budget 编码
**确认**：latency budget编码成跟模型隐藏维度一样的向量token（`budget_feat_dim=256`），这是正确的。

### 3. Stage2 Controller 简化
**改进**：简化Stage2 controller，只使用latency token + budget_feat，移除vision_feat和lang_feat。

**原因**：
- 第一层后的latency token已经包含了与vision和language tokens的交互信息
- 减少计算开销
- 简化架构

**新接口**：
```python
knob2_knob3_predictor(
    latency_token: torch.Tensor,  # (B, d_model) - 从transformer block提取
    budget_feat: torch.Tensor,     # (B, budget_feat_dim)
    insertion_position: torch.Tensor,  # (B,) 插入位置 (1-5)
)
```

### 4. Stage1 决定 Stage2 插入位置
**新功能**：Stage1现在同时预测：
- Tier (low/medium/high)
- Insertion position (1-5，表示在第1-5层之后插入)

**Stage1输出**：
```python
{
    'tier_logits': (B, 3),  # low, medium, high
    'insertion_logits': (B, 5),  # 插入位置 1-5
}
```

### 5. 动态 Knob2 和 Knob3 选项
**改进**：根据插入位置动态调整knob2和knob3的选项：

- **Knob2 (Top-K)**：始终是 [4, 5, 6, 7, 8]，作用于插入位置之后的所有blocks
- **Knob3 (Blocks)**：根据插入位置动态调整
  - 插入位置1后：剩余15个blocks，选择11-15个 → 总blocks: 12-16
  - 插入位置2后：剩余14个blocks，选择11-14个 → 总blocks: 13-16
  - 插入位置3后：剩余13个blocks，选择11-13个 → 总blocks: 14-16
  - 插入位置4后：剩余12个blocks，选择11-12个 → 总blocks: 15-16
  - 插入位置5后：剩余11个blocks，选择11个 → 总blocks: 16

**实现**：`Knob2Knob3Predictor.get_knob3_options(insertion_position)` 方法动态计算选项。

### 6. max_new_tokens 改为 64
**改进**：将`max_new_tokens`从128改为64，减少训练时间。

## 代码改动

### `controller.py`
1. **Knob1PredictorBudgetLanguage**：
   - 新增`insertion_head`，输出插入位置logits
   - `forward()`现在返回`{'tier_logits', 'insertion_logits'}`

2. **Knob2Knob3Predictor**：
   - 简化输入：只使用`latency_token`和`budget_feat`
   - 新增`insertion_position`参数
   - 新增`get_knob3_options()`方法动态计算选项

### `joint_grpo_trainer.py`
1. **train_step()**：
   - 更新Stage1调用，处理tier和insertion position
   - 更新Stage2调用，使用latency token而不是预提取的features
   - 更新`_execute_model()`调用，传递insertion position

2. **max_new_tokens**：从128改为64

### `train_joint_controller.py`
1. **Knob2Knob3Predictor初始化**：
   - 更新参数：`latency_token_dim=2048`，移除`vision_feat_dim`和`lang_feat_dim`
   - 新增`max_insertion_position=5`和`total_blocks=16`

## 待完成的工作

由于改动较大，以下部分需要进一步实现：

1. **`joint_grpo_trainer.py`的`train_step()`方法**：
   - 需要更新以支持新的Stage1和Stage2接口
   - 需要从transformer block提取latency token
   - 需要根据insertion position动态处理knob3选项

2. **`model_forward_with_stage2.py`**：
   - 需要支持动态插入位置（不仅仅是第1层后）
   - 需要根据insertion position提取latency token

3. **GRPO loss计算**：
   - 需要处理insertion position的logits和actions
   - 需要处理动态的knob3选项

## 使用建议

1. **训练时**：使用新的接口，Stage1预测tier和insertion position，Stage2使用latency token
2. **推理时**：可以根据任务复杂度选择不同的插入位置
3. **性能优化**：减少max_new_tokens可以显著加速训练

