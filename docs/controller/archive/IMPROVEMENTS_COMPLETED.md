# Controller Training Improvements - 完成总结

## 所有改进已完成

### 1. ✅ LanguageFeatureExtractor 优化确认
**状态**：无需修改
- `LanguageFeatureExtractor` 已经在使用模型的 `tokenizer` 和 `wte_layer`
- 没有重复计算，已经复用了模型模块
- 位置：`experiments/controller/feature_extractors.py` 第81-128行

### 2. ✅ Latency Budget 编码确认
**状态**：正确实现
- Latency budget 编码成 256 维向量（`budget_feat_dim=256`）
- 与模型隐藏维度匹配

### 3. ✅ Stage2 Controller 简化
**状态**：已完成
- **改动**：`Knob2Knob3Predictor` 现在只使用 `latency_token` + `budget_feat`
- **移除**：不再需要 `vision_feat` 和 `lang_feat`
- **原因**：第一层后的 latency token 已经包含了与 vision 和 language tokens 的交互信息
- **文件**：`experiments/controller/controller.py` 第237-365行

**新接口**：
```python
knob2_knob3_predictor(
    latency_token: torch.Tensor,  # (B, d_model) - 从transformer block提取
    budget_feat: torch.Tensor,     # (B, budget_feat_dim)
    insertion_position: torch.Tensor,  # (B,) 插入位置 (1-5)
)
```

### 4. ✅ Stage1 决定 Stage2 插入位置
**状态**：已完成
- **改动**：`Knob1PredictorBudgetLanguage` 现在同时预测：
  - Tier (low/medium/high)
  - Insertion position (1-5，表示在第1-5层之后插入)
- **文件**：`experiments/controller/controller.py` 第93-156行

**新输出**：
```python
{
    'tier_logits': (B, 3),  # low, medium, high
    'insertion_logits': (B, 5),  # 插入位置 1-5
}
```

### 5. ✅ 动态 Knob2 和 Knob3 选项
**状态**：已完成
- **Knob2 (Top-K)**：始终是 [4, 5, 6, 7, 8]，作用于插入位置之后的所有blocks
- **Knob3 (Blocks)**：根据插入位置动态调整
  - 插入位置1后：剩余15个blocks，选择11-15个 → 总blocks: 12-16
  - 插入位置2后：剩余14个blocks，选择11-14个 → 总blocks: 13-16
  - 插入位置3后：剩余13个blocks，选择11-13个 → 总blocks: 14-16
  - 插入位置4后：剩余12个blocks，选择11-12个 → 总blocks: 15-16
  - 插入位置5后：剩余11个blocks，选择11个 → 总blocks: 16

**实现**：
- `Knob2Knob3Predictor.get_knob3_options(insertion_position)` 方法动态计算选项
- `_select_blocks_by_importance()` 支持 `start_block` 参数
- `_set_top_k()` 支持 `start_layer` 参数（插入位置）

### 6. ✅ max_new_tokens 改为 64
**状态**：已完成
- 所有相关位置已更新：`_execute_model()`, `train_step()`, `validate()`
- 从 128 改为 64，减少训练时间

## 代码改动详情

### `controller.py`
1. **Knob1PredictorBudgetLanguage** (第93-156行)：
   - 新增 `insertion_head`，输出插入位置 logits
   - `forward()` 现在返回 `{'tier_logits', 'insertion_logits'}`

2. **Knob2Knob3Predictor** (第237-365行)：
   - 简化输入：只使用 `latency_token` 和 `budget_feat`
   - 新增 `insertion_position` 参数
   - 新增 `get_knob3_options()` 方法动态计算选项
   - `knob3_head` 固定输出 5 个选项（最大选项数）

### `joint_grpo_trainer.py`
1. **train_step()** (第721-1083行)：
   - 更新 Stage1 调用，处理 tier 和 insertion position
   - 从 transformer block 提取 latency token（根据 insertion position）
   - 更新 Stage2 调用，使用 latency token 而不是预提取的 features
   - 处理动态的 knob3 选项（不同插入位置有不同选项数量）
   - 更新 GRPO loss 计算，包含 tier 和 insertion position

2. **_execute_model()** (第361-576行)：
   - 新增 `insertion_position` 参数
   - 更新 `_set_top_k()` 调用，使用 `start_layer=insertion_position`
   - 更新 block selection 逻辑，考虑插入位置

3. **_select_blocks_by_importance()** (第148-185行)：
   - 新增 `start_block` 参数
   - 支持从指定位置开始选择 blocks

4. **validate()** (第1085-1254行)：
   - 更新以支持新的 Stage1 和 Stage2 接口
   - 提取 latency token（根据 insertion position）
   - 处理动态 knob3 选项

5. **max_new_tokens**：从 128 改为 64

### `train_joint_controller.py`
1. **Knob2Knob3Predictor初始化** (第121-126行)：
   - 更新参数：`latency_token_dim=2048`，移除 `vision_feat_dim` 和 `lang_feat_dim`
   - 新增 `max_insertion_position=5` 和 `total_blocks=16`

### `model_forward_with_dynamic_stage2.py`
1. **新文件**：创建了支持动态插入位置的 forward 函数
   - 支持在任意位置（1-5）插入 Stage2 controller
   - 根据插入位置提取 latency token
   - 动态应用 knob2 和 knob3

## 训练流程（更新后）

1. **Stage1**：
   - 输入：language features (从 prompt 提取) + latency budget (编码)
   - 输出：tier (low/medium/high) + insertion position (1-5)

2. **Vision Encoder**：
   - 根据 tier 处理图像，得到 vision tokens

3. **Transformer Blocks (到插入位置)**：
   - 运行 blocks 0 到 insertion_position-1
   - 提取最后一个 token (latency token) 作为 Stage2 输入

4. **Stage2**：
   - 输入：latency token (从插入位置 block 提取) + budget feat
   - 输出：knob2 (top_k: 4-8) + knob3 (动态选项，基于插入位置)

5. **剩余 Transformer Blocks**：
   - 应用 knob2 (top_k) 到插入位置之后的所有 blocks
   - 应用 knob3 (block selection) 选择剩余的 blocks

6. **生成**：
   - 使用配置好的模型生成文本
   - 测量 latency 和计算 accuracy

## 注意事项

1. **Latency Token 提取**：
   - 需要在每个样本的插入位置运行 forward pass
   - 当前实现是逐个样本处理（因为不同样本可能有不同插入位置）
   - 未来可以优化为批量处理（如果所有样本有相同插入位置）

2. **动态 Knob3 选项**：
   - 不同插入位置有不同的选项数量
   - 在 GRPO loss 计算中需要 mask 无效选项
   - 当前实现使用 padding 和 mask 来处理

3. **Block Selection**：
   - `_select_blocks_by_importance()` 现在支持从指定位置开始选择
   - 总是保留插入位置之前的所有 blocks

4. **Validation**：
   - Validation 使用确定性预测（argmax）
   - 与 training 使用相同的接口和逻辑

## 性能优化

1. **减少 max_new_tokens**：从 128 到 64，减少约 50% 的生成时间
2. **简化 Stage2**：移除 vision_feat 和 lang_feat，减少计算开销
3. **多GPU支持**：使用 `--use_multi_gpu` 可以并行处理多个样本

## 下一步

代码已经完成所有改进，可以开始训练。建议：

1. **测试训练**：先用小数据集测试，确保所有功能正常
2. **监控指标**：关注 reward、accuracy、latency 的变化
3. **调整超参数**：根据训练结果调整 learning rate、group_size 等

