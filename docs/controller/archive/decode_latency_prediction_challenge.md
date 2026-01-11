# Decode Latency Prediction Challenge: Output Tokens vs Semantic Features

## 问题核心

**用户的观察**：
1. Decode latency与任务类型相关（QA任务短，captioning任务长）
2. 如果能从prompt提取语义信息，可以预测`output_tokens`
3. 然后结合positioned decode per-token latency计算total decode latency
4. **但是误差会很大**，因为每个token的latency是30-45ms

**关键问题**：即使能准确预测任务类型，`output_tokens`的方差也很大，导致total decode latency预测误差被放大。

## 数据分析

### Output Tokens分布（来自实际数据）

| 任务类型 | Mean | Median | Std | CV | 范围 |
|---------|------|--------|-----|-----|------|
| **VQA** (coco_2014_vqa) | 2.22 | 2.0 | ~0.6 | 27% | 2-6 tokens |
| **Captioning** (coco_caption) | 54.67 | 64.0 | ~10 | 18% | 30-64 tokens |
| **Doc QA** | 5.51 | 5.0 | ~3 | 54% | 2-28 tokens |
| **Text VQA** | 3.44 | 3.0 | ~2 | 58% | 2-15 tokens |

**关键发现**：
1. **任务类型差异巨大**：VQA平均2.22 tokens，Captioning平均54.67 tokens（25倍差异）
2. **方差很大**：即使同一任务类型，CV在18%-58%之间
3. **分布偏斜**：很多任务的中位数远小于均值（说明有长尾分布）

### Decode Per-Token Latency

| Output Tokens范围 | Per-Token Latency (ms) |
|------------------|----------------------|
| 2 tokens | ~24.6 ms/token |
| 3-5 tokens | ~34.4 ms/token |
| 6-10 tokens | ~41.1 ms/token |
| 11-20 tokens | ~41.5 ms/token |
| 21+ tokens | ~45.2 ms/token |

**关键发现**：
- 每个token的latency很高（24-45ms）
- 随position增长（KV cache影响）

### 误差放大效应

**如果预测`output_tokens`的误差为±N tokens，total decode latency的误差为**：

```
Error(total_decode) = Error(output_tokens) × avg_per_token_latency
```

**具体计算**：

| 预测误差 | VQA任务 (28.6ms/token) | Captioning任务 (45ms/token) |
|---------|----------------------|---------------------------|
| ±1 token | ±28.6ms | ±45ms |
| ±2 tokens | ±57.2ms | ±90ms |
| ±5 tokens | ±143ms | ±225ms |
| ±10 tokens | ±286ms | ±450ms |

**结论**：即使能准确预测任务类型，`output_tokens`的预测误差也会导致**巨大的**decode latency误差。

## 方案分析

### 方案1：预测Output Tokens + Positioned Decode Latency

**思路**：
1. 从prompt提取语义特征
2. 预测`output_tokens`（期望值）
3. 使用positioned decode per-token latency计算total decode latency

**问题**：
1. **预测误差大**：
   - 即使能准确识别任务类型，`output_tokens`的方差也很大（CV=18%-58%）
   - 例如：VQA任务，mean=2.22，但可能生成2-6 tokens
   - 预测误差±2 tokens → latency误差±57ms

2. **误差放大**：
   - 每个token的latency很高（30-45ms）
   - 预测误差会被放大
   - 对于captioning任务，误差可能达到±225ms（±5 tokens）

3. **网络复杂度**：
   - 需要embedding layer处理prompt
   - 需要预测`output_tokens`的head
   - 增加overhead和参数量

**评估**：
- ❌ **不推荐**：误差太大，不值得增加复杂度

### 方案2：预测Output Tokens分布（期望值+方差）

**思路**：
1. 预测`output_tokens`的分布：`μ`（期望值）和`σ`（标准差）
2. 使用概率方法计算total decode latency：
   ```
   E[total_decode] = sum over positions: E[decode_per_token(pos)] × P(output_tokens >= pos)
   ```

**问题**：
1. **复杂度高**：需要预测分布参数，网络更复杂
2. **仍然有误差**：即使知道分布，实际值仍然不确定
3. **计算复杂**：需要积分计算期望值

**评估**：
- ⚠️ **可能可行**：但复杂度高，需要验证是否值得

### 方案3：保守估计（推荐）

**思路**：
1. **不预测total decode latency**
2. **只预测prefill latency**（主要指标，确定性高）
3. **Decode latency使用保守估计**：
   ```python
   # 使用任务类型的经验值（从数据统计）
   task_type = infer_task_type(prompt)  # VQA, Captioning, etc.
   max_output_tokens = get_max_output_tokens(task_type)  # P95或P99
   avg_decode_per_token = get_avg_decode_per_token(config)  # 从positioned latency估算
   
   # 保守估计：使用最大可能的output_tokens
   T_decode_estimate = max_output_tokens × avg_decode_per_token
   
   # Budget检查：使用保守估计
   if T_prefill + T_decode_estimate <= latency_budget:
       # Configuration可行
   ```

**优点**：
1. **简单高效**：不需要预测`output_tokens`
2. **保守可靠**：使用P95/P99，确保不超budget
3. **低overhead**：只需要任务类型识别（可以用简单的keyword matching）

**缺点**：
1. **可能过于保守**：对于短回答任务，可能浪费budget
2. **需要任务类型识别**：但可以用简单方法（keyword matching）

**评估**：
- ✅ **推荐**：简单、可靠、低overhead

### 方案4：分层估计

**思路**：
1. **第一层**：使用prefill latency进行初步筛选
   ```python
   if T_prefill > latency_budget * 0.8:
       # 直接拒绝，不需要考虑decode
       return False
   ```

2. **第二层**：对于通过第一层的configurations，使用保守估计
   ```python
   remaining_budget = latency_budget - T_prefill
   max_decode_tokens = remaining_budget / avg_decode_per_token
   
   # 检查是否可能满足（使用任务类型的最大output_tokens）
   if max_decode_tokens >= get_max_output_tokens(task_type):
       # Configuration可行
   ```

**优点**：
1. **两阶段筛选**：先快速筛选，再精确检查
2. **效率高**：大部分configurations在第一层就被拒绝
3. **准确**：第二层使用保守估计，确保不超budget

**评估**：
- ✅ **推荐**：结合了效率和准确性

## 推荐方案

### 最终推荐：方案3（保守估计）+ 方案4（分层估计）

**设计**：

```python
def check_budget_feasibility(
    latency_estimator,
    config,
    latency_budget,
    task_type=None,  # 可选：从prompt推断
):
    """
    检查configuration是否满足latency budget。
    
    策略：
    1. 主要使用prefill latency（确定性高）
    2. Decode latency使用保守估计（基于任务类型）
    """
    # 1. 预测prefill latency（主要指标）
    T_prefill = latency_estimator.predict_prefill(config)
    
    # 2. 第一层筛选：prefill必须小于budget的80%
    if T_prefill > latency_budget * 0.8:
        return False, "Prefill exceeds budget"
    
    # 3. 计算剩余budget
    remaining_budget = latency_budget - T_prefill
    
    # 4. 保守估计decode latency
    # 4.1 获取任务类型的最大output_tokens（P95或P99）
    if task_type:
        max_output_tokens = TASK_MAX_OUTPUT_TOKENS.get(task_type, 20)  # 默认20
    else:
        max_output_tokens = 20  # 保守默认值
    
    # 4.2 估算平均decode per-token latency（使用positioned latency的期望值）
    # 假设output_tokens分布，计算期望的per-token latency
    avg_decode_per_token = latency_estimator.estimate_avg_decode_per_token(
        config, 
        expected_output_tokens=max_output_tokens
    )
    
    # 4.3 保守估计：使用最大可能的output_tokens
    T_decode_estimate = max_output_tokens × avg_decode_per_token
    
    # 5. 检查是否满足budget
    if T_prefill + T_decode_estimate <= latency_budget:
        return True, "Feasible"
    else:
        return False, "Decode estimate exceeds remaining budget"

# 任务类型的最大output_tokens（从数据统计）
TASK_MAX_OUTPUT_TOKENS = {
    'vqa': 6,           # P95 of coco_2014_vqa
    'captioning': 64,   # Max of coco_caption
    'doc_qa': 28,       # Max of doc_qa
    'text_vqa': 15,     # P95 of text_vqa
    'default': 20,      # 保守默认值
}
```

**关键设计点**：

1. **Prefill latency是主要指标**：
   - 确定性高（R² > 0.9）
   - 直接用于budget检查
   - 占total latency的大部分（60-80%）

2. **Decode latency使用保守估计**：
   - 不预测`output_tokens`（误差太大）
   - 使用任务类型的最大可能值（P95或P99）
   - 结合positioned decode per-token latency估算

3. **任务类型识别**（可选）：
   - 可以用简单的keyword matching
   - 或者从prompt提取轻量级特征
   - 不需要embedding layer（低overhead）

4. **分层筛选**：
   - 第一层：prefill latency快速筛选
   - 第二层：保守估计精确检查

## 关于Text Embedding的结论

### 是否需要Text Embedding？

**结论**：**不需要**，原因：

1. **Prefill latency**：
   - 主要取决于token数量，而非语义
   - 当前设计（只用`text_tokens`数量）已经足够

2. **Decode latency**：
   - 与prompt内容无关（只与position和配置相关）
   - 不需要text embedding

3. **Output tokens预测**：
   - 即使能预测，误差也很大（CV=18%-58%）
   - 误差会被放大（每个token 30-45ms）
   - 不值得增加复杂度

### 如果需要任务类型识别

**轻量级方案**（不需要embedding）：

```python
def infer_task_type(prompt: str) -> str:
    """简单的任务类型识别（基于keyword matching）"""
    prompt_lower = prompt.lower()
    
    if any(kw in prompt_lower for kw in ['describe', 'caption', 'what do you see']):
        return 'captioning'
    elif any(kw in prompt_lower for kw in ['what', 'how', 'why', 'when', 'where', '?']):
        return 'vqa'
    elif any(kw in prompt_lower for kw in ['document', 'passage', 'text']):
        return 'doc_qa'
    else:
        return 'default'
```

**优点**：
- 零overhead（不需要embedding计算）
- 简单可靠
- 足够用于保守估计

## 总结

1. **不预测total decode latency**：误差太大（±100-200ms）
2. **只预测prefill latency**：主要指标，确定性高
3. **Decode latency使用保守估计**：基于任务类型的最大output_tokens
4. **不需要text embedding**：当前设计已经足够
5. **任务类型识别**（可选）：可以用简单的keyword matching

**最终设计**：
- Prefill latency：高精度预测（R² > 0.9）
- Decode latency：保守估计（基于任务类型和positioned latency）
- Text features：只用`text_tokens`数量（不需要embedding）

---

**下一步**：
1. 实现保守估计的budget检查
2. 验证任务类型识别的准确性
3. 评估保守估计的budget利用率（是否过于保守）



