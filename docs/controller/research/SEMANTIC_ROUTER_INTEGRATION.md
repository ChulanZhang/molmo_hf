# Semantic Router集成调研

## 概述

考虑将 [Semantic Router](https://github.com/aurelio-labs/semantic-router) 集成到Knob1预测器（选项B: Budget + Language）中，以实现更智能的语义路由决策。

## Semantic Router简介

Semantic Router是一个超快速的AI决策层，使用语义向量空间进行路由决策，而不是等待慢速的LLM生成。

### 核心特性

1. **超快速决策**: <10ms的决策时间
2. **语义理解**: 基于语义向量空间，而非关键词匹配
3. **多模态支持**: 支持文本和图像
4. **易于集成**: 简单的API接口

### 工作原理

```python
from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import CohereEncoder

# 定义路由
tier_low = Route(
    name="tier_low",
    utterances=[
        "simple question",
        "short answer needed",
        "quick response",
        "basic query",
    ],
)

tier_medium = Route(
    name="tier_medium",
    utterances=[
        "moderate complexity",
        "medium detail",
        "standard question",
    ],
)

tier_high = Route(
    name="tier_high",
    utterances=[
        "complex question",
        "detailed analysis needed",
        "comprehensive answer",
        "in-depth explanation",
    ],
)

# 创建路由层
encoder = CohereEncoder()
router = SemanticRouter(
    encoder=encoder,
    routes=[tier_low, tier_medium, tier_high],
)

# 使用
result = router("I need a quick answer")
print(result.name)  # 'tier_low'
```

## 集成方案

### 方案1: 直接替换MLP

**思路**: 用Semantic Router替换Knob1预测器中的MLP部分

**架构**:
```python
class Knob1PredictorWithSemanticRouter(nn.Module):
    def __init__(self):
        super().__init__()
        # Budget encoder (保留)
        self.budget_encoder = BudgetEncoder()
        
        # Semantic Router (替换MLP)
        self.semantic_router = SemanticRouter(
            encoder=encoder,
            routes=[tier_low, tier_medium, tier_high],
        )
    
    def forward(self, language_prompt, latency_budget):
        # Budget feature
        budget_feat = self.budget_encoder(latency_budget)
        
        # Semantic routing
        route_result = self.semantic_router(language_prompt)
        
        # Combine budget and semantic routing
        tier = self.combine(budget_feat, route_result)
        
        return tier
```

**优点**:
- 语义理解能力强
- 决策速度快
- 易于训练（基于utterances）

**缺点**:
- 需要定义好的utterances
- 可能不如端到端训练灵活

### 方案2: 混合方案

**思路**: Semantic Router作为特征提取器，然后与Budget特征融合

**架构**:
```python
class Knob1PredictorHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        # Budget encoder
        self.budget_encoder = BudgetEncoder()
        
        # Semantic Router (特征提取)
        self.semantic_router = SemanticRouter(...)
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(budget_dim + semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
    
    def forward(self, language_prompt, latency_budget):
        # Budget feature
        budget_feat = self.budget_encoder(latency_budget)
        
        # Semantic routing (获取语义特征)
        route_result = self.semantic_router(language_prompt)
        semantic_feat = route_result.embedding  # 或route_result.score
        
        # Fusion
        combined = torch.cat([budget_feat, semantic_feat], dim=-1)
        tier_logits = self.fusion(combined)
        
        return tier_logits
```

**优点**:
- 结合了语义路由和端到端训练
- 更灵活
- 可以学习Budget和Language的交互

**缺点**:
- 复杂度稍高
- 需要训练fusion层

### 方案3: 多阶段路由

**思路**: 先用Semantic Router做粗分类，再用Budget做细调

**架构**:
```python
class Knob1PredictorMultiStage(nn.Module):
    def __init__(self):
        super().__init__()
        # Stage 1: Semantic Router (粗分类)
        self.semantic_router = SemanticRouter(...)
        
        # Stage 2: Budget-based refinement
        self.budget_refiner = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
    
    def forward(self, language_prompt, latency_budget):
        # Stage 1: Semantic routing
        route_result = self.semantic_router(language_prompt)
        semantic_tier = route_result.name  # 'tier_low', 'tier_medium', 'tier_high'
        
        # Stage 2: Budget refinement
        budget_logits = self.budget_refiner(latency_budget)
        
        # Combine (可以加权或学习权重)
        final_tier = self.combine(semantic_tier, budget_logits)
        
        return final_tier
```

**优点**:
- 两阶段决策，更精细
- 语义路由提供先验，Budget提供约束

**缺点**:
- 复杂度最高
- 需要设计combine策略

## 实施建议

### 推荐方案: 方案2（混合方案）

**理由**:
1. 平衡了语义理解和端到端训练
2. 可以学习Budget和Language的交互
3. 灵活性高，易于调优

### 实施步骤

1. **定义Routes**: 为每个tier定义代表性的utterances
   ```python
   tier_low = Route(
       name="tier_low",
       utterances=[
           "what is",
           "who is",
           "where is",
           "simple question",
           "quick answer",
       ],
   )
   ```

2. **选择Encoder**: 
   - 推荐: `CohereEncoder` 或 `OpenAIEncoder`（快速）
   - 本地: `HuggingFaceEncoder`（完全本地）

3. **集成到Controller**:
   ```python
   class Knob1PredictorSemanticRouter(nn.Module):
       def __init__(self, budget_dim=128, hidden_dim=64):
           super().__init__()
           self.budget_encoder = BudgetEncoder()
           self.semantic_router = SemanticRouter(...)
           self.fusion = nn.Sequential(...)
   ```

4. **训练策略**:
   - 可以先用Semantic Router预训练
   - 然后端到端fine-tune fusion层

## 优势分析

### 相比纯MLP

1. **语义理解**: 基于语义向量空间，而非简单的特征拼接
2. **可解释性**: Routes的utterances提供了可解释性
3. **快速决策**: <10ms的决策时间
4. **易于扩展**: 添加新的tier只需添加新的Route

### 相比纯Budget-Only

1. **内容感知**: 可以利用prompt的语义信息
2. **更智能**: 可以根据问题复杂度选择tier
3. **准确性**: 预期比Budget-Only更准确

## 注意事项

1. **Utterances质量**: Routes的utterances质量直接影响性能
2. **Encoder选择**: 需要平衡速度和准确性
3. **训练数据**: 需要足够的训练数据来fine-tune fusion层
4. **Overhead**: 虽然快速，但仍需测量实际overhead

## 实验计划

1. **Baseline**: Budget-Only方案
2. **对比**: Budget + Language (MLP) vs Budget + Language (Semantic Router)
3. **评估指标**: 
   - Accuracy (tier预测准确率)
   - Latency (决策时间)
   - Overhead (相对inference时间)

## 参考资源

- **GitHub**: [https://github.com/aurelio-labs/semantic-router](https://github.com/aurelio-labs/semantic-router)
- **文档**: [https://aurelio.ai/semantic-router](https://aurelio.ai/semantic-router)
- **示例**: 查看GitHub上的notebooks

---

**状态**: 🔄 调研中
**优先级**: 中等（在Budget-Only实现后）







