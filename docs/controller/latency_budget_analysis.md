# Latency Budget范围分析

## 分析结果（基于3个数据集）

> **新增：仅对 prefill latency 的统计（核心实验全量结果）**
>
> 从 `results/core_exp_h100` 全部 JSON 重新汇总 `T_LLM_prefill`（总计 1,638,588 条样本）：
> - min: **68.7 ms**
> - p5 / p25 / median / p75 / p95: **102.3 / 119.7 / 135.0 / 151.7 / 175.3 ms**
> - max: **607.7 ms**（极端长尾）
> - mean / std: **136.3 / 22.2 ms**
>
> **建议（仅 prefill 约束时）**：
> - 训练/验证的预算采样范围：**100 ms – 180 ms**（覆盖 p5–p95，剔除长尾）
> - 如果需要更保险的上界，可放宽到 **200 ms**，但仍保持主要约束在 prefill。
>
> **decode latency 暂不计入预算**：保持当前“硬约束只看 prefill”策略，decode 长度不确定，先不做预算判定。
>
> **若未来发现 decode 主导总延迟**（例如长回答任务），可考虑：
> - 软罚项：`latency_penalty = max(0, prefill - budget_prefill) + α * max(0, decode - decode_target)`，其中 α 取 0.1~0.2；`decode_target` 可按生成 token 数（如 16）或经验均值设定。
> - 或直接缩短 `max_new_tokens`（如 8~16）来间接压制 decode 延迟。

### 数据集分析

#### Text-VQA
- **Pareto frontier点数**: 14
- **最高accuracy**: 0.7680 @ **371.52ms** (config: high tier, top_k=8, blocks=16)
- **最低accuracy**: 0.3778 @ **218.56ms** (config: low tier, top_k=4, blocks=12)
- **Latency范围**: 218.56ms - 371.52ms

#### COCO-2014-VQA
- **Pareto frontier点数**: 15
- **最高accuracy**: 0.8255 @ **294.69ms** (config: high tier, top_k=8, blocks=16)
- **最低accuracy**: 0.6535 @ **166.49ms** (config: low tier, top_k=4, blocks=12)
- **Latency范围**: 166.49ms - 294.69ms

#### OKVQA
- **Pareto frontier点数**: 11
- **最高accuracy**: 0.6443 @ **289.24ms** (config: medium tier, top_k=8, blocks=16)
- **最低accuracy**: 0.3662 @ **177.90ms** (config: low tier, top_k=4, blocks=12)
- **Latency范围**: 177.90ms - 289.24ms

---

## 聚合结果

### 跨数据集范围
- **最小latency (下界)**: **166.49ms** (COCO-2014-VQA最低accuracy点)
- **最大latency (上界)**: **371.52ms** (Text-VQA最高accuracy点)

### 统计值
- **平均最小latency**: 187.65ms
- **平均最大latency**: 318.48ms
- **中位数最小latency**: 177.90ms
- **中位数最大latency**: 294.69ms

---

## 建议的Latency Budget范围

### 训练时使用的Budget范围

**建议范围**: **170ms - 380ms**

**理由**：
- **下界 (170ms)**: 略高于最小latency (166.49ms)，确保有足够的配置空间
- **上界 (380ms)**: 略高于最大latency (371.52ms)，覆盖所有Pareto frontier点

### 具体Budget值建议

可以在这个范围内采样多个budget值进行训练：

```python
# 建议的budget值
latency_budgets = [
    170,   # 接近下界
    200,   # 低budget
    250,   # 中等budget
    300,   # 高budget
    350,   # 接近上界
    380,   # 上界
]
```

或者使用均匀采样：
```python
import numpy as np
latency_budgets = np.linspace(170, 380, num=10)  # 10个均匀分布的budget值
```

---

## 使用方式

在训练时，可以为每个样本随机分配一个latency budget：

```python
# 从范围内随机采样budget
latency_budget = np.random.uniform(170, 380)

# 或者从预定义的列表中选择
latency_budget = np.random.choice([170, 200, 250, 300, 350, 380])
```

这样可以训练controller适应不同的latency budget，学习在给定budget下最大化accuracy。

