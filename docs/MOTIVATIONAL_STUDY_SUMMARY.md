# Motivational Study 实验总结

## 已完成的工作

### 1. 实验计划文档
- ✅ `EXPERIMENT_PLAN.md`: 详细的实验计划，包含 5 个实验的完整说明

### 2. 核心代码实现

#### 基础框架
- ✅ `experiments/motivate/base_experiment.py`: 
  - 模型加载
  - 数据加载器构建
  - 延迟测量工具
  - FLOPs 估算
  - 统计计算

#### 实验实现
- ✅ `experiments/motivate/run_unified_experiments.py`: 
  - 统一实现了 Phase 1 (Exp 1 & 3) 和 Phase 2 (Exp 2 & 4a) 的逻辑
  - 包含延迟分布测量、组件分析、Scaling 测试等功能

- ✅ `experiments/motivate/exp4_token_vs_latency.py`: 
  - 独立的 Token vs Latency 实验脚本 (包含 4A 和 4B)

- ✅ `experiments/motivate/exp5_token_comparison.py`:
  - Token 对比分析工具

#### 运行脚本
- ✅ `experiments/motivate/run_phase1.sh`: 运行 Phase 1 (Dataset Profiling)
- ✅ `experiments/motivate/run_phase2.sh`: 运行 Phase 2 (Controlled Scaling)
- ✅ `docs/experiment_usage.md`: 详细的使用说明文档

## 数据集下载说明

### VQA v2（必需）

VQA v2 可以通过项目自带的下载脚本自动下载：

```bash
# 请确保已安装依赖
pip install -e ".[experiments]"

# 运行下载脚本 (如果脚本存在于 scripts/ 目录)
# 注意：当前项目结构中，下载脚本可能位于原始位置或需手动下载
# 建议直接使用 /anvil/projects/x-cis250705/data/vlm/molmo 下的共享数据
```

**注意**: 默认数据路径已配置为 `/anvil/projects/x-cis250705/molmo`，通常无需重新下载。

## 快速开始

### 1. 设置环境变量

```bash
# 激活环境
source /anvil/projects/x-cis250705/molmo/activate_env_anvil.sh

# 可选：覆盖默认数据路径
export MOLMO_DATA_DIR=/anvil/projects/x-cis250705/molmo
```

### 2. 运行实验阶段

```bash
# Phase 1: 延迟分布与组件分析 (Exp 1 & 3)
bash scripts/run_phase1.sh

# Phase 2: Scaling 分析 (Exp 2, 4a)
bash scripts/run_phase2.sh
```

### 3. 运行独立实验

```bash
# Experiment 4: Token vs Latency (独立运行)
python experiments/motivate/exp4_token_vs_latency.py \
    --model_path hf:allenai/MolmoE-1B-0924 \
    --output_dir ./results/exp4 \
    --run_both

# Experiment 5: Token 对比分析
python experiments/motivate/exp5_token_comparison.py \
    --phase2_results results/phase2/phase2_scaling.json \
    --phase3_results results/exp4/exp4b_language_tokens.json \
    --output_dir ./results/exp5
```

## 输出结果

### 数据文件
- `results/phase1-5k/phase1_dataset_profiling.json`: Phase 1 原始数据
- `results/phase2/phase2_scaling.json`: Phase 2 Scaling 数据

### 图表文件
- `results/phase1-5k/figures/`: 包含延迟直方图、饼图等
- `results/phase2/figures/`: 包含 FLOPs vs Latency、Vision Scaling 等图表

## 所有实验已实现

✅ **Experiment 1**: 延迟分布和 Tail Latency - 集成在 `run_unified_experiments.py` (Phase 1)
✅ **Experiment 2**: FLOPs vs Latency - 集成在 `run_unified_experiments.py` (Phase 2)
✅ **Experiment 3**: 组件级分析 - 集成在 `run_unified_experiments.py` (Phase 1)
✅ **Experiment 4**: Token vs Latency - `exp4_token_vs_latency.py` (也可通过 Phase 2 运行部分)
✅ **Experiment 5**: Token 对比 - `exp5_token_comparison.py`

### 实验依赖关系

- **Phase 1** 关注真实数据分布
- **Phase 2** 关注受控变量 Scaling
- **Exp 5** 依赖 Phase 2 和 Exp 4B 的结果进行对比分析

## 论文写作建议

基于实验结果，可以组织 Motivational Study 章节如下：

### Study 1: Latency 分布，而不是平均值
- **图表**: Phase 1 的直方图
- **要点**: 
  - P95/P99 远高于平均值
  - 说明 tail latency 是实际问题

### Study 2: 瓶颈在 LLM
- **图表**: Phase 1 的饼图
- **要点**:
  - Vision + Projector 占比很小
  - LLM 占绝大多数，说明需要针对 LLM 做自适应控制

### Study 3: Token 类型的影响不同
- **图表**: Exp 5 的对比图
- **要点**:
  - Vision tokens 主要影响 prefill
  - Language tokens 主要影响 decode
  - 它们的单位成本 (ms/token) 可能不同，需要细粒度控制
