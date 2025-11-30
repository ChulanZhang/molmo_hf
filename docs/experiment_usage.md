# Molmo Experiments Documentation

本文档列出了所有可用的实验脚本，包括用途、运行方式和输出说明。

---# Molmo Experiments Documentation

## 📂 Directory Structure

```text
experiments/
    ├── measure_flops_scaling.py
    ├── plot_context_scaling.py
    ├── quick_inspect_structure.py
    ├── test_hf_model.py
    └── verify_moe_topk.py
```

---

## 🧪 Motivate实验

### Motivate Experiments (基础框架与复现)

| 文件 | 类型 | 功能 |
|------|------|------|
| `base_experiment.py` | 基类 | 提供通用实验功能：模型加载、延迟测量、统计计算 |
| `run_unified_experiments.py` | 统一入口 | 实现 Phase 1 和 Phase 2 的所有实验逻辑 |
| `exp4_token_vs_latency.py` | 独立脚本 | **Token vs Latency**: 研究视觉Token和语言Token数量对延迟的影响 (包含4A和4B两部分) |
| `exp5_token_comparison.py` | 独立脚本 | **Token Comparison**: 对比视觉Token和语言Token的延迟增长率 (Slope Analysis) |

#### 实验阶段 (Phases)

我们提供了方便的 Shell 脚本来运行完整的实验阶段：

**Phase 1: Dataset Profiling**
- **脚本**: `scripts/run_phase1.sh`
- **功能**: 在真实数据集 (COCO VQA) 上运行 Profiling。
- **包含实验**:
  - **Exp 1**: Latency Distribution (直方图)
  - **Exp 3**: Component Latency Breakdown (饼图)
- **用法**:
  ```bash
  bash scripts/run_phase1.sh [GPU_ID]
  ```

**Phase 2: Controlled Scaling**
- **脚本**: `scripts/run_phase2.sh`
- **功能**: 使用合成数据进行受控 Scaling 测试。
- **包含实验**:
  - **Exp 2**: FLOPs vs Latency
  - **Exp 4a**: Vision Tokens vs Latency
  - **Exp 5**: Token Comparison (需单独运行分析脚本)
- **用法**:
  ```bash
  bash scripts/run_phase2.sh [GPU_ID]
  ```

#### 独立脚本详细说明

**1. exp4_token_vs_latency.py**
- **功能**: 
  - **4A (Vision)**: 通过调整图像分辨率控制视觉Token数量，测量Prefill延迟。
  - **4B (Language)**: 固定图像，通过调整 `max_new_tokens` 控制输出长度，测量Decode延迟。
- **用法**:
  ```bash
  python experiments/motivate/exp4_token_vs_latency.py \\
      --model_path hf:allenai/MolmoE-1B-0924 \\
      --output_dir results/exp4 \\
      --run_both
  ```
- **输出**: 生成 JSON 结果文件和 PNG 图表 (在 `figures/` 子目录下)。

**2. exp5_token_comparison.py**
- **功能**: 读取 Exp4 生成的 JSON 结果，对比视觉和语言Token的单位延迟成本 (ms/token)。
- **用法**:
  ```bash
  python experiments/motivate/exp5_token_comparison.py \\
      --phase2_results results/exp4/exp4a_coco_2014_vqa_validation.json \\
      --phase3_results results/exp4/exp4b_coco_2014_vqa_validation.json \\
      --output_dir results/exp5
  ```
- **输出**: 生成对比分析图表 `exp5_vision_scaling.png` 和 `exp5_language_scaling.png`。

**数据路径**: 
- 默认数据目录：`/anvil/projects/x-cis250705/molmo`
- 可通过环境变量 `MOLMO_DATA_DIR` 覆盖

**主要功能**:
- `_load_model()`: 加载Molmo模型
- `build_dataloader()`: 构建数据加载器
- `measure_inference_latency()`: 测量推理延迟
- `count_flops()`: 估算FLOPs
- `compute_statistics()`: 计算统计指标（P50/P95/P99/mean/std）
- `save_results()`: 保存JSON结果

---

## 🔬 Profiling实验

### 1. exp1_context_scaling.py
**类型**: 核心实验

**功能**: 研究输入文本长度对Prefill延迟的影响

**运行方式**:
```bash
python experiments/profiling/exp1_context_scaling.py \\
    --model_path hf:allenai/MolmoE-1B-0924 \\
    --output_dir ./results/context_scaling \\
    --num_samples 50 \\
    --max_length 1500 \\
    --step_size 100
```

**参数说明**:
- `--model_path`: 模型路径或HuggingFace模型ID
- `--output_dir`: 输出目录（默认：`./results/context_scaling`）
- `--num_samples`: 每个长度采样次数（默认：50）
- `--max_length`: 最大文本长度（默认：1500 tokens）
- `--step_size`: 长度步长（默认：100 tokens）

**输出**:
- `exp1_context_scaling_results.json`: 包含各长度下的延迟统计（P50/P95/P99等）

**实验设计**:
- 固定336x336图像（最小化视觉影响）
- 变化文本长度：50, 150, 250, ..., 1500 tokens
- 测量 `T_LLM_prefill` 延迟

---

### 2. exp2_moe_topk.py
**类型**: 核心实验

**功能**: 研究MoE Top-K参数对Prefill和Decode延迟的影响

**运行方式**:
```bash
python experiments/profiling/exp2_moe_topk.py \\
    --model_path hf:allenai/MolmoE-1B-0924 \\
    --output_dir ./results/moe_topk \\
    --num_samples 50
```

**参数说明**:
- `--model_path`: 模型路径
- `--output_dir`: 输出目录（默认：`./results/moe_topk`）
- `--num_samples`: 每个top_k值的采样次数（默认：50）

**输出**:
- `exp2_moe_topk_results.json`: 包含各top_k值下的Prefill和Decode延迟统计

**实验设计**:
- 测试 top_k = [1, 2, 4, 8]
- 固定输入："Describe this image." + 336x336图像
- 分别测量 `T_LLM_prefill` 和 `T_LLM_decode`

**技术细节**:
- 通过修改 `block.ffn.args.top_k` 动态调整MoE参数
- 支持对所有MoE块批量修改

---

### 3. measure_flops_scaling.py
**类型**: 分析工具

**功能**: 测量不同Top-K值下的FLOPs和延迟关系

**运行方式**:
```bash
python experiments/profiling/measure_flops_scaling.py
```

**输出**:
- `results/moe_topk/flops_scaling_analysis.json`
- 控制台输出 top_k=1 vs top_k=8 的延迟比较

**实验设计**:
- 比较极端情况：top_k=1（最小）vs top_k=8（全专家）
- 计算理论FLOPs比例 vs 实际延迟比例
- 判断计算是否为瓶颈

---

### 4. plot_context_scaling.py
**类型**: 可视化工具

**功能**: 绘制Context Scaling实验结果图表

**运行方式**:
```bash
python experiments/profiling/plot_context_scaling.py \\
    --input results/context_scaling/exp1_context_scaling_results.json \\
    --output results/context_scaling/context_scaling_plot.png
```

**输出**:
- PNG图像文件，显示文本长度 vs Prefill延迟曲线

---

### 5. analyze_tokens.py
**类型**: 调试工具

**功能**: 分析输入tokenization细节

**运行方式**:
```bash
python experiments/profiling/analyze_tokens.py
```

**输出**:
- 控制台输出：
  - 输入token shape
  - 视觉tokens数量
  - 每个token的ID和解码结果
  - Token频率统计

**用途**:
- 调试padding行为
- 验证视觉token数量
- 理解特殊token使用

---

### 6. inspect_moe_layer.py
**类型**: 调试工具

**功能**: 检查MoE层结构和配置

**运行方式**:
```bash
python experiments/profiling/inspect_moe_layer.py
```

**输出**:
- 控制台输出 MoE 层的详细信息：
  - 层类型和位置
  - Top-K配置
  - 专家数量
  - 参数统计

---

### 7. inspect_molmo_flow.py
**类型**: 调试工具

**功能**: 追踪Molmo模型的前向传播流程

**运行方式**:
```bash
python experiments/profiling/inspect_molmo_flow.py
```

**输出**:
- 每层的输入输出shape
- 中间激活的形状变化
- 内存占用估算

---

### 8. inspect_pooling_params.py
**类型**: 调试工具

**功能**: 检查视觉pooling参数

**运行方式**:
```bash
python experiments/profiling/inspect_pooling_params.py
```

**输出**:
- Pooling层配置
- 池化比例和窗口大小

---

### 9. quick_inspect_structure.py
**类型**: 调试工具

**功能**: 快速查看模型整体结构

**运行方式**:
```bash
python experiments/profiling/quick_inspect_structure.py
```

**输出**:
- 模型层次结构
- 每层类型和参数量
- 总参数统计

---

### 10. test_hf_model.py
**类型**: 验证工具

**功能**: 测试HuggingFace模型加载和基本推理

**运行方式**:
```bash
python experiments/profiling/test_hf_model.py
```

**输出**:
- 模型加载成功/失败状态
- 简单推理结果
- 设备信息

---

### 11. verify_moe_topk.py
**类型**: 验证工具

**功能**: 验证MoE Top-K修改是否生效

**运行方式**:
```bash
python experiments/profiling/verify_moe_topk.py
```

**输出**:
- 修改前后的 top_k 值对比
- 实际运行时专家选择情况
- 验证结果（成功/失败）

---

### 12. check_config_direct.py
**类型**: 调试工具

**功能**: 直接读取模型配置文件

**运行方式**:
```bash
python experiments/profiling/check_config_direct.py
```

**输出**:
- `config.json` 的完整内容
- MoE相关配置项
- 关键超参数

---

## 📊 数据保存路径说明

所有实验默认将结果保存到相对路径 `./results/` 下：

```
results/
├── context_scaling/
│   ├── exp1_context_scaling_results.json
│   └── context_scaling_plot.png
└── moe_topk/
    ├── exp2_moe_topk_results.json
    └── flops_scaling_analysis.json
```

**数据目录配置**:
- 模型数据默认路径：`/anvil/projects/x-cis250705/molmo`
- 可通过环境变量 `MOLMO_DATA_DIR` 自定义
- HuggingFace缓存：默认 `~/.cache/huggingface`，可通过 `HF_HOME` 自定义

---

## 🚀 快速开始

### 1. 运行核心实验

```bash
# Context Scaling实验
python experiments/profiling/exp1_context_scaling.py \\
    --output_dir /anvil/projects/x-cis250705/molmo/results/context_scaling

# MoE Top-K实验
python experiments/profiling/exp2_moe_topk.py \\
    --output_dir /anvil/projects/x-cis250705/molmo/results/moe_topk

# 绘制结果
python experiments/profiling/plot_context_scaling.py \\
    --input /anvil/projects/x-cis250705/molmo/results/context_scaling/exp1_context_scaling_results.json \\
    --output /anvil/projects/x-cis250705/molmo/results/context_scaling/plot.png
```

### 2. 调试和验证

```bash
# 检查模型结构
python experiments/profiling/quick_inspect_structure.py

# 检查MoE配置
python experiments/profiling/inspect_moe_layer.py

# 验证Top-K修改
python experiments/profiling/verify_moe_topk.py

# 分析tokenization
python experiments/profiling/analyze_tokens.py
```

---

## 📝 注意事项

1. **Import路径**: 实验脚本目前使用 `sys.path.append(os.getcwd())` 来导入 `base_experiment`。推荐安装项目后使用：
   ```bash
   pip install -e .
   # 然后修改import为：
   # from molmo.experiments.motivate import BaseExperiment
   ```

2. **GPU内存**: 某些实验（如长文本Context Scaling）可能需要大量GPU内存。建议：
   - 使用单GPU
   - 设置合适的 `--max_length` 和 `--num_samples`
   - 监控GPU内存使用：`nvidia-smi`

3. **数据路径**: 确保 `/anvil/projects/x-cis250705/molmo` 目录存在且有写权限

4. **缓存配置**: 首次运行会下载模型，建议预先设置：
   ```bash
   export HF_HOME=~/.cache/huggingface
   export MOLMO_DATA_DIR=/anvil/projects/x-cis250705/molmo
   ```

---

## 🔧 扩展实验

基于 `BaseExperiment` 类，您可以轻松创建新实验：

```python
from experiments.motivate.base_experiment import BaseExperiment

class MyExperiment(BaseExperiment):
    def run(self, **kwargs):
        # 1. 准备数据
        dataloader = self.build_dataloader("dataset_name")
        
        # 2. 运行测量
        for batch in dataloader:
            metrics = self.measure_inference_latency(
                batch, 
                max_new_tokens=50,
                measure_components=True
            )
            # 处理metrics...
        
        # 3. 保存结果
        self.save_results(results, "my_experiment_results.json")
```

---

**最后更新**: 2025-11-29
