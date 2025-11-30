# MolmoE-1B

Molmo是由Allen Institute for AI开发的开源视觉-语言模型家族。MolmoE-1B是基于混合专家(MoE)架构的多模态语言模型，具有1.5B活跃参数和7.2B总参数，在同等规模的多模态模型中实现了业界领先的性能。

**了解更多**: [博客文章](https://molmo.allenai.org/blog) | [论文](https://huggingface.co/papers/2409.17146) | [在线Demo](https://molmo.allenai.org/)

---

## 📁 项目结构

```
molmo_hf/
├── molmo/                      # 主Python包
│   ├── models/                 # 模型架构和配置
│   ├── preprocessors/          # 数据预处理模块
│   └── utils/                  # 工具函数
├── configs/                    # 配置文件
│   ├── model/                  # 模型配置
│   └── tokenizer/              # 分词器配置
├── checkpoints/                # 模型权重文件
├── experiments/                # 实验脚本
│   ├── profiling/              # 性能分析实验
│   └── motivate/               # 基础实验框架
├── scripts/                    # 示例运行脚本
├── tests/                      # 测试文件
├── docs/                       # 文档
├── setup.py                    # 安装配置
└── requirements.txt            # 依赖列表
```

---

## 🚀 快速开始

### 安装

**从源码安装（推荐用于开发）**

```bash
git clone <repository-url>
cd molmo_hf
# pip install -e .
pip install -e ".[experiments]"
```
## 🧪 实验与性能分析

本项目包含完整的实验套件，用于分析模型延迟和性能。

详细文档请参考：[docs/experiment_usage.md](docs/experiment_usage.md)

### 快速开始

**1. Motivation Study (Phase 1 & 2)**
```bash
bash experiments/motivate/run_phase1.sh
bash experiments/motivate/run_phase2.sh
```

**2. Profiling Experiments (Control Knobs)**
```bash
# Knob 1: Context Scaling
python experiments/profiling/knob1_tokens/exp_context_scaling.py

# Knob 2: MoE Top-K
python experiments/profiling/knob2_topk/exp_moe_topk.py

# Knob 3: Layer Skipping
python experiments/profiling/knob3_layers/exp_layer_skipping.py
```

### 基础使用

```python
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

# 从本地加载模型和处理器
model = AutoModelForCausalLM.from_pretrained(
    './molmo_hf',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

processor = AutoProcessor.from_pretrained(
    './molmo_hf',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# 处理图像和文本
inputs = processor.process(
    images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
    text="Describe this image."
)

# 生成输入批次
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

# 生成输出
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="