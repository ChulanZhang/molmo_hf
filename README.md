# MolmoE-1B

Molmo is an open-source vision-language model family from AI2. MolmoE-1B is a multimodal MoE model with ~1.5B active parameters (7.2B total), delivering strong performance among models of similar size.

**Learn more**: [Blog](https://molmo.allenai.org/blog) | [Paper](https://huggingface.co/papers/2409.17146) | [Demo](https://molmo.allenai.org/)

---
## 📁 Project layout
```
molmo_hf/
├── molmo/                      # Core Python package
│   ├── models/                 # Architecture and configs
│   ├── preprocessors/          # Data preprocessing
│   └── utils/                  # Utilities
├── configs/                    # Config files
│   ├── model/                  # Model configs
│   └── tokenizer/              # Tokenizer configs
├── checkpoints/                # Model weights
├── experiments/                # Experiment scripts
│   ├── profiling/              # Performance profiling
│   └── motivate/               # Core experiment framework
├── scripts/                    # Helper scripts
├── tests/                      # Tests
├── docs/                       # Documentation
├── setup.py                    # Package setup
└── requirements.txt            # Dependencies
```

---
## 🚀 Quickstart

### Installation
**From source (recommended for dev)**
```bash
git clone <repository-url>
cd molmo_hf
# Base install
pip install -e .
# With experiment tools
pip install -e ".[experiments]"
# With training tools (wandb, etc.)
pip install -e ".[train]"
# Everything
pip install -e ".[all]"
```

**Note**: This is the HF-compatible version of Molmo using PyTorch for MoE (not megablocks), making dynamic MoE topK experiments easier.

## 🧪 Experiments & profiling
The repo includes a full experiment suite for latency/perf analysis.
See `docs/experiment_usage.md` for details.

### Quick run
**1. Motivation study (all experiments)**
```bash
# Run all experiments in sequence
bash experiments/motivate/run_all_experiments.sh [GPU_ID]

# Or run individually
python experiments/motivate/exp1_latency_distribution.py --model_path checkpoints --num_samples 5000
python experiments/motivate/exp2_component_profiling.py --model_path checkpoints --num_samples 1000
python experiments/motivate/exp3_vision_tokens_vs_latency.py --model_path checkpoints
python experiments/motivate/exp4_language_tokens_vs_latency.py --model_path checkpoints
python experiments/motivate/exp5_flops_vs_latency.py --exp3_results ... --exp4_results ...
```

**2. Profiling experiments (knobs)**
```bash
# Knob 1: Context scaling
python experiments/profiling/knob1_tokens/exp_context_scaling.py

# Knob 2: MoE Top-K
python experiments/profiling/knob2_topk/exp_moe_topk.py

# Knob 3: Layer skipping
python experiments/profiling/knob3_layers/exp_layer_skipping.py
```

### Basic usage
```python
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

# Load model and processor locally
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

# Process image + text
inputs = processor.process(
    images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
    text="Describe this image."
)

# Batch and move to device
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

# Generate
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="\n\n")
)
print(processor.decode(output[0]))
```

---

## 📊 Results & evaluation
- Core evaluation and profiling scripts live under `experiments/`.
- Metrics/plots are stored under `experiments/profiling/plots/` and `results/`.

## 🧰 Troubleshooting
- Ensure CUDA/cuDNN match your PyTorch build.
- Some experiments require specific datasets; see `docs/ALL_9_DATASETS_DATA_REQUIREMENTS.md`.
- For precise vision token control and image-size knobs, see `docs/knobs/vision_tokens_precise_control_analysis.md`.

