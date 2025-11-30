# MolmoE-1B-0924 Profiling Experiments

This directory contains the MolmoE-1B-0924 model checkpoint and profiling experiment scripts.

## Directory Structure



## Quick Start

###  1. Activate Environment

```bash
cd /anvil/projects/x-cis250705/molmo_hf
source activate_env.sh
```

或使用便捷脚本：
```bash
ssh h000
cd /anvil/projects/x-cis250705/molmo_hf
. activate_env.sh
```

### 2. Run Experiments

#### Test HF Model Loading (with ipdb)
```bash
python scripts/profiling_experiments/test_hf_model.py
# 会在 ipdb 断点处暂停,可以交互式检查模型结构
```

#### Context Scaling Experiment  
```bash
python scripts/profiling_experiments/exp1_context_scaling.py --num_samples 10
```

#### MoE Top-K Experiment
```bash
python scripts/profiling_experiments/exp2_moe_topk.py --num_samples 10
```

## Important Notes

### Model Loading
- 从当前目录加载: `AutoModelForCausalLM.from_pretrained('.', trust_remote_code=True)`
- 模型结构: `model.model.transformer["blocks"][i].mlp` (HF structure)
- MoE Top-K 修改: 需要同时修改 `model.config.moe_top_k` 和 `block.mlp.top_k`

### Environment Variables
- `PYTHONPATH`: 包含 `/anvil/projects/x-cis250705/molmo_hf`
- `HF_HOME`: `/anvil/projects/x-cis250705/data/vlm/huggingface`
- `CUDA_VISIBLE_DEVICES`: 默认 `0`

### Debugging with ipdb
`test_hf_model.py` 已添加 ipdb 断点:
- `s` - step into
- `n` - next line
- `c` - continue
- `print(model.model.transformer["blocks"][0].mlp)` - 检查结构
- `block.mlp.top_k = 2` - 修改 top_k

## Migration Info
- Source: `/home/x-pwang1/ai_project/molmo`
- Migrated: 2025-11-29
- Files: 13 experiment scripts + base_experiment.py
