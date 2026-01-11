# Directory Structure (experiments/profiling)

This describes the layout under `experiments/profiling/`.

## Tree
```
experiments/profiling/
├── docs/                       # Documentation
│   ├── PLOTTING_STYLE_GUIDE.md
│   ├── MULTI_DATASET_EXPERIMENTS.md
│   └── DATASET_SIZE_ESTIMATION.md
├── plots/                      # Plotting scripts
│   ├── plot_accuracy_latency_pareto.py / .sh
│   ├── plot_exp1_context_scaling.py
│   ├── plot_exp1_exp2_exp3_accuracy.py
│   ├── plot_exp2_moe_topk.py
│   ├── plot_exp3_transformer_blocks_mask.py
│   ├── plot_exp4_output_tokens.py
│   └── plot_context_scaling.py
├── knob1_tokens/               # Exp1: vision token control
│   ├── exp1_accuracy.py
│   └── exp_context_scaling.py
├── knob2_topk/                 # Exp2: MoE Top-K
│   ├── exp2_accuracy.py
│   └── exp_moe_topk.py
├── knob3_layers/               # Exp3: transformer layer control
│   ├── exp3_accuracy.py
│   ├── exp3_accuracy_sensitivity.py
│   └── exp_transformer_blocks_mask.py
├── knob4_output_tokens/        # Exp4: output token control
│   └── exp_output_tokens_scaling.py
├── knob5_combined/             # Exp5/6: combined knobs
│   ├── exp5_accuracy.py        # accuracy
│   ├── exp6_accuracy.py        # latency
│   └── SPARSE_SAMPLING_STRATEGIES.md
├── utils/                      # Helper scripts (compare/analyze_*.py, etc.)
└── *.sh                        # Bash entrypoints (run_exp*.sh, plot_*.sh, test_*.sh)
```

## Usage
- Run an experiment:
```bash
./run_exp5_accuracy.sh
./run_exp5_exp6_multi_datasets.sh exp5
```
- Generate plots:
```bash
./plots/plot_accuracy_latency_pareto.sh
./plot_all_experiments.sh
```
- Read docs:
```bash
cat docs/PLOTTING_STYLE_GUIDE.md
cat docs/MULTI_DATASET_EXPERIMENTS.md
```

## Naming
- Experiments: `exp{num}_{type}.py` (e.g., `exp5_accuracy.py`)
- Plots: `plot_{experiment}.py` (e.g., `plot_exp1_context_scaling.py`)
- Runner scripts: `run_{experiment}.sh`
- Docs: `{topic}.md`
