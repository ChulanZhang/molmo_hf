# Exp 1: regenerate plots
python experiments/motivate/plot_exp1.py \
    --json_file results/motivation/exp1/exp1_latency_distribution.json \
    --dataset coco_2014_vqa \
    --split validation

# Exp 2: regenerate plots
python experiments/motivate/plot_exp2.py \
    --json_file results/motivation/exp2/exp2_component_profiling.json

# Exp 3: regenerate plots
python experiments/motivate/plot_exp3.py \
    --json_file results/motivation/exp3/exp3_vision_tokens_vs_latency.json

# Exp 4: regenerate plots
python experiments/motivate/plot_exp4.py \
    --json_file results/motivation/exp4/exp4_language_tokens_vs_latency.json
