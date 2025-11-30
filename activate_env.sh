#!/bin/bash

# Load required modules
module --force purge
module load modtree/gpu
module load cuda/12.0.1
module load cudnn/cuda-12.0_8.8
module load conda/2025.09
module use /anvil/projects/x-cis250705/modules
module load conda-env/molmo-hf-py3.12.11

# Ensure conda env name shows up in prompt (PS1 uses CONDA_PROMPT_MODIFIER)
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    export CONDA_PROMPT_MODIFIER="(${CONDA_DEFAULT_ENV}) "
fi

# Set environment variables
# MOLMO_DATA_DIR: Directory for storing MolMO data
# HF_HOME: Directory for storing Hugging Face data
export MOLMO_DATA_DIR=/anvil/projects/x-cis250705/data/vlm/molmo
export HF_HOME=/anvil/projects/x-cis250705/data/vlm/huggingface

# Re-apply colorful prompt since module loads may reset PS1
if [[ $- == *i* ]]; then
    export PS1="${CONDA_PROMPT_MODIFIER}\[\e[1;33m\]\u\[\e[0m\]@\[\e[1;36m\]\h\[\e[0m\]:\[\e[1;32m\]\w\[\e[0m\]\$ "
fi