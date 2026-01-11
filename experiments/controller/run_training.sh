#!/bin/bash
# Training script for Joint Controller (Stage1 + Stage2)
# 
# NOTE: This script is deprecated. Please use run_training.py instead:
#   python experiments/controller/run_training.py
#
# The Python version has better argument handling and wandb enabled by default.

# Redirect to Python script
python experiments/controller/run_training.py "$@"

