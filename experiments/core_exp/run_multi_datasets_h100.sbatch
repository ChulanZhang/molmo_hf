#!/bin/bash
#SBATCH --job-name=multi_datasets_h100
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --account=cis250705-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=0  # Use all available memory on the node

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Create logs directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/logs"

# Change to project root directory
cd "${PROJECT_ROOT}"

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Project Root: ${PROJECT_ROOT}"
echo "Script Directory: ${SCRIPT_DIR}"
echo "=========================================="
echo ""

# Load required modules
echo "Loading modules..."
module --force purge
module load modtree/gpu
module load cuda/12.0.1
module load cudnn/cuda-12.0_8.8
module load conda/2025.09
module use /anvil/projects/x-cis250705/modules
module load conda-env/molmo-hf-py3.12.11

# Set environment variables
# !IMPORTANT: MOLMO_DATA_DIR: Directory for storing MolMO data
# !IMPORTANT: HF_HOME: Directory for storing Hugging Face data
export MOLMO_DATA_DIR=/anvil/projects/x-cis250705/data/vlm/molmo
export HF_HOME=/anvil/projects/x-cis250705/data/vlm/huggingface
# Explicitly set HF_DATASETS_CACHE to ensure datasets library uses the correct cache directory
export HF_DATASETS_CACHE=${HF_HOME}/datasets

# Set CUDA visible devices based on SLURM GPU allocation
if [ -n "$SLURM_STEP_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES=$SLURM_STEP_GPUS
elif [ -n "$SLURM_JOB_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
fi

# Print GPU information
echo "=========================================="
echo "GPU Information"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
fi
echo "=========================================="
echo ""


# Set number of GPUs explicitly (should be 4 based on --gres=gpu:4)
export NUM_GPUS_OVERRIDE=4

# Run the experiment script
echo "=========================================="
echo "Starting Multi-Dataset Experiment"
echo "=========================================="
echo ""

# Allow passing optional arguments to the script (e.g., specific dataset name)
# Usage: sbatch run_multi_datasets_h100.sbatch [dataset_name]
"${SCRIPT_DIR}/run_multi_datasets_h100.sh" "$@"

# Print completion information
echo ""
echo "=========================================="
echo "Job Completed"
echo "=========================================="
echo "End Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

