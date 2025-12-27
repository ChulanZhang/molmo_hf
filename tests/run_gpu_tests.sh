#!/bin/bash
# GPU test script on h009
# Uses the 4th GPU (CUDA device 3)

set -e

echo "=========================================="
echo "GPU test script - using GPU 4"
echo "=========================================="

# Select GPU 4
export CUDA_VISIBLE_DEVICES=3

# Activate env
if [ -f "activate_env.sh" ]; then
    source activate_env.sh
fi

# Show GPU info
echo ""
echo "GPU info:"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device Count: {torch.cuda.device_count()}'); [print(f'  Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

echo ""
echo "Running GPU tests..."
echo ""

# Run tests
pytest tests/test_gpu.py -v -s

echo ""
echo "=========================================="
echo "GPU tests finished"
echo "=========================================="




