#!/bin/bash
# 在 h009 上运行 GPU 测试脚本
# 使用第 4 张 GPU（CUDA device 3）

set -e

echo "=========================================="
echo "GPU 测试脚本 - 使用第 4 张 GPU"
echo "=========================================="

# 设置使用第 4 张 GPU
export CUDA_VISIBLE_DEVICES=3

# 激活环境
if [ -f "activate_env.sh" ]; then
    source activate_env.sh
fi

# 显示 GPU 信息
echo ""
echo "GPU 信息："
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device Count: {torch.cuda.device_count()}'); [print(f'  Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

echo ""
echo "运行 GPU 测试..."
echo ""

# 运行测试
pytest tests/test_gpu.py -v -s

echo ""
echo "=========================================="
echo "GPU 测试完成"
echo "=========================================="



