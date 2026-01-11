# 项目约定

## 文件组织结构

- **实验脚本**：放在 `experiments/` 目录，按功能子目录组织
  - `experiments/controller/` - 控制器相关实验
  - `experiments/core_exp/` - 核心实验
  - `experiments/profiling/` - 性能分析实验
  
- **配置文件**：放在 `configs/` 目录
  - `configs/model/` - 模型配置
  - `configs/tokenizer/` - 分词器配置
  
- **文档**：放在 `docs/` 目录，按主题组织
  - `docs/evaluation/` - 评估相关文档
  - `docs/controller/` - 控制器相关文档
  - `docs/experiments/` - 实验相关文档
  - **文档命名规范**：所有文档文件必须使用小写字母和下划线（`snake_case`）
    - ✅ 正确：`evaluation_guide.md`, `training_guide.md`
    - ❌ 错误：`evaluation_guide.md`, `EvaluationGuide.md`
  
- **检查点**：放在 `checkpoints/` 目录
- **测试文件**：放在 `tests/` 目录，与源码结构对应

## 导入顺序

1. 标准库导入（如 `os`, `sys`, `json`）
2. 第三方库导入（如 `torch`, `transformers`, `numpy`）
3. 本地应用/库导入（如 `from molmo.models import ...`）

每组导入之间用空行分隔。

示例：
```python
import os
import json
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from molmo.models import MolmoModel
from molmo.utils import setup_logger
```

## 依赖管理

- 新依赖必须添加到 `setup.py` 的 `extras_require` 中
- 使用 `pip install -e ".[experiments]"` 安装开发依赖
- 不要提交 `__pycache__/`、`.pyc` 文件或虚拟环境目录
- 大型数据文件不要提交到 Git，使用 `.gitignore` 排除

## 日志规范

- 使用 `logging` 模块而不是 `print`
- 日志级别：`DEBUG` < `INFO` < `WARNING` < `ERROR` < `CRITICAL`
- 在模块级别创建 logger：`logger = logging.getLogger(__name__)`
- 示例：
  ```python
  import logging
  logger = logging.getLogger(__name__)
  
  logger.info("开始训练模型")
  logger.warning("检查点文件不存在，使用默认配置")
  logger.error(f"训练失败: {error}")
  ```

## 配置文件规范

- 配置文件使用 JSON 或 YAML 格式
- 配置文件应该包含版本信息或日期
- 重要配置项应该有注释说明
- 示例：
  ```json
  {
    "_comment": "控制器训练配置 - 2024-01-15",
    "model": {
      "name": "molmo-1b",
      "path": "checkpoints/molmo-1b"
    },
    "training": {
      "batch_size": 32,
      "learning_rate": 1e-4
    }
  }
  ```

## 实验脚本规范

- 实验脚本应该支持命令行参数
- 使用 `argparse` 或 `click` 处理命令行参数
- 实验结果应该保存到 `results/` 目录
- 脚本应该包含使用说明（docstring 或 `--help`）
- 示例：
  ```python
  """
  训练联合控制器模型。
  
  用法:
      python train_joint_controller.py --config configs/controller.json --epochs 10
  """
  import argparse
  
  def main():
      parser = argparse.ArgumentParser(description="训练联合控制器")
      parser.add_argument("--config", type=str, required=True)
      parser.add_argument("--epochs", type=int, default=10)
      args = parser.parse_args()
      ...
  ```

