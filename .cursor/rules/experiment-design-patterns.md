# 实验代码设计模式

本文档总结了项目中实验代码的最佳实践和设计模式，基于 `run_multi_datasets_h100.py` 和 `acc_lat_profiling.py` 等高质量实验代码。

## 1. 日志和输出管理

### 1.1 彩色日志输出

- **使用 colorlog 库**：提供彩色日志输出，提高可读性
- **优雅降级**：如果 colorlog 不可用，自动降级到标准 logging
- **统一的颜色类**：定义 `Colors` 类统一管理 ANSI 颜色代码

```python
class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    BRIGHT_CYAN = '\033[1;36m'
    BRIGHT_MAGENTA = '\033[1;35m'
    BRIGHT_YELLOW = '\033[1;33m'
    BRIGHT_GREEN = '\033[1;32m'
    RED = '\033[0;31m'
    # ... 更多颜色

def setup_logging():
    """Setup colored logging with fallback"""
    try:
        import colorlog
        handler = colorlog.StreamHandler(sys.stdout)
        formatter = colorlog.ColoredFormatter(...)
        # ... 配置
        return logging.getLogger(__name__)
    except ImportError:
        # Fallback to basic logging
        logging.basicConfig(...)
        return logging.getLogger(__name__)
```

### 1.2 分布式环境下的日志控制

- **只在 rank 0 输出**：避免多 GPU 环境下的日志重复
- **使用 `self.rank` 或 `get_global_rank()` 判断**：控制日志输出

```python
self.rank = int(os.environ.get("RANK", 0))
if self.rank == 0:
    log.info("Only log on rank 0")
```

### 1.3 日志文件管理

- **带时间戳的日志文件**：使用 `datetime.now().strftime('%Y%m%d_%H%M%S')` 创建唯一日志文件名
- **日志目录组织**：在输出目录下创建 `logs/` 子目录
- **同时输出到终端和文件**：使用流式传输同时写入终端和日志文件

```python
log_dir = Path(base_output_dir) / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"experiment_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
```

## 2. 错误处理和容错机制

### 2.1 失败后继续执行

- **不中断整个流程**：单个数据集失败后继续执行下一个数据集
- **收集失败信息**：维护失败数据集列表，最后统一报告

```python
failed_datasets = []
for dataset_name, split, max_new_tokens in datasets:
    success = run_experiment(...)
    if not success:
        failed_datasets.append(dataset_name)
        log.warning("Continuing to next dataset...")

# 最后统一报告
if failed_datasets:
    log.warning(f"Failed datasets: {', '.join(failed_datasets)}")
```

### 2.2 文件存在性检查

- **检查文件是否存在**：在使用前检查文件是否存在
- **提供回退机制**：如果首选文件不存在，使用备选文件

```python
importance_scores_file = "./results/layer_importance_scores_exp3_recommended.json"
if not Path(importance_scores_file).exists():
    # Fallback to alternative
    importance_scores_file = "./results/layer_importance_scores_multi_dataset_simple.json"
    if not Path(importance_scores_file).exists():
        # Another fallback
        importance_scores_file = "./results/layer_importance_scores.json"
```

### 2.3 异常处理和错误消息

- **捕获具体异常**：使用 `try-except` 捕获特定异常
- **提供有意义的错误消息**：包含上下文信息（如数据集名称、返回码）
- **记录日志文件路径**：帮助用户定位问题

```python
try:
    process = subprocess.Popen(...)
    process.wait()
    if process.returncode == 0:
        return True
    else:
        log.error(f"Command failed for {dataset_name} (return code {process.returncode})")
        log.error(f"Check log file: {log_file}")
        return False
except Exception as e:
    log.error(f"Unexpected error running {dataset_name}: {e}")
    log.error(f"Check log file: {log_file}")
    return False
```

## 3. 配置管理

### 3.1 配置集中管理

- **配置集中在 main() 函数顶部**：所有配置参数集中在函数开头，易于修改
- **使用清晰的注释分隔**：使用分隔线标记配置区域

```python
def main():
    log = setup_logging()
    
    # ============================================================================
    # Experiment Configuration
    # ============================================================================
    # Modify these values directly to change experiment settings
    
    model_path = "checkpoints"
    base_output_dir = "./results/core_exp_h100"
    num_samples = 2000
    # ... 更多配置
```

### 3.2 环境变量覆盖

- **支持环境变量覆盖**：允许通过环境变量覆盖默认配置
- **提供合理的默认值**：即使没有环境变量也能正常工作

```python
# Auto-detect number of GPUs (can be overridden with NUM_GPUS_OVERRIDE env var)
num_gpus = detect_num_gpus()
num_gpus = int(os.environ.get("NUM_GPUS_OVERRIDE", num_gpus))
```

### 3.3 配置注释和文档

- **详细的配置注释**：解释每个配置项的含义和推荐值
- **记录配置来源**：说明配置的参考来源（如实验推荐值）

```python
# Importance scores file (for block selection)
# Using EXP3 recommended scores to ensure correct block removal order:
# - num_active_blocks = 15 → Remove Block 4 (3.09% drop)
# - num_active_blocks = 14 → Remove Block 4, 13 (5.41% drop)
importance_scores_file = "./results/layer_importance_scores_exp3_recommended.json"
```

## 4. 分布式训练支持

### 4.1 自动检测 GPU 数量

- **自动检测可用 GPU**：使用 `nvidia-smi` 或 `CUDA_VISIBLE_DEVICES` 检测
- **提供回退机制**：如果检测失败，使用默认值或环境变量

```python
def detect_num_gpus() -> int:
    """Auto-detect number of GPUs"""
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, check=True)
        return len(result.stdout.strip().split('\n'))
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: check CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            return len(cuda_visible.split(','))
        return 1
```

### 4.2 子进程环境变量管理

- **设置必要的环境变量**：`PYTHONUNBUFFERED`, `TORCH_DISTRIBUTED_DEBUG` 等
- **抑制不必要的警告**：设置 `TORCH_DISTRIBUTED_DEBUG='OFF'` 减少噪音

```python
env = dict(os.environ)
env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
env['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'  # Suppress torchrun warnings
```

### 4.3 正确处理 stdout/stderr 分离

- **保持 stderr 直接输出**：让 tqdm 等工具直接写入 stderr，保持 TTY 检测
- **捕获 stdout 用于日志**：只捕获 stdout 用于日志记录
- **过滤噪音**：过滤掉 torchrun 的警告信息

```python
# IMPORTANT: Don't redirect stderr to stdout - let tqdm write directly to stderr
# This allows tqdm to detect TTY and use single-line mode
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,  # Capture stdout for logging
    stderr=None,  # Let stderr go directly to terminal (for tqdm)
    text=True,
    bufsize=1,
    env=env
)

# Stream stdout to both terminal and log file, filtering out warnings
for line in process.stdout:
    if 'OMP_NUM_THREADS' in line:  # Filter warnings
        f.write(line)  # Still log, but don't print
        continue
    sys.stdout.write(line)  # Write to terminal
    f.write(line)  # Write to log file
```

## 5. 命令行参数支持

### 5.1 可选参数支持

- **支持选择性执行**：允许通过命令行参数选择特定数据集或配置
- **保持向后兼容**：如果没有提供参数，使用默认行为

```python
# Parse command line arguments (optional: run only specific dataset)
specific_dataset = sys.argv[1] if len(sys.argv) > 1 else None

for dataset_name, split, max_new_tokens in datasets:
    if specific_dataset and dataset_name != specific_dataset:
        continue
    # ... 执行实验
```

### 5.2 使用 argparse 处理复杂参数

- **使用 argparse**：对于复杂的实验脚本，使用 `argparse` 处理参数
- **提供默认值和帮助信息**：每个参数都有清晰的说明

```python
parser = argparse.ArgumentParser(description="Combined Profiling Experiment")
parser.add_argument("--model_path", type=str, default="checkpoints", 
                   help="Path to model checkpoint")
parser.add_argument("--num_samples", type=int, default=1000,
                   help="Number of samples to use (None = all)")
```

## 6. 代码组织和可读性

### 6.1 函数职责清晰

- **单一职责原则**：每个函数只做一件事
- **函数命名清晰**：函数名清楚表达其功能
- **合理的函数长度**：避免过长的函数，必要时拆分成多个函数

```python
def setup_logging():
    """Setup colored logging"""
    ...

def detect_num_gpus() -> int:
    """Auto-detect number of GPUs"""
    ...

def run_combined_profiling(...) -> bool:
    """Run combined profiling for a single dataset"""
    ...
```

### 6.2 类型注解

- **使用类型注解**：所有函数参数和返回值都应该有类型注解
- **使用 `Optional` 和 `List`**：明确可选参数和列表类型

```python
def run_combined_profiling(
    dataset_name: str,
    split: str,
    max_new_tokens: int,
    model_path: str,
    base_output_dir: str,
    num_gpus: int,
    tier_list: List[str],
    top_k_list: List[int],
    num_active_blocks_list: List[int],
    importance_scores_file: Optional[str] = None,
    log: logging.Logger = None,
) -> bool:
    ...
```

### 6.3 文档字符串

- **详细的 docstring**：使用 Google 风格的 docstring
- **说明参数和返回值**：清楚说明每个参数的含义和返回值的含义
- **提供使用示例**：对于复杂函数，提供使用示例

```python
def run_combined_profiling(...) -> bool:
    """Run combined profiling for a single dataset
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split (validation, test, etc.)
        ...
    
    Returns:
        bool: True if successful, False if failed
    """
    ...
```

## 7. 可复现性

### 7.1 随机种子设置

- **设置所有随机种子**：`random`, `numpy`, `torch` 的随机种子
- **在初始化时设置**：在类或实验初始化时设置随机种子
- **记录种子值**：在日志或配置中记录使用的种子值

```python
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 7.2 数据集采样策略

- **可复现的采样**：使用固定的随机种子进行数据集采样
- **记录采样信息**：记录采样策略和样本数量

```python
def _create_sampled_dataset(self, dataset, num_samples: Optional[int], seed: int = 66):
    """Create sampled dataset with fixed seed for reproducibility"""
    if num_samples is None:
        return dataset
    indices = np.random.RandomState(seed).choice(len(dataset), num_samples, replace=False)
    return Subset(dataset, indices)
```

## 8. 用户体验

### 8.1 清晰的进度信息

- **简洁的头部信息**：使用分隔线和颜色突出显示关键信息
- **配置摘要**：在开始时显示实验配置摘要
- **完成消息**：实验结束时显示清晰的完成消息和结果摘要

```python
log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
log.info(f"{Colors.BRIGHT_CYAN}Multi-Dataset Combined Profiling{Colors.RESET}")
log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
log.info(f"{Colors.BRIGHT_YELLOW}Config:{Colors.RESET} {num_samples} samples | "
         f"{Colors.CYAN}Tiers:{Colors.RESET} {tier_list} | ...")
```

### 8.2 状态指示

- **使用颜色和符号**：使用颜色和符号（如 ✓）表示状态
- **区分成功和失败**：使用不同颜色区分成功和失败的消息

```python
if importance_scores_file and Path(importance_scores_file).exists():
    log.info(f"Importance Scores: {importance_scores_file} {Colors.GREEN}✓{Colors.RESET}")
else:
    log.warning(f"Importance scores file not found")
```

## 9. 实验脚本组织

### 9.1 主脚本和子脚本分离

- **主脚本负责编排**：主脚本（如 `run_multi_datasets_h100.py`）负责调用子脚本
- **子脚本负责具体实验**：子脚本（如 `acc_lat_profiling.py`）负责具体的实验逻辑
- **清晰的接口**：通过命令行参数传递配置

### 9.2 继承 BaseExperiment

- **使用基类**：实验类继承 `BaseExperiment` 获得通用功能
- **实现抽象方法**：实现 `run()` 方法定义具体实验逻辑
- **复用通用功能**：模型加载、数据加载等功能由基类提供

```python
class CombinedProfilingExperiment(BaseExperiment):
    """Combined Profiling Experiment"""
    
    def __init__(self, ...):
        super().__init__(...)
        # 实验特定的初始化
    
    def run(self, ...):
        """Run the experiment"""
        # 实验逻辑
```

## 10. 最佳实践总结

1. **日志管理**：使用彩色日志，支持分布式环境，优雅降级
2. **错误处理**：失败后继续执行，收集错误信息，提供有意义的错误消息
3. **配置管理**：配置集中管理，支持环境变量覆盖，详细注释
4. **分布式支持**：自动检测 GPU，正确处理子进程，管理环境变量
5. **可读性**：清晰的函数职责，类型注解，详细的文档字符串
6. **可复现性**：设置随机种子，可复现的采样策略
7. **用户体验**：清晰的进度信息，状态指示，简洁的输出

## 11. 代码示例模板

```python
#!/usr/bin/env python3
"""
Script description with key features.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    BRIGHT_CYAN = '\033[1;36m'
    BRIGHT_GREEN = '\033[1;32m'
    RED = '\033[0;31m'
    # ... 更多颜色

def setup_logging():
    """Setup colored logging with fallback"""
    try:
        import colorlog
        # ... 配置 colorlog
        return logging.getLogger(__name__)
    except ImportError:
        logging.basicConfig(...)
        return logging.getLogger(__name__)

def detect_num_gpus() -> int:
    """Auto-detect number of GPUs"""
    # ... 实现

def run_experiment(...) -> bool:
    """Run experiment for a single dataset
    
    Returns:
        bool: True if successful, False if failed
    """
    # ... 实现

def main():
    """Main execution"""
    log = setup_logging()
    
    # ============================================================================
    # Experiment Configuration
    # ============================================================================
    # 配置参数...
    
    # 打印配置摘要
    log.info(f"{Colors.BRIGHT_CYAN}{'='*60}{Colors.RESET}")
    # ... 配置信息
    
    # 执行实验
    failed_items = []
    for item in items:
        success = run_experiment(...)
        if not success:
            failed_items.append(item)
            log.warning("Continuing to next item...")
    
    # 完成消息
    if failed_items:
        log.warning(f"Failed items: {', '.join(failed_items)}")
    else:
        log.info(f"{Colors.BRIGHT_GREEN}All experiments completed successfully!{Colors.RESET}")

if __name__ == "__main__":
    main()
```

