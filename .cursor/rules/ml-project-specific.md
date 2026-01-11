# 机器学习项目特定规则

## 模型和训练

1. **模型加载**
   - 使用 `torch_dtype='auto'` 或 `torch_dtype=torch.float16` 以节省内存
   - 使用 `device_map='auto'` 进行自动设备分配
   - 对于大模型，考虑使用 `low_cpu_mem_usage=True`

2. **检查点管理**
   - 检查点文件保存在 `checkpoints/` 目录
   - 检查点文件名应该包含模型名称、日期或版本信息
   - 保存检查点时同时保存配置文件和训练参数

3. **训练循环**
   - 使用 `wandb` 或其他工具记录训练指标
   - 定期保存检查点（如每个 epoch 或每 N 个步骤）
   - 实现早停机制防止过拟合

## 数据处理

1. **数据加载**
   - 使用 `DataLoader` 进行批处理
   - 设置适当的 `num_workers` 和 `pin_memory`
   - 对于大型数据集，考虑使用流式加载

2. **数据预处理**
   - 预处理逻辑放在 `molmo/preprocessors/` 目录
   - 预处理应该可复现（设置随机种子）
   - 保存预处理后的数据时使用标准格式（如 JSON、Parquet）

## 实验管理

1. **实验脚本**
   - 实验脚本应该支持命令行参数
   - 使用配置文件管理实验参数
   - 实验结果应该保存到 `results/` 目录，包含时间戳和实验名称
   - **配置集中管理**：所有配置参数集中在 `main()` 函数顶部，使用清晰的分隔线标记
   - **环境变量覆盖**：支持通过环境变量覆盖默认配置（如 `NUM_GPUS_OVERRIDE`）
   - **主脚本和子脚本分离**：主脚本负责编排多个实验，子脚本负责具体实验逻辑

2. **可复现性**
   - 设置随机种子（`torch.manual_seed`, `np.random.seed`, `random.seed`）
   - 记录实验配置和超参数
   - 保存模型版本和依赖版本信息
   - **可复现的数据集采样**：使用固定的随机种子进行数据集采样
   - **记录采样信息**：记录采样策略和样本数量

3. **评估**
   - 评估脚本应该支持多个数据集
   - 评估结果应该保存为 JSON 或 CSV 格式
   - 包含评估指标的计算方法和置信区间（如适用）
   - **失败后继续执行**：单个数据集失败后继续执行下一个数据集，最后统一报告失败项
   - **文件存在性检查**：在使用文件前检查是否存在，提供回退机制

## Transformers 库使用

1. **模型加载**
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       trust_remote_code=True,
       torch_dtype='auto',
       device_map='auto'
   )
   ```

2. **处理器使用**
   ```python
   processor = AutoProcessor.from_pretrained(
       model_path,
       trust_remote_code=True
   )
   ```

3. **生成配置**
   - 使用 `GenerationConfig` 管理生成参数
   - 设置适当的 `max_new_tokens` 和 `stop_strings`
   - 考虑使用 `do_sample`, `temperature`, `top_p` 等参数

## 性能优化

1. **批处理**
   - 尽可能使用批处理提高效率
   - 注意批处理大小对内存的影响
   - 使用 `torch.compile()` 加速（PyTorch 2.0+）

2. **混合精度训练**
   - 使用 `torch.cuda.amp` 进行混合精度训练
   - 注意数值稳定性问题

3. **梯度累积**
   - 对于大模型，使用梯度累积模拟更大的批处理大小
   - 在累积步骤之间不更新参数

## 调试和日志

1. **训练日志**
   - 记录损失、学习率、梯度范数等关键指标
   - 使用 `tqdm` 显示训练进度
   - 定期打印样本输出用于调试
   - **彩色日志输出**：使用 `colorlog` 库提供彩色日志，提高可读性
   - **优雅降级**：如果 `colorlog` 不可用，自动降级到标准 logging
   - **分布式日志控制**：只在 rank 0 输出日志，避免多 GPU 环境下的日志重复
   - **日志文件管理**：使用带时间戳的日志文件名，在输出目录下创建 `logs/` 子目录

2. **错误处理**
   - 捕获 `CUDA out of memory` 错误并提供建议
   - 记录训练过程中的异常情况
   - 实现检查点恢复机制
   - **有意义的错误消息**：包含上下文信息（如数据集名称、返回码）
   - **记录日志文件路径**：帮助用户定位问题
   - **异常处理**：使用 `try-except` 捕获特定异常，提供清晰的错误信息

## 评估和指标

1. **评估脚本**
   - 评估脚本应该独立于训练脚本
   - 支持批量评估和单个样本评估
   - 评估结果应该包含详细的指标分解

2. **指标计算**
   - 使用标准库（如 `sklearn.metrics`）计算指标
   - 对于自定义指标，提供清晰的实现和文档
   - 记录评估数据集的信息（大小、分布等）

## 分布式训练

1. **GPU 检测和配置**
   - **自动检测 GPU 数量**：使用 `nvidia-smi` 或 `CUDA_VISIBLE_DEVICES` 自动检测
   - **提供回退机制**：如果检测失败，使用默认值或环境变量
   - **使用 torchrun**：使用 `torchrun` 进行分布式训练，而不是直接使用 `python -m torch.distributed.launch`

2. **子进程管理**
   - **环境变量设置**：设置 `PYTHONUNBUFFERED='1'` 强制无缓冲输出
   - **抑制警告**：设置 `TORCH_DISTRIBUTED_DEBUG='OFF'` 减少噪音
   - **正确处理 stdout/stderr**：保持 stderr 直接输出（让 tqdm 正常工作），只捕获 stdout 用于日志
   - **过滤噪音**：过滤掉 torchrun 的警告信息（如 OMP_NUM_THREADS）

3. **分布式数据加载**
   - 使用 `DistributedSampler` 确保每个进程处理不同的数据
   - 在分布式环境下正确设置 `num_workers` 和 `pin_memory`

