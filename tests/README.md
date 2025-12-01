# 测试套件说明

本目录包含 `molmo_hf` 代码库的完整测试套件。

## 测试结构

### 1. `test_imports.py` - 导入测试
测试所有核心模块的导入功能，确保：
- 所有主要模块可以正常导入
- 没有循环导入问题
- 所有依赖都正确安装

### 2. `test_models.py` - 模型测试
测试模型相关的功能：
- 模型类的导入和初始化
- 模型配置的创建
- 训练方法的可用性
- 参数计数功能

### 3. `test_config.py` - 配置系统测试
测试配置系统：
- 配置类的导入和创建
- 配置转换功能（ModelConfig <-> MolmoConfig）
- 配置桥接函数

### 4. `test_data.py` - 数据集模块测试
测试数据集相关功能：
- 数据集类的导入
- 数据加载器构建函数
- 数据集获取功能
- HF 数据集构建器

### 5. `test_train.py` - 训练模块测试
测试训练相关功能：
- Trainer 类的导入和结构
- 优化器和调度器
- 检查点管理

### 6. `test_eval.py` - 评估模块测试
测试评估相关功能：
- 评估器基类
- 推理评估器
- 损失评估器
- 各种评估工具

### 7. `test_utils.py` - 工具模块测试
测试工具函数：
- PyTorch 工具函数
- 分词器工具
- 通用工具函数
- 异常类
- Safetensors 工具

### 8. `test_integration.py` - 集成测试
测试多个模块的协同工作：
- 配置到模型的流程
- 数据到训练的流程
- 模型训练方法的完整性

### 9. `conftest.py` - Pytest 配置
包含共享的 fixtures 和测试配置。

## 运行测试

### 运行所有测试
```bash
pytest tests/
```

### 运行特定测试文件
```bash
pytest tests/test_imports.py
pytest tests/test_models.py
```

### 运行特定测试类
```bash
pytest tests/test_models.py::TestMolmoModel
```

### 运行特定测试方法
```bash
pytest tests/test_models.py::TestMolmoModel::test_model_import
```

### 显示详细输出
```bash
pytest tests/ -v
```

### 显示打印输出
```bash
pytest tests/ -s
```

### 只运行失败的测试
```bash
pytest tests/ --lf
```

### 运行并显示覆盖率
```bash
pytest tests/ --cov=molmo --cov-report=html
```

## 测试环境要求

- Python 3.10+
- pytest >= 7.0.0
- 所有项目依赖（见 `setup.py`）

## 安装测试依赖

```bash
pip install -e ".[dev]"
```

## 注意事项

1. **CUDA 测试**: 某些测试需要 CUDA，如果没有 CUDA，这些测试会被跳过。
2. **数据依赖**: 某些数据集测试可能需要下载数据，如果数据不可用，测试会被跳过。
3. **可选依赖**: 某些功能需要可选依赖（如 boto3），如果不可用，相关测试会被跳过。

## 持续集成

这些测试可以在 CI/CD 流程中使用，确保代码更改不会破坏现有功能。



