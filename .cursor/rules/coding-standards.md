# 编码标准

## Python 代码规范

- 遵循 PEP 8 代码风格指南
- 使用 4 个空格缩进，不使用 Tab
- 行长度限制为 100 字符
- 所有字符串使用双引号 `"`，除非字符串内包含双引号
- 导入语句按标准库、第三方库、本地库的顺序组织，每组之间用空行分隔

## 命名约定

- **变量和函数名**：使用 `snake_case`
  - 示例：`model_path`, `run_training`, `get_config`
  
- **类名**：使用 `PascalCase`
  - 示例：`JointGRPOTrainer`, `FeatureExtractor`, `ControllerConfig`
  
- **常量**：使用 `UPPER_SNAKE_CASE`
  - 示例：`MAX_TOKENS`, `DEFAULT_BATCH_SIZE`
  
- **私有方法/属性**：使用 `_leading_underscore`
  - 示例：`_compute_loss`, `_internal_state`

## 类型注解

- 所有函数参数和返回值都应该有类型注解
- 使用 `typing` 模块的类型（如 `List`, `Dict`, `Optional`, `Union`）
- 示例：
  ```python
  def process_data(
      data: List[Dict[str, Any]], 
      batch_size: int = 32
  ) -> Optional[torch.Tensor]:
      ...
  ```

## 文档字符串（Docstrings）

- 所有公共函数、类和模块必须有 docstring
- 使用 Google 风格的 docstring
- 示例：
  ```python
  def train_model(
      model: nn.Module,
      dataloader: DataLoader,
      epochs: int = 10
  ) -> Dict[str, float]:
      """训练模型。
      
      Args:
          model: 要训练的模型
          dataloader: 数据加载器
          epochs: 训练轮数，默认为 10
          
      Returns:
          包含训练指标的字典，如 {'loss': 0.5, 'accuracy': 0.9}
      """
      ...
  ```

## 注释规范

- 复杂逻辑必须添加行内注释
- 注释应该解释"为什么"而不是"是什么"
- 使用中文或英文注释，保持项目内一致
- 避免显而易见的注释

## 错误处理

- 使用有意义的异常类型
- 异常消息应该清晰描述问题
- 捕获异常时记录足够的上下文信息
- 示例：
  ```python
  try:
      model.load_state_dict(checkpoint)
  except KeyError as e:
      raise ValueError(f"检查点文件缺少必需的键: {e}") from e
  ```

