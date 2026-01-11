# wandb 实现说明

## 与官方示例的对比

我们的实现参考了 [wandb 官方示例](https://wandb.ai)，并遵循了最佳实践。

### 官方示例关键点

```python
run = wandb.init(
    entity="your-entity",
    project="your-project",
    config={...},
)

# 使用 run.log() 而不是 wandb.log()
run.log({"acc": acc, "loss": loss})

# 使用 run.finish() 而不是 wandb.finish()
run.finish()
```

### 我们的实现

✅ **已实现的功能**：

1. **wandb.init()** - 正确初始化
   - `entity`: 通过 `--wandb_entity` 参数设置
   - `project`: 通过 `--wandb_project` 参数设置（默认 "molmo-controller"）
   - `config`: 自动记录所有超参数
   - `tags`: 自动添加相关标签
   - `name`: 通过 `--wandb_name` 参数设置（默认自动生成）

2. **run.log()** - 使用 `run.log()` 而不是 `wandb.log()`
   - 更符合官方推荐
   - 更明确，因为我们已经有 `run` 对象

3. **run.finish()** - 使用 `run.finish()` 而不是 `wandb.finish()`
   - 在训练结束时正确关闭 run

4. **错误处理** - 如果 wandb 未安装，会优雅降级
   - 显示警告但继续训练
   - 不影响 TensorBoard 的使用

### 改进点

根据官方示例，我们做了以下优化：

1. ✅ **使用 `run.log()` 而不是 `wandb.log()`**
   - 更符合官方风格
   - 代码更清晰

2. ✅ **使用 `run.finish()` 而不是 `wandb.finish()`**
   - 更符合官方风格

3. ✅ **完整的 config 记录**
   - 所有超参数都记录到 config
   - 便于实验对比和重现

4. ✅ **自动标签**
   - 添加 'grpo', 'controller', 'joint-training'
   - 自动添加数据集名称作为标签

### 与官方示例的差异（合理的）

1. **entity 参数可选**
   - 官方示例中 entity 是必需的（在他们的例子中）
   - 我们的实现中是可选的，如果不提供会使用默认值
   - 这是合理的，因为不是所有用户都需要指定 entity

2. **自动生成 run name**
   - 官方示例中没有指定 name
   - 我们提供了自动生成功能，格式：`grpo-g{group_size}-lr{lr}-{timestamp}`
   - 这样更容易识别不同的实验

3. **同时支持 TensorBoard**
   - 官方示例只使用 wandb
   - 我们同时支持 TensorBoard 和 wandb
   - 这样更灵活，用户可以选择使用哪个

### 最佳实践

我们的实现遵循了以下最佳实践：

1. ✅ **错误处理** - 如果 wandb 未安装，优雅降级
2. ✅ **向后兼容** - 默认不使用 wandb，不影响现有代码
3. ✅ **灵活性** - 支持自定义 project、entity、name
4. ✅ **完整性** - 记录所有相关指标和超参数
5. ✅ **官方风格** - 使用 `run.log()` 和 `run.finish()`

### 使用示例

```python
# 基本使用（自动生成 run name）
run = wandb.init(
    project="molmo-controller",
    config={
        "lr": 1e-4,
        "group_size": 5,
        ...
    },
)

# 记录指标
run.log({
    "Train/Loss": loss,
    "Train/Accuracy": acc,
}, step=epoch)

# 结束 run
run.finish()
```

这与官方示例完全一致！

