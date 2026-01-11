# TensorBoard vs Weights & Biases (wandb) 对比

## 快速对比

| 特性 | TensorBoard | wandb |
|------|-------------|-------|
| **类型** | 本地工具 | 云端服务（也可本地） |
| **安装** | `pip install tensorboard` | `pip install wandb` |
| **可视化** | 本地Web界面 | 云端Dashboard + 本地 |
| **实验管理** | 基础（按目录） | 强大（项目/组/标签） |
| **超参数追踪** | 需要手动记录 | 自动追踪 |
| **模型版本管理** | 无 | 支持Artifacts |
| **协作** | 需要共享文件 | 云端自动同步 |
| **离线支持** | 完全支持 | 支持（sync later） |
| **成本** | 免费 | 免费（个人/学术），团队版付费 |
| **学习曲线** | 简单 | 中等 |
| **集成** | PyTorch原生 | 需要初始化 |

## 详细对比

### TensorBoard 优势

1. **简单直接**
   - PyTorch原生支持，无需额外配置
   - 本地运行，数据完全私有
   - 适合快速实验和调试

2. **轻量级**
   - 不需要账号注册
   - 不需要网络连接（完全离线）
   - 资源占用小

3. **适合场景**
   - 单机训练
   - 快速迭代和调试
   - 数据敏感项目（本地存储）

### wandb 优势

1. **强大的实验管理**
   - 自动追踪超参数
   - 项目/组/标签组织实验
   - 实验对比和筛选
   - 实验重现性追踪

2. **协作功能**
   - 团队共享dashboard
   - 实时查看队友实验
   - 评论和标注

3. **高级功能**
   - 模型版本管理（Artifacts）
   - 系统资源监控（GPU/CPU/内存）
   - 自动超参数扫描
   - 报告生成

4. **适合场景**
   - 团队协作
   - 长期实验追踪
   - 需要对比多个实验
   - 需要模型版本管理

## 针对你的Controller训练场景

### 当前情况
- 单机训练（可能多GPU）
- GRPO训练，需要追踪多个指标
- 需要对比不同配置的效果
- 可能需要长期实验追踪

### 建议

#### 方案1：同时支持两者（推荐）✅
**优点**：
- 灵活性：可以选择使用哪个
- TensorBoard用于本地快速查看
- wandb用于实验管理和团队协作
- 代码改动小（两者可以共存）

**实现**：
```python
# 添加 --use_wandb 参数
if args.use_wandb:
    import wandb
    wandb.init(project="molmo-controller", ...)
    wandb.log(metrics, step=epoch)
```

#### 方案2：只使用 wandb（如果团队协作）
**优点**：
- 统一的实验管理
- 更好的协作体验
- 自动超参数追踪

**缺点**：
- 需要注册账号
- 需要网络连接（或配置离线模式）

#### 方案3：只使用 TensorBoard（如果单机/快速迭代）
**优点**：
- 简单直接
- 无需额外配置
- 完全离线

**缺点**：
- 实验管理功能弱
- 难以对比多个实验

## 推荐实现：同时支持两者

我建议实现一个灵活的日志系统，支持：
1. **TensorBoard**（默认）：本地快速查看
2. **wandb**（可选）：通过 `--use_wandb` 启用

这样你可以：
- 本地开发时用TensorBoard快速查看
- 正式训练时用wandb进行实验管理
- 两者可以同时使用，互不冲突

## 代码示例

```python
# 初始化
if args.use_wandb:
    import wandb
    wandb.init(
        project="molmo-controller",
        name=f"grpo-{args.group_size}-{args.lr}",
        config=vars(args),
    )

# 记录指标
if args.use_wandb:
    wandb.log({
        'Train/Loss': loss,
        'Train/Reward_Mean': reward_mean,
        ...
    }, step=epoch)

# TensorBoard（始终启用）
writer.add_scalar('Train/Loss', loss, epoch)
```

## 最终建议

**对于你的场景，我推荐同时支持两者**：
1. TensorBoard作为默认（简单、快速）
2. wandb作为可选（通过 `--use_wandb` 启用）
3. 两者可以同时使用，互不冲突

这样既保持了简单性，又提供了强大的实验管理能力。

