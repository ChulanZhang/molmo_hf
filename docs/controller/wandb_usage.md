# Weights & Biases (wandb) 使用指南

## 快速开始

### 1. 安装 wandb

```bash
pip install wandb
```

### 2. 登录 wandb

首次使用需要登录：

```bash
wandb login
```

会提示输入 API key，可以从 https://wandb.ai/settings 获取。

### 3. 启用 wandb 日志

#### 方法1：命令行参数

```bash
python experiments/controller/train_joint_controller.py \
    --use_wandb \
    --wandb_project my-project \
    --wandb_entity my-team \
    --wandb_name my-experiment \
    ...
```

#### 方法2：修改 run_training.sh

编辑 `experiments/controller/run_training.sh`：

```bash
USE_WANDB="true"
WANDB_PROJECT="molmo-controller"  # 可选，默认是 "molmo-controller"
WANDB_ENTITY="my-team"             # 可选，默认是你的用户名
WANDB_NAME="grpo-experiment-1"     # 可选，默认自动生成
```

## 参数说明

### `--use_wandb`
启用 wandb 日志记录。默认：`False`（只使用 TensorBoard）

### `--wandb_project`
W&B 项目名称。默认：`"molmo-controller"`

### `--wandb_entity`
W&B 实体/团队名称。默认：你的用户名（如果未设置）

### `--wandb_name`
W&B 运行名称。默认：自动生成（格式：`grpo-g{group_size}-lr{lr}-{timestamp}`）

## 自动记录的内容

### 超参数（Config）
- 所有训练超参数（batch_size, lr, group_size 等）
- 模型路径、数据集名称
- 设备配置

### 训练指标（每个 epoch）
- `Train/Loss`
- `Train/Reward_Mean` 和 `Train/Reward_Std`
- `Train/Accuracy_Mean` 和 `Train/Accuracy_Std`
- `Train/Latency_Mean` 和 `Train/Latency_Std`
- `Train/Budget_Violation_Rate`

### 验证指标（每个 epoch）
- `Val/Reward_Mean`
- `Val/Accuracy_Mean`
- `Val/Latency_Mean`
- `Val/Budget_Violation_Rate`

## 查看结果

### 1. Web Dashboard
训练开始后，会输出 W&B 运行 URL，例如：
```
View run at: https://wandb.ai/your-username/molmo-controller/runs/abc123
```

### 2. 命令行
```bash
wandb status
```

### 3. 离线模式
如果网络不稳定，可以使用离线模式：
```bash
WANDB_MODE=offline python experiments/controller/train_joint_controller.py --use_wandb ...
```

训练完成后同步：
```bash
wandb sync wandb/offline-run-*
```

## TensorBoard vs wandb

### 同时使用（推荐）
- **TensorBoard**：本地快速查看，无需网络
- **wandb**：云端实验管理，团队协作

两者可以同时使用，互不冲突。

### 只使用 TensorBoard
如果不添加 `--use_wandb`，默认只使用 TensorBoard。

### 只使用 wandb
可以修改代码禁用 TensorBoard（不推荐），但建议两者都保留。

## 高级功能

### 1. 实验对比
在 wandb dashboard 中，可以：
- 对比多个实验的指标
- 筛选和排序实验
- 创建报告

### 2. 超参数扫描
wandb 支持自动超参数扫描，可以集成到训练脚本中。

### 3. 模型版本管理
可以使用 wandb Artifacts 管理模型检查点。

## 故障排除

### wandb 未安装
如果看到警告：
```
wandb not installed. Install with: pip install wandb
```

解决方法：
```bash
pip install wandb
```

### 登录问题
如果提示需要登录：
```bash
wandb login
```

### 网络问题
使用离线模式：
```bash
WANDB_MODE=offline python ... --use_wandb ...
```

## 示例

### 基本使用
```bash
python experiments/controller/train_joint_controller.py \
    --results_dir results/core_exp_h100/5run_2000samples_w_new_importance_score \
    --dataset_names text_vqa coco_2014_vqa \
    --model_path checkpoints \
    --output_dir checkpoints/joint_controller \
    --batch_size 1 \
    --num_epochs 100 \
    --lr 1e-4 \
    --group_size 5 \
    --use_wandb
```

### 自定义项目名称
```bash
python experiments/controller/train_joint_controller.py \
    ... \
    --use_wandb \
    --wandb_project my-research \
    --wandb_name grpo-baseline
```

### 团队协作
```bash
python experiments/controller/train_joint_controller.py \
    ... \
    --use_wandb \
    --wandb_project molmo-controller \
    --wandb_entity my-team \
    --wandb_name experiment-1
```

## 与 TensorBoard 的对比

| 特性 | TensorBoard | wandb |
|------|-------------|-------|
| 查看方式 | 本地 Web | 云端 Dashboard |
| 实验管理 | 基础 | 强大 |
| 协作 | 需要共享文件 | 自动同步 |
| 离线支持 | 完全支持 | 支持（sync later） |
| 超参数追踪 | 手动 | 自动 |
| 成本 | 免费 | 免费（个人/学术） |

## 最佳实践

1. **开发阶段**：使用 TensorBoard 快速查看
2. **正式训练**：同时启用 wandb 进行实验管理
3. **团队协作**：使用 wandb 共享实验结果
4. **长期追踪**：使用 wandb 管理所有实验历史

