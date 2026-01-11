# Git分支策略

> **创建日期**: 2026-01-11  
> **状态**: 当前策略

## 分支结构

### main 分支
- **用途**: 稳定版本，包含two-stage controller实现
- **状态**: 当前包含完整的two-stage controller架构

### two-stage-controller 分支
- **用途**: 备份分支，保存two-stage controller的完整实现
- **创建时间**: 2026-01-11
- **内容**: 与main分支相同，作为two-stage实现的备份
- **状态**: 备份分支，不用于主力开发

### one-stage-controller 分支
- **用途**: 主力开发分支，用于one-stage controller开发
- **创建时间**: 2026-01-11
- **内容**: 从main分支创建，将进行one-stage重构
- **状态**: 主力开发分支

## 分支关系

```
main (two-stage实现)
├── two-stage-controller (备份)
└── one-stage-controller (主力开发) ← 当前工作分支
```

## 开发流程

### 当前阶段
1. **one-stage-controller分支**: 主力开发
   - 实现one-stage controller架构
   - 简化controller设计
   - 移除two-stage相关代码

2. **two-stage-controller分支**: 备份保留
   - 保留完整的two-stage实现
   - 作为参考和回退点

### 未来计划
- one-stage controller开发完成后，可以考虑合并回main
- 或者保持one-stage-controller作为新的main分支

## 代码整理

### Two-Stage相关文件（保留在two-stage-controller分支）
- `experiments/controller/train_two_stage_controller.py` (已删除)
- `experiments/controller/stage2_grpo_trainer.py` (已删除)
- Two-stage相关的文档和设计

### One-Stage相关文件（在one-stage-controller分支开发）
- 简化的controller架构
- 单阶段预测逻辑
- 更新的训练脚本

---

**最后更新**: 2026-01-11

