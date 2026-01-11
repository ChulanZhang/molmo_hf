#!/usr/bin/env python3
"""
可视化GRPO训练中5个配置的生成过程。

这个脚本展示：
1. Stage1 Controller如何预测tier和insertion_position
2. Stage2 Controller如何预测top_k和num_active_blocks
3. 5个配置的完整参数组合
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

def visualize_config_generation(
    tier_logits: torch.Tensor,  # (5, 3)
    insertion_logits: torch.Tensor,  # (5, 5)
    knob2_logits: torch.Tensor,  # (5, 5)
    knob3_logits_list: List[torch.Tensor],  # List of (5, dynamic_length)
    insertion_positions: List[int],  # (5,)
    save_path: str = "results/latency_stats/config_generation_visualization.png",
):
    """
    可视化5个配置的生成过程。
    """
    num_configs = len(insertion_positions)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('GRPO训练中5个配置的生成过程', fontsize=16, fontweight='bold')
    
    # 1. Stage1: Tier分布
    ax = axes[0, 0]
    tier_probs = F.softmax(tier_logits, dim=-1).cpu().numpy()  # (5, 3)
    tier_names = ["low", "medium", "high"]
    x = np.arange(num_configs)
    width = 0.25
    for i, tier_name in enumerate(tier_names):
        ax.bar(x + i * width, tier_probs[:, i], width, label=tier_name, alpha=0.7)
    ax.set_xlabel('配置编号')
    ax.set_ylabel('概率')
    ax.set_title('Stage1: Tier预测分布')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Config {i+1}' for i in range(num_configs)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Stage1: Insertion Position分布
    ax = axes[0, 1]
    insertion_probs = F.softmax(insertion_logits, dim=-1).cpu().numpy()  # (5, 5)
    insertion_names = [f'After Block {i+1}' for i in range(5)]
    x = np.arange(num_configs)
    width = 0.15
    for i, ins_name in enumerate(insertion_names):
        ax.bar(x + i * width, insertion_probs[:, i], width, label=ins_name, alpha=0.7)
    ax.set_xlabel('配置编号')
    ax.set_ylabel('概率')
    ax.set_title('Stage1: Insertion Position预测分布')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f'Config {i+1}' for i in range(num_configs)])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Stage2: Top-K分布
    ax = axes[0, 2]
    knob2_probs = F.softmax(knob2_logits, dim=-1).cpu().numpy()  # (5, 5)
    topk_values = [4, 5, 6, 7, 8]
    x = np.arange(num_configs)
    width = 0.15
    for i, topk_val in enumerate(topk_values):
        ax.bar(x + i * width, knob2_probs[:, i], width, label=f'Top-K={topk_val}', alpha=0.7)
    ax.set_xlabel('配置编号')
    ax.set_ylabel('概率')
    ax.set_title('Stage2: Top-K预测分布')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f'Config {i+1}' for i in range(num_configs)])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. 配置参数表格
    ax = axes[1, 0]
    ax.axis('tight')
    ax.axis('off')
    
    # 采样实际值（模拟）
    tier_probs_np = F.softmax(tier_logits, dim=-1).cpu().numpy()
    insertion_probs_np = F.softmax(insertion_logits, dim=-1).cpu().numpy()
    knob2_probs_np = F.softmax(knob2_logits, dim=-1).cpu().numpy()
    
    # 采样tier
    tier_samples = []
    for i in range(num_configs):
        tier_idx = np.random.choice(3, p=tier_probs_np[i])
        tier_samples.append(tier_names[tier_idx])
    
    # 采样insertion_position（已经给出）
    insertion_samples = [f'After Block {pos}' for pos in insertion_positions]
    
    # 采样top_k
    topk_samples = []
    for i in range(num_configs):
        topk_idx = np.random.choice(5, p=knob2_probs_np[i])
        topk_samples.append(topk_values[topk_idx])
    
    # 采样num_active_blocks（基于insertion_position动态计算）
    blocks_samples = []
    for i, ins_pos in enumerate(insertion_positions):
        # 简化：假设从剩余blocks中选择，使得总blocks在[12,13,14,15,16]范围内
        remaining = 16 - ins_pos
        if remaining >= 11:
            # 可选：11, 12, 13, 14, 15
            options = [ins_pos + j for j in range(11, min(16, remaining + 1))]
            if len(options) > 0:
                blocks_samples.append(np.random.choice(options))
            else:
                blocks_samples.append(16)
        else:
            blocks_samples.append(16)
    
    table_data = [
        ['配置', 'Tier', 'Insertion', 'Top-K', 'Blocks'],
    ]
    for i in range(num_configs):
        table_data.append([
            f'Config {i+1}',
            tier_samples[i],
            insertion_samples[i],
            str(topk_samples[i]),
            str(blocks_samples[i]),
        ])
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.2, 0.25, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title('5个配置的完整参数组合', fontsize=12, fontweight='bold', pad=20)
    
    # 5. 配置多样性分析
    ax = axes[1, 1]
    # 计算每个参数的熵（多样性指标）
    tier_entropy = []
    insertion_entropy = []
    topk_entropy = []
    
    for i in range(num_configs):
        # Tier熵
        tier_entropy.append(-np.sum(tier_probs[i] * np.log(tier_probs[i] + 1e-10)))
        # Insertion熵
        insertion_entropy.append(-np.sum(insertion_probs[i] * np.log(insertion_probs[i] + 1e-10)))
        # Top-K熵
        topk_entropy.append(-np.sum(knob2_probs[i] * np.log(knob2_probs[i] + 1e-10)))
    
    x = np.arange(num_configs)
    width = 0.25
    ax.bar(x - width, tier_entropy, width, label='Tier', alpha=0.7)
    ax.bar(x, insertion_entropy, width, label='Insertion', alpha=0.7)
    ax.bar(x + width, topk_entropy, width, label='Top-K', alpha=0.7)
    ax.set_xlabel('配置编号')
    ax.set_ylabel('熵（多样性）')
    ax.set_title('每个配置的预测分布熵（越高越多样）')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Config {i+1}' for i in range(num_configs)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 配置空间覆盖
    ax = axes[1, 2]
    # 可视化配置在参数空间中的分布
    # X轴：tier (0=low, 1=medium, 2=high)
    # Y轴：top_k (4-8)
    # 颜色：num_active_blocks (12-16)
    # 大小：insertion_position (1-5)
    
    tier_to_num = {'low': 0, 'medium': 1, 'high': 2}
    tier_nums = [tier_to_num.get(t, 1) for t in tier_samples]
    
    scatter = ax.scatter(
        tier_nums,
        topk_samples,
        c=blocks_samples,
        s=[(ins_pos * 50) for ins_pos in insertion_positions],
        alpha=0.6,
        cmap='viridis',
        edgecolors='black',
        linewidths=1,
    )
    ax.set_xlabel('Tier (0=low, 1=medium, 2=high)')
    ax.set_ylabel('Top-K')
    ax.set_title('配置在参数空间中的分布')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Low', 'Medium', 'High'])
    ax.set_yticks([4, 5, 6, 7, 8])
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Num Active Blocks')
    
    # 添加配置编号标注
    for i in range(num_configs):
        ax.annotate(f'C{i+1}', (tier_nums[i], topk_samples[i]), 
                   fontsize=8, ha='center', va='center')
    
    plt.tight_layout()
    
    # 保存
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"可视化结果已保存到: {save_path}")
    
    return fig

def generate_example_configs():
    """
    生成示例配置数据用于可视化。
    """
    num_configs = 5
    
    # 模拟Stage1输出
    tier_logits = torch.randn(num_configs, 3)  # (5, 3)
    insertion_logits = torch.randn(num_configs, 5)  # (5, 5)
    
    # 模拟Stage2输出
    knob2_logits = torch.randn(num_configs, 5)  # (5, 5)
    
    # 模拟insertion_positions（实际从采样得到）
    insertion_positions = [1, 2, 1, 3, 2]  # 示例值
    
    # 模拟knob3_logits（动态长度）
    knob3_logits_list = []
    for ins_pos in insertion_positions:
        remaining = 16 - ins_pos
        num_options = min(5, remaining - 10 + 1)  # 假设选项数
        knob3_logits_list.append(torch.randn(1, num_options))
    
    return tier_logits, insertion_logits, knob2_logits, knob3_logits_list, insertion_positions

if __name__ == "__main__":
    # 生成示例数据
    tier_logits, insertion_logits, knob2_logits, knob3_logits_list, insertion_positions = generate_example_configs()
    
    # 可视化
    visualize_config_generation(
        tier_logits=tier_logits,
        insertion_logits=insertion_logits,
        knob2_logits=knob2_logits,
        knob3_logits_list=knob3_logits_list,
        insertion_positions=insertion_positions,
    )

