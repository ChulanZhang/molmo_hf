#!/usr/bin/env python3
"""
可视化GRPO训练中的Forward Pass流程。

展示：
1. Stage1预测（0次forward）
2. Latency token提取（5次部分forward）
3. Stage2预测（0次forward）
4. 最终执行（5次完整forward）
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def visualize_forward_passes(save_path: str = "results/latency_stats/forward_passes_visualization.png"):
    """
    可视化forward pass流程。
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 标题
    ax.text(5, 11.5, 'GRPO训练中的Forward Pass流程', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    
    # 阶段1：Stage1预测（0次forward）
    stage1_box = FancyBboxPatch((0.5, 9), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightblue', 
                                edgecolor='black', linewidth=2)
    ax.add_patch(stage1_box)
    ax.text(1.75, 10.25, 'Stage1预测', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(1.75, 9.8, '输入: lang_feat + budget_feat', ha='center', va='center', 
            fontsize=9)
    ax.text(1.75, 9.4, '输出: tier_logits + insertion_logits', ha='center', va='center', 
            fontsize=9)
    ax.text(1.75, 9.0, 'Forward: 0次 (仅embedding + MLP)', ha='center', va='center', 
            fontsize=9, style='italic', color='green')
    
    # 采样1
    sample1_box = FancyBboxPatch((3.2, 9), 1.5, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='lightyellow', 
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(sample1_box)
    ax.text(3.95, 10.25, '采样5次', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(3.95, 9.8, '得到5个配置:', ha='center', va='center', fontsize=8)
    ax.text(3.95, 9.5, '(tier, insertion_pos)', ha='center', va='center', fontsize=8)
    ax.text(3.95, 9.2, '例如: (low,1), (med,2), ...', ha='center', va='center', 
            fontsize=7, style='italic')
    
    # 箭头1
    arrow1 = FancyArrowPatch((2.5, 9.75), (3.2, 9.75),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # 阶段2：Latency token提取（5次部分forward）
    stage2_box = FancyBboxPatch((0.5, 6.5), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightcoral', 
                                edgecolor='black', linewidth=2)
    ax.add_patch(stage2_box)
    ax.text(1.75, 7.75, 'Latency Token提取', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(1.75, 7.3, '对每个配置:', ha='center', va='center', fontsize=9)
    ax.text(1.75, 7.0, '运行到insertion_position', ha='center', va='center', 
            fontsize=9)
    ax.text(1.75, 6.7, 'Forward: 5次部分', ha='center', va='center', 
            fontsize=9, style='italic', color='red', fontweight='bold')
    
    # 5个配置的latency token提取
    config_boxes = []
    for i in range(5):
        x_pos = 3.2 + i * 1.2
        config_box = FancyBboxPatch((x_pos, 6.5), 1.0, 1.5, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='lightgreen', 
                                    edgecolor='black', linewidth=1)
        ax.add_patch(config_box)
        ax.text(x_pos + 0.5, 7.75, f'Config {i+1}', ha='center', va='center', 
                fontsize=8, fontweight='bold')
        ax.text(x_pos + 0.5, 7.3, f'pos={i+1}', ha='center', va='center', 
                fontsize=7)
        ax.text(x_pos + 0.5, 7.0, '→ block 0..i', ha='center', va='center', 
                fontsize=7)
        ax.text(x_pos + 0.5, 6.7, '提取token', ha='center', va='center', 
                fontsize=7)
        config_boxes.append((x_pos + 0.5, 6.5))
    
    # 箭头2
    arrow2 = FancyArrowPatch((2.5, 7.25), (3.2, 7.25),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # 阶段3：Stage2预测（0次forward）
    stage3_box = FancyBboxPatch((0.5, 4), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightblue', 
                                edgecolor='black', linewidth=2)
    ax.add_patch(stage3_box)
    ax.text(1.75, 5.25, 'Stage2预测', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(1.75, 4.8, '输入: latency_token + insertion_pos', ha='center', va='center', 
            fontsize=9)
    ax.text(1.75, 4.4, '输出: knob2_logits + knob3_logits', ha='center', va='center', 
            fontsize=9)
    ax.text(1.75, 4.0, 'Forward: 0次 (仅MLP)', ha='center', va='center', 
            fontsize=9, style='italic', color='green')
    
    # 采样2
    sample2_box = FancyBboxPatch((3.2, 4), 1.5, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='lightyellow', 
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(sample2_box)
    ax.text(3.95, 5.25, '采样5次', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax.text(3.95, 4.8, '得到5个配置:', ha='center', va='center', fontsize=8)
    ax.text(3.95, 4.5, '(top_k, num_blocks)', ha='center', va='center', fontsize=8)
    ax.text(3.95, 4.2, '例如: (4,12), (6,14), ...', ha='center', va='center', 
            fontsize=7, style='italic')
    
    # 箭头3
    arrow3 = FancyArrowPatch((2.5, 4.75), (3.2, 4.75),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    # 阶段4：最终执行（5次完整forward）
    stage4_box = FancyBboxPatch((0.5, 1.5), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightcoral', 
                                edgecolor='black', linewidth=2)
    ax.add_patch(stage4_box)
    ax.text(1.75, 2.75, '最终执行', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(1.75, 2.3, '对每个配置:', ha='center', va='center', fontsize=9)
    ax.text(1.75, 2.0, 'model.generate()', ha='center', va='center', 
            fontsize=9)
    ax.text(1.75, 1.7, 'Forward: 5次完整', ha='center', va='center', 
            fontsize=9, style='italic', color='red', fontweight='bold')
    
    # 5个配置的最终执行
    for i in range(5):
        x_pos = 3.2 + i * 1.2
        exec_box = FancyBboxPatch((x_pos, 1.5), 1.0, 1.5, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor='orange', 
                                  edgecolor='black', linewidth=1)
        ax.add_patch(exec_box)
        ax.text(x_pos + 0.5, 2.75, f'Config {i+1}', ha='center', va='center', 
                fontsize=8, fontweight='bold')
        ax.text(x_pos + 0.5, 2.3, 'Prefill', ha='center', va='center', 
                fontsize=7)
        ax.text(x_pos + 0.5, 2.0, '+ Decode', ha='center', va='center', 
                fontsize=7)
        ax.text(x_pos + 0.5, 1.7, '(64 tokens)', ha='center', va='center', 
                fontsize=7)
    
    # 箭头4
    arrow4 = FancyArrowPatch((2.5, 2.25), (3.2, 2.25),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    # 垂直箭头连接各个阶段
    # Stage1 -> Stage2
    arrow_v1 = FancyArrowPatch((1.75, 9), (1.75, 8),
                              arrowstyle='->', mutation_scale=20, 
                              linewidth=2, color='blue', linestyle='--')
    ax.add_patch(arrow_v1)
    
    # Stage2 -> Stage3
    arrow_v2 = FancyArrowPatch((1.75, 6.5), (1.75, 5.5),
                              arrowstyle='->', mutation_scale=20, 
                              linewidth=2, color='blue', linestyle='--')
    ax.add_patch(arrow_v2)
    
    # Stage3 -> Stage4
    arrow_v3 = FancyArrowPatch((1.75, 4), (1.75, 3),
                              arrowstyle='->', mutation_scale=20, 
                              linewidth=2, color='blue', linestyle='--')
    ax.add_patch(arrow_v3)
    
    # 总结框
    summary_box = FancyBboxPatch((8.5, 1.5), 1.5, 8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightgray', 
                                edgecolor='black', linewidth=2)
    ax.add_patch(summary_box)
    ax.text(9.25, 9.5, '总结', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    
    summary_text = [
        'Stage1预测:',
        '  0次forward',
        '',
        'Latency提取:',
        '  5次部分forward',
        '',
        'Stage2预测:',
        '  0次forward',
        '',
        '最终执行:',
        '  5次完整forward',
        '',
        '总计:',
        '  10次forward',
        '  (5部分+5完整)'
    ]
    
    y_start = 8.5
    for i, text in enumerate(summary_text):
        ax.text(9.25, y_start - i * 0.4, text, ha='center', va='center', 
                fontsize=9 if 'forward' in text or '总计' in text else 8,
                fontweight='bold' if 'forward' in text or '总计' in text else 'normal',
                color='red' if 'forward' in text and '次' in text else 'black')
    
    plt.tight_layout()
    
    # 保存
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"可视化结果已保存到: {save_path}")
    
    return fig

if __name__ == "__main__":
    visualize_forward_passes()

