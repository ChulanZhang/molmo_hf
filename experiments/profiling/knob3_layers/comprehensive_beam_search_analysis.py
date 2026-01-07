#!/usr/bin/env python3
"""
Comprehensive analysis and visualization of beam search results.
Includes:
1. Rankings by number of removed blocks (1, 2, 3, 4)
2. Task type differences (VQA, Captioning, MC, Document QA, etc.)
3. Block position effects (early, middle, late)
4. Combination effects vs individual effects
5. Cross-dataset consistency
6. Accuracy drop distributions
7. Stability analysis (std dev)
8. Dataset-specific responses
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Task type mapping
TASK_TYPES = {
    "coco-2014-vqa": "VQA",
    "text-vqa": "VQA",
    "okvqa": "VQA",
    "coco-caption": "Captioning",
    "science-qa-img": "Multiple Choice",
    "doc-qa": "Document QA",
    "st-qa": "Scene Text QA",
    "tally-qa": "Exact Match",
}

# Block position categories
EARLY_BLOCKS = [0, 1, 2, 3, 4]
MIDDLE_BLOCKS = [5, 6, 7, 8, 9, 10]
LATE_BLOCKS = [11, 12, 13, 14, 15]

def get_block_position_category(block_idx: int) -> str:
    """Categorize block by position."""
    if block_idx in EARLY_BLOCKS:
        return "Early (0-4)"
    elif block_idx in MIDDLE_BLOCKS:
        return "Middle (5-10)"
    else:
        return "Late (11-15)"

def load_analysis_results(analysis_file: Path) -> Dict:
    """Load cross-dataset analysis results."""
    with open(analysis_file, 'r') as f:
        return json.load(f)

def analyze_by_num_removed(results: Dict) -> Dict[int, List[Dict]]:
    """Group combinations by number of removed blocks."""
    by_num_removed = {1: [], 2: [], 3: [], 4: []}
    
    for key, stats in results['combination_stats'].items():
        num_removed = stats['num_removed']
        if num_removed in by_num_removed:
            # Only include combinations tested on at least 2 datasets and with non-negative drop
            if stats['num_datasets_tested'] >= 2 and stats['avg_accuracy_drop'] >= 0:
                by_num_removed[num_removed].append(stats)
    
    # Sort by average accuracy drop
    for num_removed in by_num_removed:
        by_num_removed[num_removed].sort(key=lambda x: x['avg_accuracy_drop'])
    
    return by_num_removed

def plot_rankings_by_num_removed(by_num_removed: Dict[int, List[Dict]], output_dir: Path, top_k: int = 10):
    """Plot rankings for each number of removed blocks."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Top Block Combinations by Number of Removed Blocks', fontsize=16, fontweight='bold')
    
    for idx, num_removed in enumerate([1, 2, 3, 4]):
        ax = axes[idx // 2, idx % 2]
        combinations = by_num_removed[num_removed][:top_k]
        
        if not combinations:
            ax.text(0.5, 0.5, f'No combinations for {num_removed} block(s)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Removing {num_removed} Block(s)')
            continue
        
        # Extract data
        labels = [str(combo['removed_blocks']).replace(' ', '') for combo in combinations]
        drops = [combo['avg_accuracy_drop'] * 100 for combo in combinations]
        stds = [combo['std_accuracy_drop'] * 100 for combo in combinations]
        num_datasets = [combo['num_datasets_tested'] for combo in combinations]
        
        # Create bar plot
        x_pos = np.arange(len(labels))
        bars = ax.barh(x_pos, drops, xerr=stds, capsize=3, alpha=0.7)
        
        # Color by number of datasets tested
        colors = plt.cm.viridis(np.array(num_datasets) / max(num_datasets) if max(num_datasets) > 0 else 1)
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add dataset count annotations
        for i, (drop, num_ds) in enumerate(zip(drops, num_datasets)):
            ax.text(drop + stds[i] + 0.5, i, f'{num_ds}ds', 
                   va='center', fontsize=8, alpha=0.7)
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Average Accuracy Drop (%)', fontsize=11)
        ax.set_title(f'Removing {num_removed} Block(s) - Top {len(combinations)}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rankings_by_num_removed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved rankings_by_num_removed.png")

def analyze_by_task_type(results: Dict) -> Dict[str, Dict]:
    """Analyze results by task type."""
    task_stats = defaultdict(lambda: {
        'combinations': [],
        'avg_drops': [],
        'std_drops': [],
        'num_datasets': 0,
    })
    
    for key, stats in results['combination_stats'].items():
        if stats['num_datasets_tested'] < 2 or stats['avg_accuracy_drop'] < 0:
            continue
        
        # Group by task type based on datasets tested
        task_types_in_combo = set()
        for dataset in stats['datasets']:
            if dataset in TASK_TYPES:
                task_types_in_combo.add(TASK_TYPES[dataset])
        
        # If combination spans multiple task types, add to each
        for task_type in task_types_in_combo:
            task_stats[task_type]['combinations'].append(stats)
            task_stats[task_type]['avg_drops'].append(stats['avg_accuracy_drop'])
            task_stats[task_type]['std_drops'].append(stats['std_accuracy_drop'])
            task_stats[task_type]['num_datasets'] = max(
                task_stats[task_type]['num_datasets'],
                stats['num_datasets_tested']
            )
    
    return dict(task_stats)

def plot_task_type_comparison(task_stats: Dict[str, Dict], output_dir: Path):
    """Plot comparison across task types."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Block Importance Analysis by Task Type', fontsize=16, fontweight='bold')
    
    # 1. Average drop by task type
    ax1 = axes[0, 0]
    task_types = list(task_stats.keys())
    avg_drops = [np.mean(task_stats[t]['avg_drops']) * 100 for t in task_types]
    std_drops = [np.std(task_stats[t]['avg_drops']) * 100 for t in task_types]
    
    bars = ax1.barh(task_types, avg_drops, xerr=std_drops, capsize=5, alpha=0.7)
    ax1.set_xlabel('Average Accuracy Drop (%)', fontsize=11)
    ax1.set_title('Mean Drop by Task Type', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Distribution of drops by task type
    ax2 = axes[0, 1]
    for task_type in task_types:
        drops = np.array(task_stats[task_type]['avg_drops']) * 100
        ax2.hist(drops, bins=20, alpha=0.5, label=task_type, density=True)
    ax2.set_xlabel('Accuracy Drop (%)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Drop Distribution by Task Type', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Stability (std dev) by task type
    ax3 = axes[1, 0]
    avg_stds = [np.mean(task_stats[t]['std_drops']) * 100 for t in task_types]
    std_stds = [np.std(task_stats[t]['std_drops']) * 100 for t in task_types]
    
    bars = ax3.barh(task_types, avg_stds, xerr=std_stds, capsize=5, alpha=0.7, color='orange')
    ax3.set_xlabel('Average Std Dev (%)', fontsize=11)
    ax3.set_title('Stability by Task Type', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Number of combinations by task type
    ax4 = axes[1, 1]
    num_combos = [len(task_stats[t]['combinations']) for t in task_types]
    bars = ax4.barh(task_types, num_combos, alpha=0.7, color='green')
    ax4.set_xlabel('Number of Combinations', fontsize=11)
    ax4.set_title('Combination Count by Task Type', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'task_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved task_type_comparison.png")

def analyze_by_block_position(results: Dict) -> Dict[str, Dict]:
    """Analyze results by block position (early, middle, late)."""
    position_stats = defaultdict(lambda: {
        'combinations': [],
        'avg_drops': [],
        'std_drops': [],
    })
    
    for key, stats in results['combination_stats'].items():
        if stats['num_datasets_tested'] < 2 or stats['avg_accuracy_drop'] < 0:
            continue
        
        # Categorize by position of removed blocks
        removed = stats['removed_blocks']
        positions = [get_block_position_category(b) for b in removed]
        
        # If all blocks in same category
        if len(set(positions)) == 1:
            pos = positions[0]
            position_stats[pos]['combinations'].append(stats)
            position_stats[pos]['avg_drops'].append(stats['avg_accuracy_drop'])
            position_stats[pos]['std_drops'].append(stats['std_accuracy_drop'])
        else:
            # Mixed positions
            position_stats['Mixed']['combinations'].append(stats)
            position_stats['Mixed']['avg_drops'].append(stats['avg_accuracy_drop'])
            position_stats['Mixed']['std_drops'].append(stats['std_accuracy_drop'])
    
    return dict(position_stats)

def plot_block_position_analysis(position_stats: Dict[str, Dict], output_dir: Path):
    """Plot analysis by block position."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Block Importance by Position (Early/Middle/Late)', fontsize=16, fontweight='bold')
    
    positions = ['Early (0-4)', 'Middle (5-10)', 'Late (11-15)', 'Mixed']
    positions = [p for p in positions if p in position_stats]
    
    # 1. Average drop
    ax1 = axes[0]
    avg_drops = [np.mean(position_stats[p]['avg_drops']) * 100 for p in positions]
    std_drops = [np.std(position_stats[p]['avg_drops']) * 100 for p in positions]
    bars = ax1.bar(positions, avg_drops, yerr=std_drops, capsize=5, alpha=0.7)
    ax1.set_ylabel('Average Accuracy Drop (%)', fontsize=11)
    ax1.set_title('Mean Drop by Block Position', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Stability
    ax2 = axes[1]
    avg_stds = [np.mean(position_stats[p]['std_drops']) * 100 for p in positions]
    std_stds = [np.std(position_stats[p]['std_drops']) * 100 for p in positions]
    bars = ax2.bar(positions, avg_stds, yerr=std_stds, capsize=5, alpha=0.7, color='orange')
    ax2.set_ylabel('Average Std Dev (%)', fontsize=11)
    ax2.set_title('Stability by Block Position', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Count
    ax3 = axes[2]
    counts = [len(position_stats[p]['combinations']) for p in positions]
    bars = ax3.bar(positions, counts, alpha=0.7, color='green')
    ax3.set_ylabel('Number of Combinations', fontsize=11)
    ax3.set_title('Combination Count by Position', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'block_position_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved block_position_analysis.png")

def analyze_combination_vs_individual(results: Dict) -> Dict:
    """Compare combination effects vs individual block effects."""
    # Get single block removals
    single_blocks = {}
    for key, stats in results['combination_stats'].items():
        if stats['num_removed'] == 1 and stats['num_datasets_tested'] >= 2:
            block = stats['removed_blocks'][0]
            single_blocks[block] = stats['avg_accuracy_drop']
    
    # Get two-block combinations
    two_blocks = {}
    for key, stats in results['combination_stats'].items():
        if stats['num_removed'] == 2 and stats['num_datasets_tested'] >= 2:
            blocks = tuple(sorted(stats['removed_blocks']))
            two_blocks[blocks] = {
                'drop': stats['avg_accuracy_drop'],
                'individual_sum': sum(single_blocks.get(b, 0) for b in blocks),
                'synergy': stats['avg_accuracy_drop'] - sum(single_blocks.get(b, 0) for b in blocks),
            }
    
    return {
        'single_blocks': single_blocks,
        'two_blocks': two_blocks,
    }

def plot_combination_vs_individual(combo_analysis: Dict, output_dir: Path):
    """Plot combination effects vs individual effects."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Combination Effects vs Individual Block Effects', fontsize=16, fontweight='bold')
    
    two_blocks = combo_analysis['two_blocks']
    
    # Extract data
    actual_drops = [v['drop'] * 100 for v in two_blocks.values()]
    predicted_drops = [v['individual_sum'] * 100 for v in two_blocks.values()]
    synergies = [v['synergy'] * 100 for v in two_blocks.values()]
    
    # 1. Actual vs Predicted (sum of individuals)
    ax1 = axes[0, 0]
    ax1.scatter(predicted_drops, actual_drops, alpha=0.6, s=50)
    min_val = min(min(predicted_drops), min(actual_drops))
    max_val = max(max(predicted_drops), max(actual_drops))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    ax1.set_xlabel('Predicted Drop (Sum of Individuals) (%)', fontsize=11)
    ax1.set_ylabel('Actual Drop (%)', fontsize=11)
    ax1.set_title('Actual vs Predicted Drop', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Synergy distribution
    ax2 = axes[0, 1]
    ax2.hist(synergies, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='r', linestyle='--', linewidth=2, label='No synergy')
    ax2.set_xlabel('Synergy Effect (%)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Synergy Distribution (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Top synergistic combinations
    ax3 = axes[1, 0]
    sorted_synergies = sorted(two_blocks.items(), key=lambda x: x[1]['synergy'], reverse=True)
    top_10 = sorted_synergies[:10]
    labels = [str(list(k)).replace(' ', '') for k, v in top_10]
    synergy_values = [v['synergy'] * 100 for k, v in top_10]
    bars = ax3.barh(range(len(labels)), synergy_values, alpha=0.7)
    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_xlabel('Synergy Effect (%)', fontsize=11)
    ax3.set_title('Top 10 Synergistic Combinations', fontsize=12, fontweight='bold')
    ax3.axvline(0, color='r', linestyle='--', linewidth=1)
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_yaxis()
    
    # 4. Top antagonistic combinations
    ax4 = axes[1, 1]
    bottom_10 = sorted_synergies[-10:]
    labels = [str(list(k)).replace(' ', '') for k, v in bottom_10]
    synergy_values = [v['synergy'] * 100 for k, v in bottom_10]
    bars = ax4.barh(range(len(labels)), synergy_values, alpha=0.7, color='orange')
    ax4.set_yticks(range(len(labels)))
    ax4.set_yticklabels(labels, fontsize=9)
    ax4.set_xlabel('Synergy Effect (%)', fontsize=11)
    ax4.set_title('Top 10 Antagonistic Combinations', fontsize=12, fontweight='bold')
    ax4.axvline(0, color='r', linestyle='--', linewidth=1)
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combination_vs_individual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved combination_vs_individual.png")

def analyze_cross_dataset_consistency(results: Dict) -> Dict:
    """Analyze consistency across datasets."""
    # For each combination, calculate coefficient of variation
    consistency_stats = []
    
    for key, stats in results['combination_stats'].items():
        if stats['num_datasets_tested'] < 2 or stats['avg_accuracy_drop'] < 0:
            continue
        
        avg_drop = stats['avg_accuracy_drop']
        std_drop = stats['std_accuracy_drop']
        
        if avg_drop > 0:
            cv = std_drop / avg_drop  # Coefficient of variation
        else:
            cv = float('inf')
        
        consistency_stats.append({
            'removed_blocks': stats['removed_blocks'],
            'num_removed': stats['num_removed'],
            'avg_drop': avg_drop,
            'std_drop': std_drop,
            'cv': cv,
            'num_datasets': stats['num_datasets_tested'],
        })
    
    return consistency_stats

def plot_consistency_analysis(consistency_stats: List[Dict], output_dir: Path):
    """Plot cross-dataset consistency analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Dataset Consistency Analysis', fontsize=16, fontweight='bold')
    
    # Filter out infinite CV
    valid_stats = [s for s in consistency_stats if s['cv'] != float('inf')]
    
    # 1. CV distribution
    ax1 = axes[0, 0]
    cvs = [s['cv'] for s in valid_stats]
    ax1.hist(cvs, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Coefficient of Variation (CV)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Consistency Distribution (Lower CV = More Consistent)', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # 2. CV vs Number of datasets
    ax2 = axes[0, 1]
    num_datasets = [s['num_datasets'] for s in valid_stats]
    ax2.scatter(num_datasets, cvs, alpha=0.5, s=30)
    ax2.set_xlabel('Number of Datasets Tested', fontsize=11)
    ax2.set_ylabel('Coefficient of Variation', fontsize=11)
    ax2.set_title('Consistency vs Dataset Coverage', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Most consistent combinations (low CV)
    ax3 = axes[1, 0]
    sorted_by_cv = sorted(valid_stats, key=lambda x: x['cv'])[:15]
    labels = [str(s['removed_blocks']).replace(' ', '') for s in sorted_by_cv]
    cv_values = [s['cv'] for s in sorted_by_cv]
    bars = ax3.barh(range(len(labels)), cv_values, alpha=0.7)
    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.set_xlabel('Coefficient of Variation', fontsize=11)
    ax3.set_title('Most Consistent Combinations (Top 15)', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_yaxis()
    
    # 4. CV by number of removed blocks
    ax4 = axes[1, 1]
    by_num_removed = defaultdict(list)
    for s in valid_stats:
        by_num_removed[s['num_removed']].append(s['cv'])
    
    num_removed_list = sorted(by_num_removed.keys())
    cv_means = [np.mean(by_num_removed[n]) for n in num_removed_list]
    cv_stds = [np.std(by_num_removed[n]) for n in num_removed_list]
    
    bars = ax4.bar(num_removed_list, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color='green')
    ax4.set_xlabel('Number of Removed Blocks', fontsize=11)
    ax4.set_ylabel('Mean Coefficient of Variation', fontsize=11)
    ax4.set_title('Consistency by Number of Removed Blocks', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'consistency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved consistency_analysis.png")

def analyze_dataset_specific_responses(results: Dict) -> Dict[str, List[float]]:
    """Analyze how each dataset responds to block removals."""
    dataset_responses = defaultdict(list)
    
    for key, stats in results['combination_stats'].items():
        if stats['num_datasets_tested'] < 2:
            continue
        
        for dataset, drop in stats['accuracy_drops_by_dataset'].items():
            if drop >= 0:  # Only positive drops
                dataset_responses[dataset].append(drop)
    
    return dict(dataset_responses)

def plot_dataset_responses(dataset_responses: Dict[str, List[float]], output_dir: Path):
    """Plot dataset-specific response patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset-Specific Response Patterns', fontsize=16, fontweight='bold')
    
    datasets = sorted(dataset_responses.keys())
    
    # 1. Mean drop by dataset
    ax1 = axes[0, 0]
    mean_drops = [np.mean(dataset_responses[d]) * 100 for d in datasets]
    std_drops = [np.std(dataset_responses[d]) * 100 for d in datasets]
    bars = ax1.barh(datasets, mean_drops, xerr=std_drops, capsize=5, alpha=0.7)
    ax1.set_xlabel('Mean Accuracy Drop (%)', fontsize=11)
    ax1.set_title('Average Sensitivity by Dataset', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Drop distribution by dataset (violin plot style)
    ax2 = axes[0, 1]
    data_to_plot = [np.array(dataset_responses[d]) * 100 for d in datasets]
    parts = ax2.violinplot(data_to_plot, positions=range(len(datasets)), showmeans=True)
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Accuracy Drop (%)', fontsize=11)
    ax2.set_title('Drop Distribution by Dataset', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Task type grouping
    ax3 = axes[1, 0]
    task_groups = defaultdict(list)
    for dataset in datasets:
        task_type = TASK_TYPES.get(dataset, 'Unknown')
        task_groups[task_type].extend([d * 100 for d in dataset_responses[dataset]])
    
    task_types = sorted(task_groups.keys())
    task_means = [np.mean(task_groups[t]) for t in task_types]
    task_stds = [np.std(task_groups[t]) for t in task_types]
    bars = ax3.bar(task_types, task_means, yerr=task_stds, capsize=5, alpha=0.7, color='purple')
    ax3.set_ylabel('Mean Accuracy Drop (%)', fontsize=11)
    ax3.set_title('Average Sensitivity by Task Type', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Coefficient of variation by dataset
    ax4 = axes[1, 1]
    cvs = []
    cv_labels = []
    for dataset in datasets:
        drops = np.array(dataset_responses[dataset])
        if np.mean(drops) > 0:
            cv = np.std(drops) / np.mean(drops)
            cvs.append(cv)
            cv_labels.append(dataset)
    
    bars = ax4.barh(cv_labels, cvs, alpha=0.7, color='orange')
    ax4.set_xlabel('Coefficient of Variation', fontsize=11)
    ax4.set_title('Response Variability by Dataset', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_responses.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved dataset_responses.png")

def plot_accuracy_drop_distributions(results: Dict, output_dir: Path):
    """Plot accuracy drop distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Accuracy Drop Distributions', fontsize=16, fontweight='bold')
    
    # Collect all drops
    all_drops = []
    drops_by_num_removed = defaultdict(list)
    
    for key, stats in results['combination_stats'].items():
        if stats['num_datasets_tested'] >= 2 and stats['avg_accuracy_drop'] >= 0:
            drop = stats['avg_accuracy_drop'] * 100
            all_drops.append(drop)
            drops_by_num_removed[stats['num_removed']].append(drop)
    
    # 1. Overall distribution
    ax1 = axes[0, 0]
    ax1.hist(all_drops, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(all_drops), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_drops):.2f}%')
    ax1.axvline(np.median(all_drops), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(all_drops):.2f}%')
    ax1.set_xlabel('Accuracy Drop (%)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Overall Drop Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Distribution by number of removed blocks
    ax2 = axes[0, 1]
    for num_removed in sorted(drops_by_num_removed.keys()):
        drops = drops_by_num_removed[num_removed]
        ax2.hist(drops, bins=30, alpha=0.5, label=f'{num_removed} block(s)', density=True)
    ax2.set_xlabel('Accuracy Drop (%)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Drop Distribution by Number Removed', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Box plot by number of removed blocks
    ax3 = axes[1, 0]
    data_to_plot = [drops_by_num_removed[n] for n in sorted(drops_by_num_removed.keys())]
    bp = ax3.boxplot(data_to_plot, labels=[f'{n} block(s)' for n in sorted(drops_by_num_removed.keys())])
    ax3.set_ylabel('Accuracy Drop (%)', fontsize=11)
    ax3.set_title('Drop Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Cumulative distribution
    ax4 = axes[1, 1]
    sorted_drops = np.sort(all_drops)
    cumulative = np.arange(1, len(sorted_drops) + 1) / len(sorted_drops) * 100
    ax4.plot(sorted_drops, cumulative, linewidth=2)
    ax4.axvline(np.percentile(all_drops, 50), color='r', linestyle='--', alpha=0.7, label='50th percentile')
    ax4.axvline(np.percentile(all_drops, 90), color='g', linestyle='--', alpha=0.7, label='90th percentile')
    ax4.set_xlabel('Accuracy Drop (%)', fontsize=11)
    ax4.set_ylabel('Cumulative Percentage', fontsize=11)
    ax4.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'drop_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved drop_distributions.png")

def main():
    """Main execution."""
    analysis_file = Path("results/profiling/exp3_beam_search_multi_dataset/analysis/cross_dataset_analysis.json")
    output_dir = Path("results/profiling/exp3_beam_search_multi_dataset/analysis")
    
    if not analysis_file.exists():
        print(f"Error: Analysis file not found: {analysis_file}")
        print("Please run analyze_multi_dataset_beam_search.py first")
        sys.exit(1)
    
    print("Loading analysis results...")
    results = load_analysis_results(analysis_file)
    
    print("Generating comprehensive visualizations...")
    
    # 1. Rankings by number of removed blocks
    print("\n1. Generating rankings by number of removed blocks...")
    by_num_removed = analyze_by_num_removed(results)
    plot_rankings_by_num_removed(by_num_removed, output_dir, top_k=15)
    
    # 2. Task type comparison
    print("\n2. Analyzing task type differences...")
    task_stats = analyze_by_task_type(results)
    plot_task_type_comparison(task_stats, output_dir)
    
    # 3. Block position analysis
    print("\n3. Analyzing block position effects...")
    position_stats = analyze_by_block_position(results)
    plot_block_position_analysis(position_stats, output_dir)
    
    # 4. Combination vs individual
    print("\n4. Analyzing combination effects...")
    combo_analysis = analyze_combination_vs_individual(results)
    plot_combination_vs_individual(combo_analysis, output_dir)
    
    # 5. Consistency analysis
    print("\n5. Analyzing cross-dataset consistency...")
    consistency_stats = analyze_cross_dataset_consistency(results)
    plot_consistency_analysis(consistency_stats, output_dir)
    
    # 6. Dataset-specific responses
    print("\n6. Analyzing dataset-specific responses...")
    dataset_responses = analyze_dataset_specific_responses(results)
    plot_dataset_responses(dataset_responses, output_dir)
    
    # 7. Drop distributions
    print("\n7. Analyzing drop distributions...")
    plot_accuracy_drop_distributions(results, output_dir)
    
    print("\n" + "="*80)
    print("✅ All visualizations generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()

