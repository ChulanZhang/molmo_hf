#!/usr/bin/env python3
"""
Detailed analysis of differences between datasets.
Shows how different datasets respond to the same block removals.
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
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

def load_analysis_results(analysis_file: Path) -> Dict:
    """Load cross-dataset analysis results."""
    with open(analysis_file, 'r') as f:
        return json.load(f)

def analyze_dataset_pairwise_differences(results: Dict) -> Dict:
    """Analyze pairwise differences between datasets."""
    # Get all combinations tested on multiple datasets
    multi_dataset_combos = []
    for key, stats in results['combination_stats'].items():
        if stats['num_datasets_tested'] >= 2:
            multi_dataset_combos.append(stats)
    
    # Calculate pairwise correlations and differences
    datasets = sorted(TASK_TYPES.keys())
    pairwise_stats = {}
    
    for i, dataset1 in enumerate(datasets):
        for dataset2 in datasets[i+1:]:
            pair_key = f"{dataset1} vs {dataset2}"
            drops1 = []
            drops2 = []
            
            for combo in multi_dataset_combos:
                if dataset1 in combo['accuracy_drops_by_dataset'] and \
                   dataset2 in combo['accuracy_drops_by_dataset']:
                    drop1 = combo['accuracy_drops_by_dataset'][dataset1]
                    drop2 = combo['accuracy_drops_by_dataset'][dataset2]
                    if drop1 >= 0 and drop2 >= 0:
                        drops1.append(drop1)
                        drops2.append(drop2)
            
            if len(drops1) >= 5:  # Need at least 5 common combinations
                correlation = np.corrcoef(drops1, drops2)[0, 1]
                mean_diff = np.mean(np.array(drops1) - np.array(drops2))
                std_diff = np.std(np.array(drops1) - np.array(drops2))
                
                pairwise_stats[pair_key] = {
                    'correlation': correlation,
                    'mean_diff': mean_diff,
                    'std_diff': std_diff,
                    'num_common': len(drops1),
                    'dataset1': dataset1,
                    'dataset2': dataset2,
                    'task1': TASK_TYPES.get(dataset1, 'Unknown'),
                    'task2': TASK_TYPES.get(dataset2, 'Unknown'),
                }
    
    return pairwise_stats

def plot_dataset_pairwise_comparison(pairwise_stats: Dict, output_dir: Path):
    """Plot pairwise dataset comparisons."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Pairwise Dataset Comparison Analysis', fontsize=16, fontweight='bold')
    
    pairs = list(pairwise_stats.keys())
    correlations = [pairwise_stats[p]['correlation'] for p in pairs]
    mean_diffs = [pairwise_stats[p]['mean_diff'] * 100 for p in pairs]
    task_same = [pairwise_stats[p]['task1'] == pairwise_stats[p]['task2'] for p in pairs]
    
    # 1. Correlation distribution
    ax1 = axes[0, 0]
    same_task_corr = [c for c, same in zip(correlations, task_same) if same]
    diff_task_corr = [c for c, same in zip(correlations, task_same) if not same]
    
    ax1.hist(same_task_corr, bins=20, alpha=0.6, label='Same Task Type', density=True)
    ax1.hist(diff_task_corr, bins=20, alpha=0.6, label='Different Task Type', density=True)
    ax1.set_xlabel('Correlation Coefficient', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Correlation Distribution by Task Type', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Correlation heatmap (if we have enough pairs)
    ax2 = axes[0, 1]
    datasets = sorted(set([pairwise_stats[p]['dataset1'] for p in pairs] + 
                          [pairwise_stats[p]['dataset2'] for p in pairs]))
    
    if len(datasets) <= 10:  # Only if manageable number
        corr_matrix = np.ones((len(datasets), len(datasets)))
        for i, d1 in enumerate(datasets):
            for j, d2 in enumerate(datasets):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    pair_key = f"{d1} vs {d2}" if d1 < d2 else f"{d2} vs {d1}"
                    if pair_key in pairwise_stats:
                        corr_matrix[i, j] = pairwise_stats[pair_key]['correlation']
                    else:
                        corr_matrix[i, j] = np.nan
        
        im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax2.set_xticks(range(len(datasets)))
        ax2.set_yticks(range(len(datasets)))
        ax2.set_xticklabels([d.replace('-', '\n') for d in datasets], fontsize=8, rotation=45, ha='right')
        ax2.set_yticklabels([d.replace('-', '\n') for d in datasets], fontsize=8)
        ax2.set_title('Pairwise Correlation Matrix', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax2)
    else:
        ax2.text(0.5, 0.5, 'Too many datasets\nfor correlation matrix', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Pairwise Correlation Matrix', fontsize=12, fontweight='bold')
    
    # 3. Mean difference by task type
    ax3 = axes[1, 0]
    task_pairs = defaultdict(list)
    for p in pairs:
        task_pair = tuple(sorted([pairwise_stats[p]['task1'], pairwise_stats[p]['task2']]))
        task_pairs[task_pair].append(pairwise_stats[p]['mean_diff'] * 100)
    
    task_pair_labels = [f"{t1} vs {t2}" if t1 != t2 else t1 for (t1, t2) in sorted(task_pairs.keys())]
    task_pair_means = [np.mean(task_pairs[k]) for k in sorted(task_pairs.keys())]
    task_pair_stds = [np.std(task_pairs[k]) for k in sorted(task_pairs.keys())]
    
    bars = ax3.barh(task_pair_labels, task_pair_means, xerr=task_pair_stds, capsize=5, alpha=0.7)
    ax3.set_xlabel('Mean Difference in Drop (%)', fontsize=11)
    ax3.set_title('Mean Difference by Task Type Pair', fontsize=12, fontweight='bold')
    ax3.axvline(0, color='r', linestyle='--', linewidth=1)
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Top correlated and least correlated pairs
    ax4 = axes[1, 1]
    sorted_pairs = sorted(pairs, key=lambda p: pairwise_stats[p]['correlation'], reverse=True)
    top_5 = sorted_pairs[:5]
    bottom_5 = sorted_pairs[-5:]
    
    labels = [f"{pairwise_stats[p]['dataset1'][:8]} vs\n{pairwise_stats[p]['dataset2'][:8]}" 
              for p in top_5 + bottom_5]
    corrs = [pairwise_stats[p]['correlation'] for p in top_5 + bottom_5]
    colors = ['green'] * 5 + ['red'] * 5
    
    bars = ax4.barh(range(len(labels)), corrs, color=colors, alpha=0.7)
    ax4.set_yticks(range(len(labels)))
    ax4.set_yticklabels(labels, fontsize=8)
    ax4.set_xlabel('Correlation Coefficient', fontsize=11)
    ax4.set_title('Top 5 Correlated (Green) vs Least Correlated (Red)', fontsize=12, fontweight='bold')
    ax4.axvline(0, color='black', linestyle='--', linewidth=1)
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_pairwise_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved dataset_pairwise_comparison.png")

def analyze_task_type_differences_detailed(results: Dict) -> Dict:
    """Detailed analysis of task type differences."""
    task_combo_drops = defaultdict(lambda: defaultdict(list))
    
    for key, stats in results['combination_stats'].items():
        if stats['num_datasets_tested'] < 2:
            continue
        
        for dataset, drop in stats['accuracy_drops_by_dataset'].items():
            if dataset in TASK_TYPES and drop >= 0:
                task_type = TASK_TYPES[dataset]
                num_removed = stats['num_removed']
                task_combo_drops[task_type][num_removed].append(drop)
    
    # Calculate statistics
    task_stats = {}
    for task_type, num_removed_dict in task_combo_drops.items():
        task_stats[task_type] = {}
        for num_removed, drops in num_removed_dict.items():
            task_stats[task_type][num_removed] = {
                'mean': np.mean(drops) * 100,
                'std': np.std(drops) * 100,
                'median': np.median(drops) * 100,
                'count': len(drops),
            }
    
    return task_stats

def plot_task_type_detailed_comparison(task_stats: Dict, output_dir: Path):
    """Plot detailed task type comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Task Type Comparison', fontsize=16, fontweight='bold')
    
    task_types = sorted(task_stats.keys())
    num_removed_list = [1, 2, 3, 4]
    
    # 1. Mean drop by task type and number removed
    ax1 = axes[0, 0]
    x = np.arange(len(task_types))
    width = 0.2
    
    for i, num_removed in enumerate(num_removed_list):
        means = [task_stats[t].get(num_removed, {}).get('mean', 0) for t in task_types]
        ax1.bar(x + i*width, means, width, label=f'{num_removed} block(s)', alpha=0.7)
    
    ax1.set_xlabel('Task Type', fontsize=11)
    ax1.set_ylabel('Mean Drop (%)', fontsize=11)
    ax1.set_title('Mean Drop by Task Type and Number Removed', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(task_types, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Stability (std) by task type
    ax2 = axes[0, 1]
    for i, num_removed in enumerate(num_removed_list):
        stds = [task_stats[t].get(num_removed, {}).get('std', 0) for t in task_types]
        ax2.bar(x + i*width, stds, width, label=f'{num_removed} block(s)', alpha=0.7)
    
    ax2.set_xlabel('Task Type', fontsize=11)
    ax2.set_ylabel('Std Dev (%)', fontsize=11)
    ax2.set_title('Stability by Task Type and Number Removed', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(task_types, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Count of combinations by task type
    ax3 = axes[1, 0]
    for i, num_removed in enumerate(num_removed_list):
        counts = [task_stats[t].get(num_removed, {}).get('count', 0) for t in task_types]
        ax3.bar(x + i*width, counts, width, label=f'{num_removed} block(s)', alpha=0.7)
    
    ax3.set_xlabel('Task Type', fontsize=11)
    ax3.set_ylabel('Number of Combinations', fontsize=11)
    ax3.set_title('Combination Count by Task Type', fontsize=12, fontweight='bold')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(task_types, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Median drop by task type
    ax4 = axes[1, 1]
    for i, num_removed in enumerate(num_removed_list):
        medians = [task_stats[t].get(num_removed, {}).get('median', 0) for t in task_types]
        ax4.bar(x + i*width, medians, width, label=f'{num_removed} block(s)', alpha=0.7)
    
    ax4.set_xlabel('Task Type', fontsize=11)
    ax4.set_ylabel('Median Drop (%)', fontsize=11)
    ax4.set_title('Median Drop by Task Type and Number Removed', fontsize=12, fontweight='bold')
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(task_types, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'task_type_detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved task_type_detailed_comparison.png")

def main():
    """Main execution."""
    analysis_file = Path("results/profiling/exp3_beam_search_multi_dataset/analysis/cross_dataset_analysis.json")
    output_dir = Path("results/profiling/exp3_beam_search_multi_dataset/analysis")
    
    if not analysis_file.exists():
        print(f"Error: Analysis file not found: {analysis_file}")
        sys.exit(1)
    
    print("Loading analysis results...")
    results = load_analysis_results(analysis_file)
    
    print("Analyzing pairwise dataset differences...")
    pairwise_stats = analyze_dataset_pairwise_differences(results)
    plot_dataset_pairwise_comparison(pairwise_stats, output_dir)
    
    print("Analyzing detailed task type differences...")
    task_stats = analyze_task_type_differences_detailed(results)
    plot_task_type_detailed_comparison(task_stats, output_dir)
    
    # Save pairwise stats
    output_file = output_dir / 'pairwise_dataset_stats.json'
    with open(output_file, 'w') as f:
        json.dump(pairwise_stats, f, indent=2)
    print(f"✅ Saved pairwise_dataset_stats.json")
    
    print("\n" + "="*80)
    print("✅ Dataset difference analysis completed!")
    print("="*80)

if __name__ == "__main__":
    main()

