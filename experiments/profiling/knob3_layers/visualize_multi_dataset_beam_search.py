#!/usr/bin/env python3
"""
Visualize cross-dataset beam search analysis results.
"""

import json
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

def plot_best_combinations(analysis_results: Dict, output_dir: Path):
    """Plot top combinations by average accuracy drop."""
    best_combos = analysis_results['best_combinations'][:15]
    
    if not best_combos:
        print("No combinations to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    labels = [f"Remove {str(c['removed_blocks']).replace(' ', '')}" for c in best_combos]
    avg_drops = [c['avg_accuracy_drop'] for c in best_combos]
    std_drops = [c['std_accuracy_drop'] for c in best_combos]
    num_datasets = [c['num_datasets_tested'] for c in best_combos]
    
    # Create bar plot
    x_pos = np.arange(len(labels))
    bars = ax.barh(x_pos, avg_drops, xerr=std_drops, capsize=5, alpha=0.7)
    
    # Color by number of datasets tested
    colors = plt.cm.viridis(np.array(num_datasets) / max(num_datasets))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Labels
    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Average Accuracy Drop', fontsize=12)
    ax.set_title('Top 15 Block Combinations (Lowest Average Accuracy Drop)\n(Filtered: ≥2 datasets, drop ≥0)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Add number of datasets as text
    for i, (drop, num_ds) in enumerate(zip(avg_drops, num_datasets)):
        ax.text(drop + std_drops[i] + 0.005, i, f'({num_ds} datasets)', 
                va='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    output_file = output_dir / "best_combinations.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved {output_file}")
    plt.close()


def plot_single_block_comparison(analysis_results: Dict, output_dir: Path):
    """Plot single-block removal comparison."""
    # Get single-block removals
    single_blocks = [
        (key, stats) for key, stats in analysis_results['combination_stats'].items()
        if stats['num_removed'] == 1 and stats['num_datasets_tested'] >= 2
    ]
    single_blocks.sort(key=lambda x: x[1]['avg_accuracy_drop'])
    
    if not single_blocks:
        print("No single-block removals to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    blocks = [stats['removed_blocks'][0] for _, stats in single_blocks]
    avg_drops = [stats['avg_accuracy_drop'] for _, stats in single_blocks]
    std_drops = [stats['std_accuracy_drop'] for _, stats in single_blocks]
    num_datasets = [stats['num_datasets_tested'] for _, stats in single_blocks]
    
    x_pos = np.arange(len(blocks))
    bars = ax.bar(x_pos, avg_drops, yerr=std_drops, capsize=5, alpha=0.7)
    
    # Color by number of datasets
    colors = plt.cm.viridis(np.array(num_datasets) / max(num_datasets))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Block {b}" for b in blocks], fontsize=10)
    ax.set_ylabel('Average Accuracy Drop', fontsize=12)
    ax.set_title('Single-Block Removal Impact (All Datasets)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add number of datasets
    for i, (drop, num_ds) in enumerate(zip(avg_drops, num_datasets)):
        ax.text(i, drop + std_drops[i] + 0.005, f'({num_ds})', 
                ha='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    output_file = output_dir / "single_block_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved {output_file}")
    plt.close()


def plot_accuracy_drop_distribution(analysis_results: Dict, output_dir: Path):
    """Plot distribution of accuracy drops."""
    all_drops = [
        stats['avg_accuracy_drop'] 
        for stats in analysis_results['combination_stats'].values()
        if stats['num_datasets_tested'] >= 2 and stats['avg_accuracy_drop'] >= 0
    ]
    
    if not all_drops:
        print("No data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(all_drops, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(all_drops), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_drops):.4f}')
    ax.axvline(np.median(all_drops), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_drops):.4f}')
    
    ax.set_xlabel('Average Accuracy Drop', fontsize=12)
    ax.set_ylabel('Number of Combinations', fontsize=12)
    ax.set_title('Distribution of Accuracy Drops (Combinations tested on ≥2 datasets)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "accuracy_drop_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved {output_file}")
    plt.close()


def main():
    """Main execution"""
    analysis_file = Path("results/profiling/exp3_beam_search_multi_dataset/analysis/cross_dataset_analysis.json")
    
    if not analysis_file.exists():
        print(f"Error: Analysis file not found: {analysis_file}")
        print("Please run analyze_multi_dataset_beam_search.py first")
        sys.exit(1)
    
    with open(analysis_file, 'r') as f:
        analysis_results = json.load(f)
    
    output_dir = analysis_file.parent
    output_dir.mkdir(exist_ok=True)
    
    print("Generating visualizations...")
    plot_best_combinations(analysis_results, output_dir)
    plot_single_block_comparison(analysis_results, output_dir)
    plot_accuracy_drop_distribution(analysis_results, output_dir)
    
    print("\n✅ All visualizations saved to", output_dir)


if __name__ == "__main__":
    main()


