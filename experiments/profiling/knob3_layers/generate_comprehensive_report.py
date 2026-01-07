#!/usr/bin/env python3
"""
Generate comprehensive text report with detailed statistics and insights.
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import numpy as np

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

def analyze_by_num_removed(results: Dict) -> Dict[int, List[Dict]]:
    """Group combinations by number of removed blocks."""
    by_num_removed = {1: [], 2: [], 3: [], 4: []}
    
    for key, stats in results['combination_stats'].items():
        num_removed = stats['num_removed']
        if num_removed in by_num_removed:
            if stats['num_datasets_tested'] >= 2 and stats['avg_accuracy_drop'] >= 0:
                by_num_removed[num_removed].append(stats)
    
    for num_removed in by_num_removed:
        by_num_removed[num_removed].sort(key=lambda x: x['avg_accuracy_drop'])
    
    return by_num_removed

def analyze_task_type_differences(results: Dict) -> Dict[str, Dict]:
    """Analyze differences across task types."""
    task_combo_stats = defaultdict(lambda: {
        'drops': [],
        'stds': [],
        'combinations': [],
    })
    
    for key, stats in results['combination_stats'].items():
        if stats['num_datasets_tested'] < 2 or stats['avg_accuracy_drop'] < 0:
            continue
        
        # Get task types for this combination
        task_types = set()
        for dataset in stats['datasets']:
            if dataset in TASK_TYPES:
                task_types.add(TASK_TYPES[dataset])
        
        for task_type in task_types:
            task_combo_stats[task_type]['drops'].append(stats['avg_accuracy_drop'])
            task_combo_stats[task_type]['stds'].append(stats['std_accuracy_drop'])
            task_combo_stats[task_type]['combinations'].append(stats)
    
    # Calculate statistics
    task_stats = {}
    for task_type, data in task_combo_stats.items():
        task_stats[task_type] = {
            'mean_drop': np.mean(data['drops']) * 100,
            'std_drop': np.std(data['drops']) * 100,
            'mean_std': np.mean(data['stds']) * 100,
            'num_combinations': len(data['combinations']),
            'min_drop': np.min(data['drops']) * 100,
            'max_drop': np.max(data['drops']) * 100,
            'median_drop': np.median(data['drops']) * 100,
        }
    
    return task_stats

def analyze_block_importance_by_task(results: Dict) -> Dict[str, Dict[int, float]]:
    """Analyze individual block importance by task type."""
    block_importance = defaultdict(lambda: defaultdict(list))
    
    for key, stats in results['combination_stats'].items():
        if stats['num_removed'] != 1 or stats['num_datasets_tested'] < 2:
            continue
        
        block = stats['removed_blocks'][0]
        
        for dataset in stats['datasets']:
            if dataset in TASK_TYPES:
                task_type = TASK_TYPES[dataset]
                if dataset in stats['accuracy_drops_by_dataset']:
                    drop = stats['accuracy_drops_by_dataset'][dataset]
                    if drop >= 0:
                        block_importance[task_type][block].append(drop)
    
    # Calculate averages
    block_avg_importance = {}
    for task_type, blocks in block_importance.items():
        block_avg_importance[task_type] = {}
        for block, drops in blocks.items():
            block_avg_importance[task_type][block] = np.mean(drops) * 100
    
    return block_avg_importance

def generate_report(results: Dict, output_file: Path):
    """Generate comprehensive text report."""
    by_num_removed = analyze_by_num_removed(results)
    task_stats = analyze_task_type_differences(results)
    block_importance_by_task = analyze_block_importance_by_task(results)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COMPREHENSIVE BEAM SEARCH ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 1. Executive Summary
    report_lines.append("1. EXECUTIVE SUMMARY")
    report_lines.append("-" * 80)
    total_combinations = sum(len(by_num_removed[n]) for n in [1, 2, 3, 4])
    report_lines.append(f"Total valid combinations analyzed: {total_combinations}")
    report_lines.append(f"  - 1 block removed: {len(by_num_removed[1])}")
    report_lines.append(f"  - 2 blocks removed: {len(by_num_removed[2])}")
    report_lines.append(f"  - 3 blocks removed: {len(by_num_removed[3])}")
    report_lines.append(f"  - 4 blocks removed: {len(by_num_removed[4])}")
    report_lines.append("")
    
    # 2. Rankings by Number of Removed Blocks
    report_lines.append("2. TOP COMBINATIONS BY NUMBER OF REMOVED BLOCKS")
    report_lines.append("-" * 80)
    
    for num_removed in [1, 2, 3, 4]:
        report_lines.append(f"\n2.{num_removed} Removing {num_removed} Block(s) - Top 10:")
        report_lines.append(f"{'Rank':<6} {'Removed Blocks':<25} {'Avg Drop (%)':<15} {'Std (%)':<12} {'# Datasets':<12}")
        report_lines.append("-" * 80)
        
        for i, combo in enumerate(by_num_removed[num_removed][:10]):
            removed_str = str(combo['removed_blocks']).replace(' ', '')
            report_lines.append(
                f"{i+1:<6} {removed_str:<25} {combo['avg_accuracy_drop']*100:<15.4f} "
                f"{combo['std_accuracy_drop']*100:<12.4f} {combo['num_datasets_tested']:<12}"
            )
        report_lines.append("")
    
    # 3. Task Type Analysis
    report_lines.append("\n3. TASK TYPE ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Task Type':<20} {'Mean Drop (%)':<15} {'Std Drop (%)':<15} {'# Combos':<12} {'Min':<10} {'Max':<10} {'Median':<10}")
    report_lines.append("-" * 80)
    
    for task_type in sorted(task_stats.keys()):
        stats = task_stats[task_type]
        report_lines.append(
            f"{task_type:<20} {stats['mean_drop']:<15.4f} {stats['std_drop']:<15.4f} "
            f"{stats['num_combinations']:<12} {stats['min_drop']:<10.4f} {stats['max_drop']:<10.4f} "
            f"{stats['median_drop']:<10.4f}"
        )
    report_lines.append("")
    
    # 4. Block Importance by Task Type
    report_lines.append("\n4. INDIVIDUAL BLOCK IMPORTANCE BY TASK TYPE")
    report_lines.append("-" * 80)
    
    for task_type in sorted(block_importance_by_task.keys()):
        report_lines.append(f"\n4.{task_type}:")
        blocks = block_importance_by_task[task_type]
        sorted_blocks = sorted(blocks.items(), key=lambda x: x[1])
        
        report_lines.append(f"{'Rank':<6} {'Block':<10} {'Avg Drop (%)':<15}")
        report_lines.append("-" * 40)
        for i, (block, drop) in enumerate(sorted_blocks):
            report_lines.append(f"{i+1:<6} {block:<10} {drop:<15.4f}")
        report_lines.append("")
    
    # 5. Key Insights
    report_lines.append("\n5. KEY INSIGHTS")
    report_lines.append("-" * 80)
    
    # Best single block removal
    if by_num_removed[1]:
        best_single = by_num_removed[1][0]
        report_lines.append(f"\n5.1 Best Single Block Removal:")
        report_lines.append(f"   Block: {best_single['removed_blocks']}")
        report_lines.append(f"   Average drop: {best_single['avg_accuracy_drop']*100:.4f}%")
        report_lines.append(f"   Tested on: {best_single['num_datasets_tested']} datasets")
        report_lines.append(f"   Datasets: {', '.join(best_single['datasets'])}")
    
    # Most consistent task type
    if task_stats:
        most_consistent = min(task_stats.items(), key=lambda x: x[1]['mean_std'])
        report_lines.append(f"\n5.2 Most Consistent Task Type:")
        report_lines.append(f"   Task: {most_consistent[0]}")
        report_lines.append(f"   Mean std dev: {most_consistent[1]['mean_std']:.4f}%")
    
    # Most sensitive task type
    if task_stats:
        most_sensitive = max(task_stats.items(), key=lambda x: x[1]['mean_drop'])
        report_lines.append(f"\n5.3 Most Sensitive Task Type:")
        report_lines.append(f"   Task: {most_sensitive[0]}")
        report_lines.append(f"   Mean drop: {most_sensitive[1]['mean_drop']:.4f}%")
    
    # Block position insights
    early_blocks = [0, 1, 2, 3, 4]
    middle_blocks = [5, 6, 7, 8, 9, 10]
    late_blocks = [11, 12, 13, 14, 15]
    
    single_block_drops = {}
    for combo in by_num_removed[1]:
        block = combo['removed_blocks'][0]
        single_block_drops[block] = combo['avg_accuracy_drop']
    
    early_drops = [single_block_drops.get(b, 0) for b in early_blocks if b in single_block_drops]
    middle_drops = [single_block_drops.get(b, 0) for b in middle_blocks if b in single_block_drops]
    late_drops = [single_block_drops.get(b, 0) for b in late_blocks if b in single_block_drops]
    
    if early_drops and middle_drops and late_drops:
        report_lines.append(f"\n5.4 Block Position Analysis:")
        report_lines.append(f"   Early blocks (0-4) avg drop: {np.mean(early_drops)*100:.4f}%")
        report_lines.append(f"   Middle blocks (5-10) avg drop: {np.mean(middle_drops)*100:.4f}%")
        report_lines.append(f"   Late blocks (11-15) avg drop: {np.mean(late_drops)*100:.4f}%")
    
    # 6. Recommendations
    report_lines.append("\n6. RECOMMENDATIONS")
    report_lines.append("-" * 80)
    
    if by_num_removed[1]:
        best = by_num_removed[1][0]
        report_lines.append(f"\n6.1 Conservative Pruning (1 block):")
        report_lines.append(f"   Remove: Block {best['removed_blocks'][0]}")
        report_lines.append(f"   Expected drop: {best['avg_accuracy_drop']*100:.2f}%")
        report_lines.append(f"   Confidence: High (tested on {best['num_datasets_tested']} datasets)")
    
    if len(by_num_removed[2]) >= 3:
        report_lines.append(f"\n6.2 Moderate Pruning (2 blocks):")
        top3 = by_num_removed[2][:3]
        for i, combo in enumerate(top3, 1):
            report_lines.append(f"   Option {i}: Remove {combo['removed_blocks']} "
                              f"(drop: {combo['avg_accuracy_drop']*100:.2f}%, "
                              f"tested on {combo['num_datasets_tested']} datasets)")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Comprehensive report saved to {output_file}")

def main():
    """Main execution."""
    analysis_file = Path("results/profiling/exp3_beam_search_multi_dataset/analysis/cross_dataset_analysis.json")
    output_file = Path("results/profiling/exp3_beam_search_multi_dataset/analysis/comprehensive_report.txt")
    
    if not analysis_file.exists():
        print(f"Error: Analysis file not found: {analysis_file}")
        sys.exit(1)
    
    print("Loading analysis results...")
    results = load_analysis_results(analysis_file)
    
    print("Generating comprehensive report...")
    generate_report(results, output_file)

if __name__ == "__main__":
    main()

