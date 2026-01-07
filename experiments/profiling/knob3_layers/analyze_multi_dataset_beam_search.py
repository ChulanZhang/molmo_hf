#!/usr/bin/env python3
"""
Analyze beam search results across multiple datasets.
Find optimal block combinations that work well across all datasets.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np

def load_all_results(base_dir: Path) -> Dict[str, Dict]:
    """Load results from all datasets.
    
    Returns:
        dict: {dataset_name: results_dict}
    """
    datasets = [
        ("coco-2014-vqa", "train+validation"),
        ("text-vqa", "train+validation"),
        ("okvqa", "train+validation"),
        ("science-qa-img", "train+validation"),
        ("st-qa", "train+validation"),
        ("doc-qa", "train+validation"),
        ("tally-qa", "train+test"),
        ("coco-caption", "train+validation"),
    ]
    
    all_results = {}
    
    for dataset_name, split in datasets:
        results_file = base_dir / dataset_name / split / "exp3_accuracy_sensitivity_v2_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    all_results[dataset_name] = results
                    print(f"✅ Loaded {dataset_name}: {len(results.get('summary', []))} configurations")
            except Exception as e:
                print(f"⚠️  Failed to load {dataset_name}: {e}")
        else:
            print(f"⚠️  Results file not found: {results_file}")
    
    return all_results


def analyze_cross_dataset_results(all_results: Dict[str, Dict]) -> Dict[str, Any]:
    """Analyze results across all datasets.
    
    Returns:
        dict: Analysis results including:
            - average_accuracy_drops: Average accuracy drop for each block combination
            - best_combinations: Top combinations by average accuracy drop
            - dataset_specific_best: Best combination for each dataset
            - consistency_analysis: How consistent results are across datasets
    """
    # Group results by block combination (removed blocks)
    combination_stats = defaultdict(lambda: {
        'accuracy_drops': [],
        'accuracies': [],
        'datasets': [],
        'steps': [],
    })
    
    # Process each dataset
    for dataset_name, results in all_results.items():
        summary = results.get('summary', [])
        baseline = results.get('baseline_accuracy', 0.0)
        
        for config in summary:
            removed_blocks = tuple(sorted(config.get('removed_block_indices', [])))
            accuracy_drop = config.get('accuracy_drop', 0.0)
            accuracy = config.get('accuracy', 0.0)
            step = config.get('step', 0)
            
            combination_stats[removed_blocks]['accuracy_drops'].append(accuracy_drop)
            combination_stats[removed_blocks]['accuracies'].append(accuracy)
            combination_stats[removed_blocks]['datasets'].append(dataset_name)
            combination_stats[removed_blocks]['steps'].append(step)
    
    # Calculate statistics for each combination
    analysis_results = {
        'combination_stats': {},
        'best_combinations': [],
        'dataset_specific_best': {},
        'summary_stats': {},
    }
    
    for removed_blocks, stats in combination_stats.items():
        if len(stats['accuracy_drops']) == 0:
            continue
        
        avg_drop = np.mean(stats['accuracy_drops'])
        std_drop = np.std(stats['accuracy_drops'])
        avg_accuracy = np.mean(stats['accuracies'])
        num_datasets = len(set(stats['datasets']))
        
        analysis_results['combination_stats'][str(removed_blocks)] = {
            'removed_blocks': list(removed_blocks),
            'num_removed': len(removed_blocks),
            'avg_accuracy_drop': float(avg_drop),
            'std_accuracy_drop': float(std_drop),
            'avg_accuracy': float(avg_accuracy),
            'num_datasets_tested': num_datasets,
            'datasets': list(set(stats['datasets'])),
            'accuracy_drops_by_dataset': {
                dataset: drop for dataset, drop in zip(stats['datasets'], stats['accuracy_drops'])
            },
        }
    
    # Find best combinations (lowest average accuracy drop)
    # Filter: only consider combinations tested on at least 2 datasets
    # and with non-negative accuracy drop (negative means accuracy increased, likely measurement error)
    filtered_combinations = [
        (key, stats) for key, stats in analysis_results['combination_stats'].items()
        if stats['num_datasets_tested'] >= 2 and stats['avg_accuracy_drop'] >= 0
    ]
    
    sorted_combinations = sorted(
        filtered_combinations,
        key=lambda x: x[1]['avg_accuracy_drop']
    )
    
    # Top 20 combinations
    analysis_results['best_combinations'] = [
        {
            'rank': i + 1,
            **stats
        }
        for i, (_, stats) in enumerate(sorted_combinations[:20])
    ]
    
    # Also keep all combinations (for reference)
    analysis_results['all_combinations'] = [
        {
            'rank': i + 1,
            **stats
        }
        for i, (_, stats) in enumerate(sorted(
            analysis_results['combination_stats'].items(),
            key=lambda x: x[1]['avg_accuracy_drop']
        )[:50])
    ]
    
    # Find best combination for each dataset
    for dataset_name, results in all_results.items():
        summary = results.get('summary', [])
        if not summary:
            continue
        
        # Find configuration with lowest accuracy drop
        best_config = min(summary, key=lambda x: x.get('accuracy_drop', float('inf')))
        analysis_results['dataset_specific_best'][dataset_name] = {
            'removed_blocks': best_config.get('removed_block_indices', []),
            'accuracy_drop': best_config.get('accuracy_drop', 0.0),
            'accuracy': best_config.get('accuracy', 0.0),
            'step': best_config.get('step', 0),
        }
    
    # Summary statistics
    all_drops = [stats['avg_accuracy_drop'] for stats in analysis_results['combination_stats'].values()]
    analysis_results['summary_stats'] = {
        'total_unique_combinations': len(analysis_results['combination_stats']),
        'avg_accuracy_drop_mean': float(np.mean(all_drops)) if all_drops else 0.0,
        'avg_accuracy_drop_std': float(np.std(all_drops)) if all_drops else 0.0,
        'avg_accuracy_drop_min': float(np.min(all_drops)) if all_drops else 0.0,
        'avg_accuracy_drop_max': float(np.max(all_drops)) if all_drops else 0.0,
        'num_datasets': len(all_results),
    }
    
    return analysis_results


def find_consensus_combinations(analysis_results: Dict[str, Any], top_k: int = 10) -> List[Dict]:
    """Find combinations that appear in top-K of multiple datasets.
    
    Args:
        analysis_results: Results from analyze_cross_dataset_results
        top_k: Number of top combinations to consider per dataset
    
    Returns:
        list: Consensus combinations with their statistics
    """
    # For each dataset, find top-K combinations
    dataset_top_combinations = {}
    
    for dataset_name in analysis_results['dataset_specific_best'].keys():
        # We need to load individual dataset results to find top-K
        # For now, we'll use the best_combinations which are already sorted
        pass
    
    # Count how many datasets have each combination in their top-K
    combination_counts = defaultdict(int)
    combination_details = defaultdict(lambda: {
        'datasets': [],
        'drops': [],
    })
    
    # This is a simplified version - in practice, we'd need to check each dataset's top-K
    # For now, we'll use the best_combinations which are sorted by average drop
    consensus = []
    
    for combo in analysis_results['best_combinations'][:top_k]:
        removed_blocks = tuple(combo['removed_blocks'])
        datasets = combo['datasets']
        
        consensus.append({
            'removed_blocks': list(removed_blocks),
            'num_removed': combo['num_removed'],
            'avg_accuracy_drop': combo['avg_accuracy_drop'],
            'std_accuracy_drop': combo['std_accuracy_drop'],
            'num_datasets': len(datasets),
            'datasets': datasets,
        })
    
    return consensus


def generate_report(analysis_results: Dict[str, Any], output_file: Path):
    """Generate a detailed report."""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("Cross-Dataset Beam Search Analysis Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary statistics
    summary = analysis_results['summary_stats']
    report_lines.append("Summary Statistics:")
    report_lines.append(f"  Total unique block combinations tested: {summary['total_unique_combinations']}")
    report_lines.append(f"  Number of datasets: {summary['num_datasets']}")
    report_lines.append(f"  Average accuracy drop (mean): {summary['avg_accuracy_drop_mean']:.4f}")
    report_lines.append(f"  Average accuracy drop (std): {summary['avg_accuracy_drop_std']:.4f}")
    report_lines.append(f"  Min accuracy drop: {summary['avg_accuracy_drop_min']:.4f}")
    report_lines.append(f"  Max accuracy drop: {summary['avg_accuracy_drop_max']:.4f}")
    report_lines.append("")
    
    # Best combinations overall (filtered: at least 2 datasets, non-negative drop)
    report_lines.append("Top 10 Best Block Combinations (Lowest Average Accuracy Drop):")
    report_lines.append("(Filtered: tested on ≥2 datasets, accuracy drop ≥0)")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Rank':<6} {'Removed Blocks':<25} {'Avg Drop':<12} {'Std Drop':<12} {'# Datasets':<12}")
    report_lines.append("-" * 80)
    
    if analysis_results['best_combinations']:
        for combo in analysis_results['best_combinations'][:10]:
            removed_str = str(combo['removed_blocks']).replace(' ', '')
            report_lines.append(
                f"{combo['rank']:<6} {removed_str:<25} {combo['avg_accuracy_drop']:<12.4f} "
                f"{combo['std_accuracy_drop']:<12.4f} {combo['num_datasets_tested']:<12}"
            )
    else:
        report_lines.append("No combinations meet the filter criteria (≥2 datasets, drop ≥0)")
    report_lines.append("")
    
    # Best single-block removals (for comparison)
    single_block_combos = [
        (key, stats) for key, stats in analysis_results['combination_stats'].items()
        if stats['num_removed'] == 1 and stats['num_datasets_tested'] >= 2
    ]
    single_block_combos.sort(key=lambda x: x[1]['avg_accuracy_drop'])
    
    if single_block_combos:
        report_lines.append("Best Single-Block Removals (for comparison):")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Block':<8} {'Avg Drop':<12} {'Std Drop':<12} {'# Datasets':<12}")
        report_lines.append("-" * 80)
        for key, stats in single_block_combos[:10]:
            block = stats['removed_blocks'][0]
            report_lines.append(
                f"Block {block:<4} {stats['avg_accuracy_drop']:<12.4f} "
                f"{stats['std_accuracy_drop']:<12.4f} {stats['num_datasets_tested']:<12}"
            )
        report_lines.append("")
    
    # Dataset-specific best
    report_lines.append("Best Combination for Each Dataset:")
    report_lines.append("-" * 80)
    for dataset_name, best in analysis_results['dataset_specific_best'].items():
        removed_str = str(best['removed_blocks']).replace(' ', '')
        report_lines.append(
            f"{dataset_name:<20} Removed: {removed_str:<25} "
            f"Drop: {best['accuracy_drop']:.4f} Accuracy: {best['accuracy']:.4f}"
        )
    report_lines.append("")
    
    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Report saved to {output_file}")


def main():
    """Main execution"""
    base_dir = Path("results/profiling/exp3_beam_search_multi_dataset")
    
    print("Loading results from all datasets...")
    all_results = load_all_results(base_dir)
    
    if not all_results:
        print("❌ No results found!")
        sys.exit(1)
    
    print(f"\nAnalyzing {len(all_results)} datasets...")
    analysis_results = analyze_cross_dataset_results(all_results)
    
    # Save analysis results
    output_dir = base_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    analysis_file = output_dir / "cross_dataset_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"✅ Analysis saved to {analysis_file}")
    
    # Generate report
    report_file = output_dir / "cross_dataset_report.txt"
    generate_report(analysis_results, report_file)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total unique combinations: {analysis_results['summary_stats']['total_unique_combinations']}")
    if analysis_results['best_combinations']:
        best = analysis_results['best_combinations'][0]
        print(f"  Best combination (avg drop): {best['avg_accuracy_drop']:.4f}")
        print(f"    Removed blocks: {best['removed_blocks']}")
        print(f"    Tested on {best['num_datasets_tested']} datasets")
        print(f"    Datasets: {', '.join(best['datasets'])}")
    else:
        print("  No combinations meet filter criteria (≥2 datasets, drop ≥0)")
    print("=" * 80)


if __name__ == "__main__":
    main()

