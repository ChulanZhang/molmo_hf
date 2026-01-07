#!/usr/bin/env python3
"""
Find best block combinations for removing 1, 2, 3, and 4 blocks.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

def find_best_by_num_removed(analysis_file: Path) -> Dict[int, List[Dict]]:
    """Find best combinations for each number of removed blocks.
    
    Returns:
        dict: {num_removed: [best_combinations]}
    """
    with open(analysis_file, 'r') as f:
        analysis_results = json.load(f)
    
    # Group by number of removed blocks
    by_num_removed = {1: [], 2: [], 3: [], 4: []}
    
    for key, stats in analysis_results['combination_stats'].items():
        num_removed = stats['num_removed']
        if num_removed in by_num_removed:
            # Only include combinations tested on at least 2 datasets and with non-negative drop
            if stats['num_datasets_tested'] >= 2 and stats['avg_accuracy_drop'] >= 0:
                by_num_removed[num_removed].append(stats)
    
    # Sort each group by average accuracy drop
    for num_removed in by_num_removed:
        by_num_removed[num_removed].sort(key=lambda x: x['avg_accuracy_drop'])
    
    return by_num_removed


def print_best_combinations(by_num_removed: Dict[int, List[Dict]], top_k: int = 5):
    """Print best combinations for each number of removed blocks."""
    print("=" * 80)
    print("Best Block Combinations by Number of Removed Blocks")
    print("=" * 80)
    print()
    
    for num_removed in [1, 2, 3, 4]:
        combinations = by_num_removed[num_removed]
        
        if not combinations:
            print(f"❌ No combinations found for removing {num_removed} block(s)")
            print("   (Need at least 2 datasets tested, accuracy drop ≥0)")
            print()
            continue
        
        print(f"🔹 Removing {num_removed} Block(s) - Top {min(top_k, len(combinations))} Combinations:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Removed Blocks':<30} {'Avg Drop':<12} {'Std Drop':<12} {'# Datasets':<12}")
        print("-" * 80)
        
        for i, combo in enumerate(combinations[:top_k]):
            removed_str = str(combo['removed_blocks']).replace(' ', '')
            print(
                f"{i+1:<6} {removed_str:<30} {combo['avg_accuracy_drop']:<12.4f} "
                f"{combo['std_accuracy_drop']:<12.4f} {combo['num_datasets_tested']:<12}"
            )
        
        # Show datasets for the best one
        if combinations:
            best = combinations[0]
            print(f"\n   Best: Remove {best['removed_blocks']}")
            print(f"   Average accuracy drop: {best['avg_accuracy_drop']:.4f} ± {best['std_accuracy_drop']:.4f}")
            print(f"   Tested on {best['num_datasets_tested']} dataset(s): {', '.join(best['datasets'])}")
            print(f"   Average accuracy: {best['avg_accuracy']:.4f}")
        
        print()
        print()


def generate_recommendations(by_num_removed: Dict[int, List[Dict]]) -> Dict[int, Dict]:
    """Generate step-by-step recommendations."""
    recommendations = {}
    
    # Step 1: Remove 1 block
    if by_num_removed[1]:
        best_1 = by_num_removed[1][0]
        recommendations[1] = {
            'removed_blocks': best_1['removed_blocks'],
            'avg_drop': best_1['avg_accuracy_drop'],
            'std_drop': best_1['std_accuracy_drop'],
            'num_datasets': best_1['num_datasets_tested'],
            'datasets': best_1['datasets'],
        }
    
    # Step 2: Remove 2 blocks (starting from best 1-block removal)
    if by_num_removed[2] and 1 in recommendations:
        # Find 2-block combinations that include the best 1-block removal
        best_1_blocks = set(recommendations[1]['removed_blocks'])
        candidates = [
            combo for combo in by_num_removed[2]
            if set(combo['removed_blocks']).issuperset(best_1_blocks)
        ]
        
        if candidates:
            candidates.sort(key=lambda x: x['avg_accuracy_drop'])
            best_2 = candidates[0]
            recommendations[2] = {
                'removed_blocks': best_2['removed_blocks'],
                'avg_drop': best_2['avg_accuracy_drop'],
                'std_drop': best_2['std_accuracy_drop'],
                'num_datasets': best_2['num_datasets_tested'],
                'datasets': best_2['datasets'],
                'incremental_drop': best_2['avg_accuracy_drop'] - recommendations[1]['avg_drop'],
            }
        else:
            # If no combination includes the best 1-block, just use the best 2-block overall
            best_2 = by_num_removed[2][0]
            recommendations[2] = {
                'removed_blocks': best_2['removed_blocks'],
                'avg_drop': best_2['avg_accuracy_drop'],
                'std_drop': best_2['std_accuracy_drop'],
                'num_datasets': best_2['num_datasets_tested'],
                'datasets': best_2['datasets'],
                'incremental_drop': best_2['avg_accuracy_drop'] - recommendations[1]['avg_drop'] if 1 in recommendations else None,
            }
    
    # Step 3: Remove 3 blocks
    if by_num_removed[3] and 2 in recommendations:
        best_2_blocks = set(recommendations[2]['removed_blocks'])
        candidates = [
            combo for combo in by_num_removed[3]
            if set(combo['removed_blocks']).issuperset(best_2_blocks)
        ]
        
        if candidates:
            candidates.sort(key=lambda x: x['avg_accuracy_drop'])
            best_3 = candidates[0]
            recommendations[3] = {
                'removed_blocks': best_3['removed_blocks'],
                'avg_drop': best_3['avg_accuracy_drop'],
                'std_drop': best_3['std_accuracy_drop'],
                'num_datasets': best_3['num_datasets_tested'],
                'datasets': best_3['datasets'],
                'incremental_drop': best_3['avg_accuracy_drop'] - recommendations[2]['avg_drop'],
            }
        else:
            best_3 = by_num_removed[3][0]
            recommendations[3] = {
                'removed_blocks': best_3['removed_blocks'],
                'avg_drop': best_3['avg_accuracy_drop'],
                'std_drop': best_3['std_accuracy_drop'],
                'num_datasets': best_3['num_datasets_tested'],
                'datasets': best_3['datasets'],
                'incremental_drop': best_3['avg_accuracy_drop'] - recommendations[2]['avg_drop'] if 2 in recommendations else None,
            }
    
    # Step 4: Remove 4 blocks
    if by_num_removed[4] and 3 in recommendations:
        best_3_blocks = set(recommendations[3]['removed_blocks'])
        candidates = [
            combo for combo in by_num_removed[4]
            if set(combo['removed_blocks']).issuperset(best_3_blocks)
        ]
        
        if candidates:
            candidates.sort(key=lambda x: x['avg_accuracy_drop'])
            best_4 = candidates[0]
            recommendations[4] = {
                'removed_blocks': best_4['removed_blocks'],
                'avg_drop': best_4['avg_accuracy_drop'],
                'std_drop': best_4['std_accuracy_drop'],
                'num_datasets': best_4['num_datasets_tested'],
                'datasets': best_4['datasets'],
                'incremental_drop': best_4['avg_accuracy_drop'] - recommendations[3]['avg_drop'],
            }
        else:
            best_4 = by_num_removed[4][0]
            recommendations[4] = {
                'removed_blocks': best_4['removed_blocks'],
                'avg_drop': best_4['avg_accuracy_drop'],
                'std_drop': best_4['std_accuracy_drop'],
                'num_datasets': best_4['num_datasets_tested'],
                'datasets': best_4['datasets'],
                'incremental_drop': best_4['avg_accuracy_drop'] - recommendations[3]['avg_drop'] if 3 in recommendations else None,
            }
    
    return recommendations


def print_recommendations(recommendations: Dict[int, Dict]):
    """Print step-by-step recommendations."""
    print("=" * 80)
    print("Step-by-Step Pruning Recommendations")
    print("=" * 80)
    print()
    print("This shows the best choices for removing blocks incrementally.")
    print("Each step builds on the previous one (greedy approach).")
    print()
    
    for num_removed in [1, 2, 3, 4]:
        if num_removed not in recommendations:
            print(f"❌ Step {num_removed}: No recommendation available")
            print()
            continue
        
        rec = recommendations[num_removed]
        print(f"Step {num_removed}: Remove {rec['removed_blocks']}")
        print(f"  Average accuracy drop: {rec['avg_drop']:.4f} ± {rec['std_drop']:.4f}")
        if 'incremental_drop' in rec and rec['incremental_drop'] is not None:
            print(f"  Incremental drop (from step {num_removed-1}): {rec['incremental_drop']:.4f}")
        print(f"  Tested on {rec['num_datasets']} dataset(s): {', '.join(rec['datasets'])}")
        print()


def main():
    """Main execution"""
    analysis_file = Path("results/profiling/exp3_beam_search_multi_dataset/analysis/cross_dataset_analysis.json")
    
    if not analysis_file.exists():
        print(f"Error: Analysis file not found: {analysis_file}")
        print("Please run analyze_multi_dataset_beam_search.py first")
        sys.exit(1)
    
    print("Analyzing best combinations by number of removed blocks...")
    by_num_removed = find_best_by_num_removed(analysis_file)
    
    print_best_combinations(by_num_removed, top_k=5)
    
    print_recommendations(generate_recommendations(by_num_removed))
    
    # Save recommendations to file
    recommendations = generate_recommendations(by_num_removed)
    output_file = analysis_file.parent / "pruning_recommendations.json"
    with open(output_file, 'w') as f:
        json.dump(recommendations, f, indent=2)
    print(f"✅ Recommendations saved to {output_file}")


if __name__ == "__main__":
    main()


