#!/usr/bin/env python3
"""
Merge importance scores from train and validation sets into a single ranking.

Given that Spearman correlation is high (0.8374) but not perfect, this script
provides multiple methods to combine train and validation scores into a final
ranking for block pruning.

Usage:
    python experiments/profiling/knob3_layers/merge_importance_scores.py \
        --comparison_file results/profiling/exp3_importance_comparison/coco_2014_vqa/importance_comparison_coco_2014_vqa.json \
        --method weighted_avg \
        --output_file results/layer_importance_scores_merged.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def merge_scores_average(train_scores: Dict[int, float], val_scores: Dict[int, float]) -> Dict[int, float]:
    """Method 1: Simple average of train and validation scores."""
    merged = {}
    for block in range(16):
        merged[block] = (train_scores[block] + val_scores[block]) / 2
    return merged


def merge_scores_weighted_avg(
    train_scores: Dict[int, float], 
    val_scores: Dict[int, float],
    train_weight: float = 0.6
) -> Dict[int, float]:
    """Method 2: Weighted average (default: 60% train, 40% validation).
    
    Rationale: Train set has more samples, so more reliable.
    """
    merged = {}
    val_weight = 1.0 - train_weight
    for block in range(16):
        merged[block] = train_scores[block] * train_weight + val_scores[block] * val_weight
    return merged


def merge_scores_conservative(train_scores: Dict[int, float], val_scores: Dict[int, float]) -> Dict[int, float]:
    """Method 3: Conservative approach - use maximum (worst case impact).
    
    Rationale: Be conservative when there's uncertainty. Use the higher score
    to ensure we don't underestimate block importance.
    """
    merged = {}
    for block in range(16):
        merged[block] = max(train_scores[block], val_scores[block])
    return merged


def merge_scores_median_rank(train_scores: Dict[int, float], val_scores: Dict[int, float]) -> Dict[int, float]:
    """Method 4: Median rank approach.
    
    Rationale: More robust to outliers. Focuses on relative ordering rather
    than absolute values. Converts scores to ranks, then averages ranks,
    then converts back to scores (normalized).
    """
    # Get rankings (1 = least important, 16 = most important)
    train_ranking = sorted(train_scores.items(), key=lambda x: x[1])
    val_ranking = sorted(val_scores.items(), key=lambda x: x[1])
    
    train_ranks = {block: rank for rank, (block, _) in enumerate(train_ranking, start=1)}
    val_ranks = {block: rank for rank, (block, _) in enumerate(val_ranking, start=1)}
    
    # Average ranks
    avg_ranks = {}
    for block in range(16):
        avg_ranks[block] = (train_ranks[block] + val_ranks[block]) / 2
    
    # Convert back to scores (inverse rank, normalized)
    # Lower rank = less important = lower score
    # We want to preserve the relative ordering
    max_rank = 16
    merged = {}
    for block in range(16):
        # Inverse: rank 1 (least important) -> score 0, rank 16 (most important) -> score 1
        # Then scale to match original score range
        normalized_rank = (max_rank - avg_ranks[block] + 1) / max_rank
        # Scale to approximate original score range
        # Use average of train and val max scores as reference
        max_score = max(max(train_scores.values()), max(val_scores.values()))
        merged[block] = normalized_rank * max_score * 0.1  # Scale factor to match typical score range
    
    return merged


def merge_scores_robust_avg(
    train_scores: Dict[int, float], 
    val_scores: Dict[int, float],
    outlier_threshold: float = 0.5
) -> Dict[int, float]:
    """Method 5: Robust average - exclude outliers (like Block 0) from averaging.
    
    For blocks with very high scores (outliers), use the minimum (more conservative).
    For other blocks, use weighted average.
    """
    merged = {}
    max_score = max(max(train_scores.values()), max(val_scores.values()))
    outlier_threshold_value = max_score * outlier_threshold
    
    for block in range(16):
        train = train_scores[block]
        val = val_scores[block]
        
        # If either score is an outlier, use minimum (more conservative)
        if train > outlier_threshold_value or val > outlier_threshold_value:
            merged[block] = min(train, val)
        else:
            # Normal blocks: weighted average
            merged[block] = train * 0.6 + val * 0.4
    
    return merged


METHODS = {
    'average': merge_scores_average,
    'weighted_avg': merge_scores_weighted_avg,
    'conservative': merge_scores_conservative,
    'median_rank': merge_scores_median_rank,
    'robust_avg': merge_scores_robust_avg,
}


def main():
    parser = argparse.ArgumentParser(
        description="Merge train and validation importance scores into final ranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  average       - Simple average: (train + val) / 2
  weighted_avg  - Weighted average: 60% train + 40% val (RECOMMENDED)
  conservative  - Maximum (worst case): max(train, val)
  median_rank   - Median rank: average of train and val ranks
  robust_avg    - Robust average: exclude outliers, use min for outliers

Examples:
  # Use recommended weighted average method
  python merge_importance_scores.py \\
      --comparison_file results/profiling/exp3_importance_comparison/coco_2014_vqa/importance_comparison_coco_2014_vqa.json \\
      --method weighted_avg \\
      --output_file results/layer_importance_scores_merged.json

  # Compare all methods
  python merge_importance_scores.py \\
      --comparison_file results/profiling/exp3_importance_comparison/coco_2014_vqa/importance_comparison_coco_2014_vqa.json \\
      --compare_all
        """
    )
    parser.add_argument("--comparison_file", type=str, required=True,
                    help="Path to importance comparison JSON file")
    parser.add_argument("--method", type=str, default="weighted_avg",
                    choices=list(METHODS.keys()),
                    help="Method to merge scores (default: weighted_avg)")
    parser.add_argument("--output_file", type=str, default=None,
                    help="Output file path (default: auto-generated)")
    parser.add_argument("--compare_all", action="store_true",
                    help="Compare all methods and show rankings")
    parser.add_argument("--train_weight", type=float, default=0.6,
                    help="Weight for train set in weighted_avg method (default: 0.6)")
    
    args = parser.parse_args()
    
    # Load comparison file
    comparison_path = Path(args.comparison_file)
    if not comparison_path.exists():
        log.error(f"Comparison file not found: {comparison_path}")
        return
    
    with open(comparison_path, 'r') as f:
        comparison_data = json.load(f)
    
    dataset_name = comparison_data['dataset_name']
    train_scores = {int(k): v for k, v in comparison_data['train_scores'].items()}
    val_scores = {int(k): v for k, v in comparison_data['validation_scores'].items()}
    spearman_corr = comparison_data.get('spearman_correlation', 0.0)
    
    log.info(f"Dataset: {dataset_name}")
    log.info(f"Spearman correlation: {spearman_corr:.4f}")
    log.info("")
    
    if args.compare_all:
        # Compare all methods
        log.info("=" * 80)
        log.info("COMPARING ALL METHODS")
        log.info("=" * 80)
        log.info("")
        
        all_results = {}
        for method_name, method_func in METHODS.items():
            if method_name == 'weighted_avg':
                merged = method_func(train_scores, val_scores, train_weight=args.train_weight)
            else:
                merged = method_func(train_scores, val_scores)
            
            # Sort by score (ascending: least important first)
            ranking = sorted(merged.items(), key=lambda x: x[1])
            all_results[method_name] = {
                'scores': merged,
                'ranking': ranking
            }
            
            log.info(f"Method: {method_name}")
            log.info("-" * 80)
            log.info("Top 5 least important (can remove first):")
            for i, (block, score) in enumerate(ranking[:5]):
                log.info(f"  {i+1}. Block {block:2d}: {score:.4f} "
                        f"(train={train_scores[block]:.4f}, val={val_scores[block]:.4f})")
            log.info("")
        
        # Show consensus (blocks that appear in top 5 across all methods)
        log.info("=" * 80)
        log.info("CONSENSUS ANALYSIS")
        log.info("=" * 80)
        log.info("")
        
        top5_blocks = {}
        for method_name, result in all_results.items():
            top5 = [block for block, _ in result['ranking'][:5]]
            for block in top5:
                top5_blocks[block] = top5_blocks.get(block, 0) + 1
        
        consensus_blocks = sorted(top5_blocks.items(), key=lambda x: x[1], reverse=True)
        log.info("Blocks appearing in top 5 across methods:")
        for block, count in consensus_blocks:
            log.info(f"  Block {block:2d}: appears in {count}/{len(METHODS)} methods")
        log.info("")
        
        # Save all results
        output_file = args.output_file or f"results/layer_importance_scores_all_methods_{dataset_name}.json"
        output_data = {
            'dataset_name': dataset_name,
            'spearman_correlation': spearman_corr,
            'train_scores': train_scores,
            'validation_scores': val_scores,
            'merged_scores': {method: result['scores'] for method, result in all_results.items()},
            'rankings': {method: [[block, score] for block, score in result['ranking']] 
                        for method, result in all_results.items()},
            'consensus': {block: count for block, count in consensus_blocks}
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        log.info(f"All methods comparison saved to: {output_file}")
        
    else:
        # Use single method
        method_func = METHODS[args.method]
        if args.method == 'weighted_avg':
            merged_scores = method_func(train_scores, val_scores, train_weight=args.train_weight)
        else:
            merged_scores = method_func(train_scores, val_scores)
        
        # Sort by score (ascending: least important first)
        ranking = sorted(merged_scores.items(), key=lambda x: x[1])
        
        log.info(f"Method: {args.method}")
        log.info("=" * 80)
        log.info("Final Block Importance Ranking (least → most important):")
        log.info("-" * 80)
        for i, (block, score) in enumerate(ranking):
            train = train_scores[block]
            val = val_scores[block]
            diff = abs(train - val)
            log.info(f"{i+1:2d}. Block {block:2d}: {score:.4f} "
                    f"(train={train:.4f}, val={val:.4f}, diff={diff:.4f})")
        log.info("")
        
        # Save results
        output_file = args.output_file or f"results/layer_importance_scores_{args.method}_{dataset_name}.json"
        output_data = {
            'dataset_name': dataset_name,
            'method': args.method,
            'spearman_correlation': spearman_corr,
            'train_scores': train_scores,
            'validation_scores': val_scores,
            'merged_scores': merged_scores,
            'ranking': [[block, score] for block, score in ranking]
        }
        
        if args.method == 'weighted_avg':
            output_data['train_weight'] = args.train_weight
        
        # Also save in format expected by other scripts (string keys)
        output_data_formatted = {str(k): v for k, v in merged_scores.items()}
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        log.info(f"Results saved to: {output_file}")
        
        # Also save simplified version (just scores, string keys)
        simple_output_file = output_file.replace('.json', '_simple.json')
        with open(simple_output_file, 'w') as f:
            json.dump(output_data_formatted, f, indent=2)
        log.info(f"Simplified scores (for use in other scripts) saved to: {simple_output_file}")


if __name__ == "__main__":
    main()

