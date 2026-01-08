#!/usr/bin/env python3
"""
Merge importance scores from multiple datasets into a single cross-dataset ranking.

This script:
1. Loads importance comparison files from exp3_importance_comparison
2. For each dataset, merges train and validation scores (using weighted_avg)
3. Merges scores across all datasets (using average or weighted average)
4. Generates a unified layer_importance_scores.json file

Usage:
    python experiments/profiling/knob3_layers/merge_multi_dataset_importance_scores.py \
        --comparison_dir results/profiling/exp3_importance_comparison \
        --output_file results/layer_importance_scores_multi_dataset.json \
        --method average
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def merge_train_val_scores(
    train_scores: Dict[int, float],
    val_scores: Dict[int, float],
    train_weight: float = 0.6
) -> Dict[int, float]:
    """
    Merge train and validation scores using weighted average.
    
    Args:
        train_scores: Train set importance scores
        val_scores: Validation set importance scores
        train_weight: Weight for train set (default: 0.6)
    
    Returns:
        Merged scores
    """
    merged = {}
    val_weight = 1.0 - train_weight
    for block in range(16):
        merged[block] = train_scores[block] * train_weight + val_scores[block] * val_weight
    return merged


def merge_cross_dataset_scores_average(
    dataset_scores: List[Dict[int, float]]
) -> Dict[int, float]:
    """
    Merge scores across datasets using simple average.
    
    Args:
        dataset_scores: List of importance scores from different datasets
    
    Returns:
        Averaged scores across all datasets
    """
    merged = {}
    num_datasets = len(dataset_scores)
    
    for block in range(16):
        total_score = sum(scores[block] for scores in dataset_scores)
        merged[block] = total_score / num_datasets
    
    return merged


def merge_cross_dataset_scores_weighted_avg(
    dataset_scores: List[Dict[int, float]],
    dataset_weights: Optional[List[float]] = None
) -> Dict[int, float]:
    """
    Merge scores across datasets using weighted average.
    
    Args:
        dataset_scores: List of importance scores from different datasets
        dataset_weights: Optional weights for each dataset (default: equal weights)
    
    Returns:
        Weighted averaged scores across all datasets
    """
    merged = {}
    num_datasets = len(dataset_scores)
    
    if dataset_weights is None:
        dataset_weights = [1.0 / num_datasets] * num_datasets
    
    # Normalize weights
    total_weight = sum(dataset_weights)
    normalized_weights = [w / total_weight for w in dataset_weights]
    
    for block in range(16):
        weighted_sum = sum(scores[block] * weight for scores, weight in zip(dataset_scores, normalized_weights))
        merged[block] = weighted_sum
    
    return merged


def merge_cross_dataset_scores_median(
    dataset_scores: List[Dict[int, float]]
) -> Dict[int, float]:
    """
    Merge scores across datasets using median (robust to outliers).
    
    Args:
        dataset_scores: List of importance scores from different datasets
    
    Returns:
        Median scores across all datasets
    """
    import statistics
    
    merged = {}
    for block in range(16):
        block_scores = [scores[block] for scores in dataset_scores]
        merged[block] = statistics.median(block_scores)
    
    return merged


METHODS = {
    'average': merge_cross_dataset_scores_average,
    'weighted_avg': merge_cross_dataset_scores_weighted_avg,
    'median': merge_cross_dataset_scores_median,
}


def load_all_dataset_scores(
    comparison_dir: Path,
    train_val_weight: float = 0.6
) -> Tuple[List[Dict[int, float]], List[str], Dict[str, Dict]]:
    """
    Load and merge importance scores from all datasets.
    
    Args:
        comparison_dir: Directory containing importance_comparison_*.json files
        train_val_weight: Weight for train set when merging train/val scores
    
    Returns:
        Tuple of:
        - List of merged scores (one per dataset)
        - List of dataset names
        - Dictionary of per-dataset metadata
    """
    all_scores = []
    dataset_names = []
    metadata = {}
    
    # Find all importance_comparison_*.json files
    comparison_files = sorted(comparison_dir.glob("*/importance_comparison_*.json"))
    
    if not comparison_files:
        log.warning(f"No importance comparison files found in {comparison_dir}")
        return all_scores, dataset_names, metadata
    
    log.info(f"Found {len(comparison_files)} dataset comparison files")
    
    for comparison_file in comparison_files:
        try:
            with open(comparison_file, 'r') as f:
                data = json.load(f)
            
            dataset_name = data['dataset_name']
            train_scores = {int(k): v for k, v in data['train_scores'].items()}
            val_scores = {int(k): v for k, v in data['validation_scores'].items()}
            spearman_corr = data.get('spearman_correlation', 0.0)
            
            # Merge train and validation scores
            merged_scores = merge_train_val_scores(train_scores, val_scores, train_val_weight)
            
            all_scores.append(merged_scores)
            dataset_names.append(dataset_name)
            metadata[dataset_name] = {
                'spearman_correlation': spearman_corr,
                'train_scores': train_scores,
                'validation_scores': val_scores,
                'merged_scores': merged_scores,
            }
            
            log.info(f"  ✓ {dataset_name}: Spearman={spearman_corr:.4f}")
            
        except Exception as e:
            log.error(f"Error loading {comparison_file}: {e}")
            continue
    
    return all_scores, dataset_names, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Merge importance scores from multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods for cross-dataset merging:
  average      - Simple average across all datasets
  weighted_avg  - Weighted average (can specify dataset weights)
  median       - Median across datasets (robust to outliers)

Examples:
  # Use simple average
  python merge_multi_dataset_importance_scores.py \\
      --comparison_dir results/profiling/exp3_importance_comparison \\
      --output_file results/layer_importance_scores_multi_dataset.json \\
      --method average

  # Use weighted average with custom weights
  python merge_multi_dataset_importance_scores.py \\
      --comparison_dir results/profiling/exp3_importance_comparison \\
      --output_file results/layer_importance_scores_multi_dataset.json \\
      --method weighted_avg \\
      --dataset_weights 0.2 0.2 0.15 0.15 0.1 0.1 0.05 0.05
        """
    )
    parser.add_argument("--comparison_dir", type=str, required=True,
                        help="Directory containing importance_comparison_*.json files")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file path for merged scores")
    parser.add_argument("--method", type=str, default="average",
                        choices=list(METHODS.keys()),
                        help="Method to merge scores across datasets (default: average)")
    parser.add_argument("--train_val_weight", type=float, default=0.6,
                        help="Weight for train set when merging train/val scores (default: 0.6)")
    parser.add_argument("--dataset_weights", type=float, nargs="+", default=None,
                        help="Optional weights for each dataset (only used with weighted_avg)")
    
    args = parser.parse_args()
    
    comparison_dir = Path(args.comparison_dir)
    if not comparison_dir.exists():
        log.error(f"Comparison directory not found: {comparison_dir}")
        return
    
    # Load all dataset scores
    log.info("=" * 80)
    log.info("Loading dataset importance scores")
    log.info("=" * 80)
    all_scores, dataset_names, metadata = load_all_dataset_scores(
        comparison_dir,
        train_val_weight=args.train_val_weight
    )
    
    if not all_scores:
        log.error("No valid dataset scores found")
        return
    
    log.info("")
    log.info(f"Loaded {len(all_scores)} datasets: {', '.join(dataset_names)}")
    log.info("")
    
    # Merge across datasets
    log.info("=" * 80)
    log.info(f"Merging scores across datasets using method: {args.method}")
    log.info("=" * 80)
    
    method_func = METHODS[args.method]
    if args.method == 'weighted_avg':
        merged_scores = method_func(all_scores, dataset_weights=args.dataset_weights)
    else:
        merged_scores = method_func(all_scores)
    
    # Sort by score (ascending: least important first)
    ranking = sorted(merged_scores.items(), key=lambda x: x[1])
    
    log.info("")
    log.info("Final Cross-Dataset Block Importance Ranking (least → most important):")
    log.info("-" * 80)
    for i, (block, score) in enumerate(ranking):
        # Show per-dataset scores for context
        dataset_scores_str = ", ".join(f"{name}:{scores[block]:.4f}" 
                                      for name, scores in zip(dataset_names, all_scores))
        log.info(f"{i+1:2d}. Block {block:2d}: {score:.4f}  [{dataset_scores_str}]")
    log.info("")
    
    # Save results
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'method': args.method,
        'train_val_weight': args.train_val_weight,
        'num_datasets': len(dataset_names),
        'datasets': dataset_names,
        'per_dataset_metadata': metadata,
        'merged_scores': merged_scores,
        'ranking': [[block, score] for block, score in ranking]
    }
    
    if args.method == 'weighted_avg' and args.dataset_weights:
        output_data['dataset_weights'] = args.dataset_weights
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    log.info(f"Results saved to: {output_file}")
    
    # Also save simplified version (just scores, string keys) for use in other scripts
    simple_output_file = output_file.with_name(output_file.stem + "_simple.json")
    simple_scores = {str(k): v for k, v in merged_scores.items()}
    with open(simple_output_file, 'w') as f:
        json.dump(simple_scores, f, indent=2)
    log.info(f"Simplified scores (for use in other scripts) saved to: {simple_output_file}")
    
    log.info("")
    log.info("=" * 80)
    log.info("Summary")
    log.info("=" * 80)
    log.info(f"Top 5 least important blocks (can prune first):")
    for i, (block, score) in enumerate(ranking[:5]):
        log.info(f"  {i+1}. Block {block:2d}: {score:.4f}")
    log.info("")
    log.info(f"Top 5 most important blocks (keep these):")
    for i, (block, score) in enumerate(ranking[-5:][::-1]):
        log.info(f"  {i+1}. Block {block:2d}: {score:.4f}")
    log.info("")


if __name__ == "__main__":
    main()

