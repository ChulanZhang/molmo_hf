"""
Validate importance score consistency across train/val splits and datasets.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.base_experiment import BaseExperiment
from experiments.profiling.knob3_layers.exp3_accuracy_sensitivity_v2 import Exp3SensitivityExperimentV2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def compute_spearman_correlation(
    scores1: Dict[int, float],
    scores2: Dict[int, float]
) -> Tuple[float, float]:
    """
    Compute Spearman correlation between two importance score dictionaries.
    
    Args:
        scores1: Dict mapping block index to importance score
        scores2: Dict mapping block index to importance score
    
    Returns:
        correlation: Spearman correlation coefficient
        p_value: P-value
    """
    # Get common blocks
    common_blocks = sorted(set(scores1.keys()) & set(scores2.keys()))
    
    if len(common_blocks) < 2:
        return 0.0, 1.0
    
    # Extract scores in same order
    scores1_list = [scores1[block] for block in common_blocks]
    scores2_list = [scores2[block] for block in common_blocks]
    
    # Compute correlation
    correlation, p_value = spearmanr(scores1_list, scores2_list)
    
    return float(correlation), float(p_value)


def validate_train_val_consistency(
    dataset_name: str,
    model_path: str,
    num_samples: int = 5000,
    batch_size: int = 64,
    max_new_tokens: int = 16,
    output_dir: str = "results/importance_validation",
) -> Dict:
    """
    Validate importance score consistency between train and val splits.
    
    Args:
        dataset_name: Dataset name
        model_path: Path to model
        num_samples: Number of samples for importance computation
        batch_size: Batch size
        max_new_tokens: Max new tokens for generation
        output_dir: Output directory
    
    Returns:
        results: {
            'dataset_name': str,
            'spearman_correlation': float,
            'p_value': float,
            'is_consistent': bool,
            'train_scores': Dict[int, float],
            'val_scores': Dict[int, float],
        }
    """
    log.info(f"Validating train/val consistency for {dataset_name}...")
    
    # Initialize experiment
    experiment = Exp3SensitivityExperimentV2(
        model_path=model_path,
        device="cuda",
    )
    
    # Compute importance scores on train split
    log.info("Computing importance scores on train split...")
    train_scores = experiment._sensitivity_analysis(
        dataset_name=dataset_name,
        split="train",
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
    )
    
    # Compute importance scores on val split
    log.info("Computing importance scores on validation split...")
    val_scores = experiment._sensitivity_analysis(
        dataset_name=dataset_name,
        split="validation",
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
    )
    
    # Compute correlation
    correlation, p_value = compute_spearman_correlation(train_scores, val_scores)
    
    # Check consistency (correlation > 0.8 and p < 0.05)
    is_consistent = correlation > 0.8 and p_value < 0.05
    
    results = {
        'dataset_name': dataset_name,
        'spearman_correlation': correlation,
        'p_value': p_value,
        'is_consistent': is_consistent,
        'train_scores': {str(k): v for k, v in train_scores.items()},
        'val_scores': {str(k): v for k, v in val_scores.items()},
    }
    
    log.info(f"Train/Val Consistency for {dataset_name}:")
    log.info(f"  Spearman correlation: {correlation:.4f}")
    log.info(f"  P-value: {p_value:.4e}")
    log.info(f"  Is consistent: {is_consistent}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"train_val_consistency_{dataset_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Results saved to {output_file}")
    
    return results


def validate_cross_dataset_consistency(
    dataset_names: List[str],
    model_path: str,
    num_samples: int = 5000,
    batch_size: int = 64,
    max_new_tokens: int = 16,
    output_dir: str = "results/importance_validation",
) -> Dict:
    """
    Validate importance score consistency across multiple datasets.
    
    Args:
        dataset_names: List of dataset names
        model_path: Path to model
        num_samples: Number of samples for importance computation
        batch_size: Batch size
        max_new_tokens: Max new tokens for generation
        output_dir: Output directory
    
    Returns:
        results: {
            'pairwise_correlations': Dict[Tuple[str, str], float],
            'mean_correlation': float,
            'is_consistent': bool,
            'dataset_scores': Dict[str, Dict[int, float]],
            'merged_scores': Dict[int, float],
        }
    """
    log.info(f"Validating cross-dataset consistency for {dataset_names}...")
    
    # Initialize experiment
    experiment = Exp3SensitivityExperimentV2(
        model_path=model_path,
        device="cuda",
    )
    
    # Compute importance scores for each dataset
    dataset_scores = {}
    for dataset_name in dataset_names:
        log.info(f"Computing importance scores for {dataset_name}...")
        scores = experiment._sensitivity_analysis(
            dataset_name=dataset_name,
            split="validation",
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
        )
        dataset_scores[dataset_name] = scores
    
    # Compute pairwise correlations
    pairwise_correlations = {}
    datasets = list(dataset_scores.keys())
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            corr, p_value = compute_spearman_correlation(
                dataset_scores[datasets[i]],
                dataset_scores[datasets[j]]
            )
            pairwise_correlations[(datasets[i], datasets[j])] = {
                'correlation': corr,
                'p_value': p_value,
            }
            log.info(f"  {datasets[i]} vs {datasets[j]}: correlation={corr:.4f}, p={p_value:.4e}")
    
    mean_correlation = np.mean([
        v['correlation'] for v in pairwise_correlations.values()
    ])
    
    # Check consistency (mean correlation > 0.7)
    is_consistent = mean_correlation > 0.7
    
    # Merge scores (average across datasets)
    merged_scores = {}
    all_blocks = set()
    for scores in dataset_scores.values():
        all_blocks.update(scores.keys())
    
    for block_idx in sorted(all_blocks):
        scores = [dataset_scores[ds].get(block_idx, 0.0) for ds in datasets]
        merged_scores[block_idx] = float(np.mean(scores))
    
    results = {
        'pairwise_correlations': {
            f"{ds1}_{ds2}": v for (ds1, ds2), v in pairwise_correlations.items()
        },
        'mean_correlation': float(mean_correlation),
        'is_consistent': is_consistent,
        'dataset_scores': {
            ds: {str(k): v for k, v in scores.items()}
            for ds, scores in dataset_scores.items()
        },
        'merged_scores': {str(k): v for k, v in merged_scores.items()},
    }
    
    log.info(f"Cross-Dataset Consistency:")
    log.info(f"  Mean correlation: {mean_correlation:.4f}")
    log.info(f"  Is consistent: {is_consistent}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "cross_dataset_consistency.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Results saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate importance score consistency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Single dataset name for train/val validation"
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        default=None,
        help="Multiple dataset names for cross-dataset validation"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples for importance computation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Max new tokens for generation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/importance_validation",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    if args.dataset_name:
        # Train/val consistency
        validate_train_val_consistency(
            dataset_name=args.dataset_name,
            model_path=args.model_path,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            output_dir=args.output_dir,
        )
    
    if args.dataset_names:
        # Cross-dataset consistency
        validate_cross_dataset_consistency(
            dataset_names=args.dataset_names,
            model_path=args.model_path,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()







