"""
Analyze profiling results to determine latency budget range from Pareto frontier.

For each dataset:
- Find Pareto frontier points
- Highest accuracy point → latency upper bound
- Lowest accuracy point → latency lower bound
- Aggregate across datasets to get overall range
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_profiling_results(results_dir: str, dataset_name: str) -> List[Dict]:
    """
    Load profiling results for a dataset.
    
    Args:
        results_dir: Directory containing profiling results
        dataset_name: Dataset name (e.g., "text-vqa", "coco-2014-vqa")
    
    Returns:
        List of configuration results
    """
    dataset_dir = Path(results_dir) / dataset_name
    if not dataset_dir.exists():
        log.warning(f"Dataset directory not found: {dataset_dir}")
        return []
    
    results = []
    json_files = list(dataset_dir.glob("*.json"))
    
    log.info(f"Loading {len(json_files)} result files from {dataset_name}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract configuration and metrics
            tier = data.get('tier', 'medium')
            top_k = data.get('top_k', 8)
            num_active_blocks = data.get('num_active_blocks', 16)
            
            accuracy = data.get('accuracy', 0.0)
            latency_stats = data.get('latency_stats', {})
            T_total_mean = latency_stats.get('T_total_mean', 0.0)
            
            if accuracy > 0 and T_total_mean > 0:
                results.append({
                    'tier': tier,
                    'top_k': top_k,
                    'num_active_blocks': num_active_blocks,
                    'accuracy': accuracy,
                    'latency': T_total_mean,
                    'config_key': (tier, top_k, num_active_blocks),
                })
        except Exception as e:
            log.warning(f"Error loading {json_file}: {e}")
            continue
    
    log.info(f"Loaded {len(results)} valid configurations from {dataset_name}")
    return results


def compute_pareto_frontier(
    points: List[Dict]
) -> List[Dict]:
    """
    Compute Pareto frontier points.
    
    A point is on the Pareto frontier if there's no other point with both
    higher accuracy AND lower latency.
    
    Args:
        points: List of dicts with 'accuracy' and 'latency' keys
    
    Returns:
        List of Pareto frontier points, sorted by latency (ascending)
    """
    if len(points) == 0:
        return []
    
    # Sort by latency (ascending), then by accuracy (descending)
    sorted_points = sorted(points, key=lambda x: (x['latency'], -x['accuracy']))
    
    pareto_points = []
    
    for i, point in enumerate(sorted_points):
        is_pareto = True
        
        # Check if this point is dominated by any other point
        for j, other_point in enumerate(sorted_points):
            if i == j:
                continue
            
            # A point dominates if it has >= accuracy AND <= latency, with at least one strict inequality
            if (other_point['accuracy'] >= point['accuracy'] and 
                other_point['latency'] <= point['latency'] and
                (other_point['accuracy'] > point['accuracy'] or 
                 other_point['latency'] < point['latency'])):
                is_pareto = False
                break
        
        if is_pareto:
            pareto_points.append(point)
    
    # Sort Pareto points by latency for analysis
    pareto_points.sort(key=lambda x: x['latency'])
    
    return pareto_points


def analyze_dataset(
    results_dir: str,
    dataset_name: str,
) -> Optional[Dict]:
    """
    Analyze a single dataset to find Pareto frontier and latency budget range.
    
    Returns:
        Dict with 'pareto_frontier', 'highest_acc', 'lowest_acc', 'latency_range'
    """
    results = load_profiling_results(results_dir, dataset_name)
    
    if len(results) == 0:
        return None
    
    # Compute Pareto frontier
    pareto_frontier = compute_pareto_frontier(results)
    
    if len(pareto_frontier) == 0:
        log.warning(f"No Pareto frontier points found for {dataset_name}")
        return None
    
    # Find highest and lowest accuracy points on Pareto frontier
    highest_acc_point = max(pareto_frontier, key=lambda x: x['accuracy'])
    lowest_acc_point = min(pareto_frontier, key=lambda x: x['accuracy'])
    
    return {
        'dataset': dataset_name,
        'pareto_frontier': pareto_frontier,
        'num_pareto_points': len(pareto_frontier),
        'highest_acc': {
            'accuracy': highest_acc_point['accuracy'],
            'latency': highest_acc_point['latency'],
            'config': highest_acc_point['config_key'],
        },
        'lowest_acc': {
            'accuracy': lowest_acc_point['accuracy'],
            'latency': lowest_acc_point['latency'],
            'config': lowest_acc_point['config_key'],
        },
        'latency_range': {
            'min': lowest_acc_point['latency'],
            'max': highest_acc_point['latency'],
        },
    }


def analyze_all_datasets(
    results_dir: str,
    dataset_names: List[str],
) -> Dict:
    """
    Analyze all datasets and aggregate results.
    
    Returns:
        Dict with per-dataset analysis and aggregated ranges
    """
    all_analyses = {}
    
    for dataset_name in dataset_names:
        log.info(f"\n{'='*60}")
        log.info(f"Analyzing {dataset_name}")
        log.info(f"{'='*60}")
        
        analysis = analyze_dataset(results_dir, dataset_name)
        if analysis:
            all_analyses[dataset_name] = analysis
            
            log.info(f"Pareto frontier points: {analysis['num_pareto_points']}")
            log.info(f"Highest accuracy: {analysis['highest_acc']['accuracy']:.4f} @ {analysis['highest_acc']['latency']:.2f}ms (config: {analysis['highest_acc']['config']})")
            log.info(f"Lowest accuracy: {analysis['lowest_acc']['accuracy']:.4f} @ {analysis['lowest_acc']['latency']:.2f}ms (config: {analysis['lowest_acc']['config']})")
            log.info(f"Latency range: {analysis['latency_range']['min']:.2f}ms - {analysis['latency_range']['max']:.2f}ms")
    
    # Aggregate across datasets
    if len(all_analyses) == 0:
        log.error("No valid analyses found!")
        return {}
    
    all_latency_mins = [a['latency_range']['min'] for a in all_analyses.values()]
    all_latency_maxs = [a['latency_range']['max'] for a in all_analyses.values()]
    
    aggregated = {
        'per_dataset': all_analyses,
        'aggregated_range': {
            'min_latency': min(all_latency_mins),  # Lower bound (lowest accuracy on Pareto)
            'max_latency': max(all_latency_maxs),  # Upper bound (highest accuracy on Pareto)
            'mean_min_latency': np.mean(all_latency_mins),
            'mean_max_latency': np.mean(all_latency_maxs),
            'median_min_latency': np.median(all_latency_mins),
            'median_max_latency': np.median(all_latency_maxs),
        },
    }
    
    log.info(f"\n{'='*60}")
    log.info("Aggregated Results Across All Datasets")
    log.info(f"{'='*60}")
    log.info(f"Min latency (lower bound): {aggregated['aggregated_range']['min_latency']:.2f}ms")
    log.info(f"Max latency (upper bound): {aggregated['aggregated_range']['max_latency']:.2f}ms")
    log.info(f"Mean min latency: {aggregated['aggregated_range']['mean_min_latency']:.2f}ms")
    log.info(f"Mean max latency: {aggregated['aggregated_range']['mean_max_latency']:.2f}ms")
    log.info(f"Median min latency: {aggregated['aggregated_range']['median_min_latency']:.2f}ms")
    log.info(f"Median max latency: {aggregated['aggregated_range']['median_max_latency']:.2f}ms")
    
    return aggregated


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze profiling results to determine latency budget range"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing profiling results"
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        default=["text-vqa", "coco-2014-vqa", "okvqa"],
        help="Dataset names to analyze"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional: Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Analyze all datasets
    results = analyze_all_datasets(args.results_dir, args.dataset_names)
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        log.info(f"\nResults saved to {args.output_file}")
    
    return results


if __name__ == "__main__":
    main()

