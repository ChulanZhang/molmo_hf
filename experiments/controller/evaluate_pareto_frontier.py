"""
Evaluate Lookup Table Baseline Controller for Pareto Frontier Analysis.

This script evaluates multiple datasets and latency budgets, then collects
results in a format suitable for plotting Pareto frontier curves.

Usage:
    python experiments/controller/evaluate_pareto_frontier.py \
        --model_path checkpoints/molmo \
        --lookup_table_path ./checkpoints/controller/lookup_table_baseline.json \
        --datasets text_vqa okvqa coco_2014_vqa \
        --latency_budgets 170 200 230 260 290 320 350 380 \
        --num_samples 1000 \
        --output_path ./results/logs_eval/pareto_frontier/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.controller.evaluate_lookup_table_baseline import evaluate_lookup_table_baseline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def collect_pareto_data(
    results_dir: Path,
    datasets: List[str],
    latency_budgets: List[float],
) -> Dict[str, List[Dict]]:
    """
    Collect evaluation results for Pareto frontier analysis.
    
    Args:
        results_dir: Base directory containing evaluation results
        datasets: List of dataset names
        latency_budgets: List of latency budgets evaluated
    
    Returns:
        Dict mapping dataset_name -> List of {budget, accuracy, latency, config} dicts
    """
    pareto_data = defaultdict(list)
    
    for dataset in datasets:
        for budget in latency_budgets:
            # Look for results file
            budget_dir = results_dir / dataset / f"budget_{budget:.0f}"
            results_file = budget_dir / f"{dataset}_validation_budget_{budget:.0f}_results.json"
            
            if not results_file.exists():
                log.warning(f"Results file not found: {results_file}")
                continue
            
            try:
                with open(results_file, 'r') as f:
                    result = json.load(f)
                
                metrics = result.get('metrics', {})
                predicted_config = result.get('predicted_config', {})
                
                pareto_data[dataset].append({
                    'budget': budget,
                    'accuracy': metrics.get('accuracy', 0.0),
                    'latency': metrics.get('avg_latency_ms', 0.0),
                    'latency_std': metrics.get('latency_std_ms', 0.0),
                    'accuracy_std': metrics.get('accuracy_std', 0.0),
                    'budget_violation_rate': metrics.get('budget_violation_rate', 0.0),
                    'config': {
                        'tier': predicted_config.get('tier', 'unknown'),
                        'top_k': predicted_config.get('top_k', 0),
                        'num_active_blocks': predicted_config.get('num_active_blocks', 0),
                    },
                    'expected_accuracy': predicted_config.get('accuracy', 0.0),
                    'expected_latency': predicted_config.get('latency', 0.0),
                })
                
            except Exception as e:
                log.error(f"Error loading {results_file}: {e}")
                continue
    
    return dict(pareto_data)


def compute_pareto_frontier(
    points: List[Dict],
    accuracy_key: str = 'accuracy',
    latency_key: str = 'latency',
) -> List[Dict]:
    """
    Compute Pareto frontier from evaluation points.
    
    For accuracy-latency tradeoff:
    - A point is on the Pareto frontier if there's no other point with both 
      higher accuracy AND lower latency
    - Or equivalently: no other point dominates it
    
    Args:
        points: List of point dictionaries with accuracy and latency
        accuracy_key: Key for accuracy in point dict
        latency_key: Key for latency in point dict
    
    Returns:
        List of Pareto frontier points, sorted by latency (ascending)
    """
    if len(points) == 0:
        return []
    
    # Sort by latency (ascending), then by accuracy (descending)
    sorted_points = sorted(
        points,
        key=lambda x: (x.get(latency_key, float('inf')), -x.get(accuracy_key, 0.0))
    )
    
    pareto_points = []
    
    for i, point in enumerate(sorted_points):
        is_pareto = True
        point_accuracy = point.get(accuracy_key, 0.0)
        point_latency = point.get(latency_key, float('inf'))
        
        # Check if this point is dominated by any other point
        for j, other_point in enumerate(sorted_points):
            if i == j:
                continue
            
            other_accuracy = other_point.get(accuracy_key, 0.0)
            other_latency = other_point.get(latency_key, float('inf'))
            
            # A point dominates if it has >= accuracy AND <= latency, 
            # with at least one strict inequality
            if (other_accuracy >= point_accuracy and 
                other_latency <= point_latency and
                (other_accuracy > point_accuracy or other_latency < point_latency)):
                is_pareto = False
                break
        
        if is_pareto:
            pareto_points.append(point)
    
    # Sort Pareto points by latency for plotting
    pareto_points.sort(key=lambda x: x.get(latency_key, float('inf')))
    
    return pareto_points


def save_pareto_data(
    pareto_data: Dict[str, List[Dict]],
    output_file: Path,
):
    """
    Save Pareto data to JSON file.
    
    Args:
        pareto_data: Dict mapping dataset -> list of points
        output_file: Output JSON file path
    """
    # Compute Pareto frontiers for each dataset
    pareto_frontiers = {}
    all_points = {}
    
    for dataset, points in pareto_data.items():
        all_points[dataset] = points
        pareto_frontiers[dataset] = compute_pareto_frontier(points)
    
    output_data = {
        'all_points': all_points,
        'pareto_frontiers': pareto_frontiers,
        'summary': {
            dataset: {
                'total_points': len(points),
                'pareto_points': len(pareto_frontiers[dataset]),
                'latency_range': [
                    min(p.get('latency', 0) for p in points) if points else 0,
                    max(p.get('latency', 0) for p in points) if points else 0,
                ],
                'accuracy_range': [
                    min(p.get('accuracy', 0) for p in points) if points else 0,
                    max(p.get('accuracy', 0) for p in points) if points else 0,
                ],
            }
            for dataset, points in pareto_data.items()
        }
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    log.info(f"Pareto data saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Lookup Table Baseline for Pareto Frontier Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--lookup_table_path",
        type=str,
        required=True,
        help="Path to lookup table JSON file"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["text_vqa", "okvqa", "coco_2014_vqa"],
        help="List of dataset names to evaluate"
    )
    parser.add_argument(
        "--latency_budgets",
        type=float,
        nargs="+",
        default=[200.0, 260.0, 320.0, 380.0],  # Reduced for debugging
        help="List of latency budgets to evaluate (in ms)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,  # Reduced for debugging
        help="Number of samples to evaluate per dataset"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",  # Use GPU 1 by default
        help="Device to use"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/logs_eval/pareto_frontier/",
        help="Base output directory for results"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save individual predictions to file"
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip evaluation, only collect existing results and compute Pareto frontiers"
    )
    
    args = parser.parse_args()
    
    log.info("=" * 80)
    log.info("Pareto Frontier Evaluation")
    log.info("=" * 80)
    log.info(f"Model path: {args.model_path}")
    log.info(f"Lookup table path: {args.lookup_table_path}")
    log.info(f"Datasets: {args.datasets}")
    log.info(f"Latency budgets: {args.latency_budgets}")
    log.info(f"Num samples per dataset: {args.num_samples}")
    log.info(f"Output path: {args.output_path}")
    log.info("=" * 80)
    
    output_dir = Path(args.output_path)
    results_dir = output_dir / "results"
    
    # Run evaluations if not skipping
    if not args.skip_evaluation:
        total_experiments = len(args.datasets) * len(args.latency_budgets)
        log.info(f"Total experiments: {total_experiments}")
        
        completed = 0
        failed = []
        
        for dataset in args.datasets:
            for budget in args.latency_budgets:
                completed += 1
                log.info(f"\n[{completed}/{total_experiments}] Evaluating {dataset} with budget {budget}ms")
                
                try:
                    dataset_output_dir = results_dir / dataset / f"budget_{budget:.0f}"
                    
                    evaluate_lookup_table_baseline(
                        model_path=args.model_path,
                        lookup_table_path=args.lookup_table_path,
                        dataset=dataset,
                        split="validation",
                        num_samples=args.num_samples,
                        latency_budget=budget,
                        max_new_tokens=args.max_new_tokens,
                        batch_size=1,
                        device=args.device,
                        output_path=str(dataset_output_dir),
                        save_predictions=args.save_predictions,
                    )
                    
                    log.info(f"✓ Completed {dataset} @ {budget}ms")
                    
                except Exception as e:
                    log.error(f"✗ Failed {dataset} @ {budget}ms: {e}")
                    failed.append((dataset, budget, str(e)))
                    import traceback
                    traceback.print_exc()
                    continue
        
        log.info("\n" + "=" * 80)
        log.info("Evaluation Summary")
        log.info("=" * 80)
        log.info(f"Completed: {completed - len(failed)}/{total_experiments}")
        log.info(f"Failed: {len(failed)}/{total_experiments}")
        
        if failed:
            log.warning("\nFailed experiments:")
            for dataset, budget, error in failed:
                log.warning(f"  {dataset} @ {budget}ms: {error}")
    
    # Collect results and compute Pareto frontiers
    log.info("\n" + "=" * 80)
    log.info("Collecting Results and Computing Pareto Frontiers")
    log.info("=" * 80)
    
    pareto_data = collect_pareto_data(
        results_dir=results_dir,
        datasets=args.datasets,
        latency_budgets=args.latency_budgets,
    )
    
    # Print summary
    for dataset, points in pareto_data.items():
        log.info(f"\n{dataset}:")
        log.info(f"  Total points: {len(points)}")
        if points:
            pareto_frontier = compute_pareto_frontier(points)
            log.info(f"  Pareto frontier points: {len(pareto_frontier)}")
            log.info(f"  Latency range: [{min(p['latency'] for p in points):.1f}, {max(p['latency'] for p in points):.1f}]ms")
            log.info(f"  Accuracy range: [{min(p['accuracy'] for p in points):.4f}, {max(p['accuracy'] for p in points):.4f}]")
            
            log.info("  Pareto frontier points:")
            for point in pareto_frontier:
                log.info(f"    Budget: {point['budget']:.0f}ms, "
                        f"Latency: {point['latency']:.1f}ms, "
                        f"Accuracy: {point['accuracy']:.4f}, "
                        f"Config: {point['config']}")
    
    # Save Pareto data
    pareto_file = output_dir / "pareto_data.json"
    save_pareto_data(pareto_data, pareto_file)
    
    log.info("\n" + "=" * 80)
    log.info("Pareto Frontier Analysis Complete")
    log.info("=" * 80)
    log.info(f"Results saved to: {results_dir}")
    log.info(f"Pareto data saved to: {pareto_file}")
    log.info(f"\nTo plot Pareto frontiers, run:")
    log.info(f"  python experiments/controller/plot_pareto_frontier.py --pareto_data {pareto_file}")


if __name__ == "__main__":
    main()

