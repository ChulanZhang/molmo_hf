#!/usr/bin/env python3
"""
Analyze dataset correlation for Knob1 (vision tokens) and Knob2 (MoE top-K).

This script analyzes multi-dataset profiling results to determine if knob values
are dataset-dependent (requiring content-aware control) or dataset-independent
(permitting content-agnostic control like lookup tables).

Key Question:
- Are optimal vision_tokens and top_k values similar across datasets?
- High correlation → Use content-agnostic control (lookup table)
- Low correlation → Use content-aware control (controller)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_profiling_results(results_dir: Path) -> Dict[str, List[Dict]]:
    """
    Load profiling results from core experiment JSON files.
    
    Args:
        results_dir: Directory containing profiling results
    
    Returns:
        Dict mapping dataset_name -> list of profiling results
    """
    results = defaultdict(list)
    
    # Find all JSON files in results directory
    json_files = list(results_dir.glob("**/*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Extract dataset name from path
            # Expected structure: results/core_exp_h100/{dataset_name}/{config}.json
            parts = json_file.parts
            if len(parts) >= 3 and "core_exp" in parts[-3]:
                dataset_name = parts[-2]
            else:
                # Try to infer from filename
                dataset_name = json_file.stem.split("_")[0]
            
            # Extract relevant metrics
            if isinstance(data, list):
                results[dataset_name].extend(data)
            elif isinstance(data, dict):
                results[dataset_name].append(data)
                
        except Exception as e:
            log.warning(f"Failed to load {json_file}: {e}")
            continue
    
    return dict(results)


def find_optimal_configs(
    results: List[Dict],
    latency_budgets: List[float],
    metric: str = "accuracy"
) -> Dict[float, Dict]:
    """
    Find optimal (vision_tokens, top_k) configuration for each latency budget.
    
    Args:
        results: List of profiling results
        latency_budgets: List of target latency budgets (ms)
        metric: Metric to optimize (default: "accuracy")
    
    Returns:
        Dict mapping latency_budget -> optimal_config
    """
    optimal_configs = {}
    
    for budget in latency_budgets:
        # Filter results within budget range (±10%)
        budget_min = budget * 0.9
        budget_max = budget * 1.1
        
        candidates = [
            r for r in results
            if budget_min <= r.get("T_total", float('inf')) <= budget_max
        ]
        
        if not candidates:
            # If no exact match, find closest
            candidates = sorted(
                results,
                key=lambda x: abs(x.get("T_total", float('inf')) - budget)
            )[:10]  # Top 10 closest
        
        # Find configuration with highest accuracy within budget
        best = max(candidates, key=lambda x: x.get(metric, 0))
        
        optimal_configs[budget] = {
            "vision_tokens": best.get("vision_tokens", 0),
            "top_k": best.get("top_k", 8),
            "num_blocks": best.get("num_active_blocks", 16),
            "accuracy": best.get(metric, 0),
            "latency": best.get("T_total", 0),
        }
    
    return optimal_configs


def calculate_correlation(values_dict: Dict[str, List[float]]) -> float:
    """
    Calculate Spearman correlation across datasets.
    
    Args:
        values_dict: Dict mapping dataset_name -> list of values
    
    Returns:
        Spearman correlation coefficient
    """
    from scipy.stats import spearmanr
    
    datasets = list(values_dict.keys())
    if len(datasets) < 2:
        return 1.0  # Only one dataset, perfect correlation
    
    # Get common length (minimum length across datasets)
    min_len = min(len(values_dict[d]) for d in datasets)
    if min_len == 0:
        return 0.0
    
    # Truncate all to same length
    values_list = [values_dict[d][:min_len] for d in datasets]
    
    # Calculate pairwise correlations
    correlations = []
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            corr, pval = spearmanr(values_list[i], values_list[j])
            correlations.append(corr)
    
    # Return average correlation
    return sum(correlations) / len(correlations) if correlations else 0.0


def analyze_knob_correlation(
    profiling_results: Dict[str, List[Dict]],
    latency_budgets: List[float] = [50, 100, 150, 200, 300, 400, 500]
) -> Dict:
    """
    Analyze correlation of optimal knob values across datasets.
    
    Args:
        profiling_results: Dict mapping dataset_name -> list of results
        latency_budgets: List of latency budgets to analyze
    
    Returns:
        Analysis results with correlations and recommendations
    """
    log.info("=" * 80)
    log.info("Knob Dataset Correlation Analysis")
    log.info("=" * 80)
    
    # Find optimal configurations for each dataset
    optimal_configs_per_dataset = {}
    for dataset_name, results in profiling_results.items():
        log.info(f"\nAnalyzing {dataset_name}...")
        optimal_configs = find_optimal_configs(results, latency_budgets)
        optimal_configs_per_dataset[dataset_name] = optimal_configs
        
        # Log summary
        log.info(f"  Found {len(optimal_configs)} optimal configurations")
        for budget, config in list(optimal_configs.items())[:3]:
            log.info(f"    Budget {budget}ms: vision_tokens={config['vision_tokens']}, "
                    f"top_k={config['top_k']}, accuracy={config['accuracy']:.3f}")
    
    # Extract vision_tokens and top_k values for correlation analysis
    vision_tokens_by_dataset = {}
    top_k_by_dataset = {}
    
    for dataset_name, configs in optimal_configs_per_dataset.items():
        vision_tokens_by_dataset[dataset_name] = [
            configs[b]["vision_tokens"] for b in latency_budgets
            if b in configs
        ]
        top_k_by_dataset[dataset_name] = [
            configs[b]["top_k"] for b in latency_budgets
            if b in configs
        ]
    
    # Calculate correlations
    vision_tokens_corr = calculate_correlation(vision_tokens_by_dataset)
    top_k_corr = calculate_correlation(top_k_by_dataset)
    
    # Make recommendations
    knob1_recommendation = (
        "content_agnostic" if vision_tokens_corr > 0.7
        else "content_aware"
    )
    knob2_recommendation = (
        "content_agnostic" if top_k_corr > 0.7
        else "content_aware"
    )
    
    # Prepare results
    results = {
        "vision_tokens_correlation": vision_tokens_corr,
        "top_k_correlation": top_k_corr,
        "knob1_recommendation": knob1_recommendation,
        "knob2_recommendation": knob2_recommendation,
        "optimal_configs": optimal_configs_per_dataset,
        "datasets_analyzed": list(profiling_results.keys()),
    }
    
    # Print summary
    log.info("\n" + "=" * 80)
    log.info("ANALYSIS RESULTS")
    log.info("=" * 80)
    log.info(f"\nVision Tokens (Knob1) Correlation: {vision_tokens_corr:.4f}")
    log.info(f"  Recommendation: {knob1_recommendation.upper()}")
    if vision_tokens_corr > 0.7:
        log.info("  → Use lookup table based on latency budget")
    else:
        log.info("  → Use content-aware controller")
    
    log.info(f"\nMoE Top-K (Knob2) Correlation: {top_k_corr:.4f}")
    log.info(f"  Recommendation: {knob2_recommendation.upper()}")
    if top_k_corr > 0.7:
        log.info("  → Use lookup table based on latency budget")
    else:
        log.info("  → Use content-aware controller")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze knob dataset correlation"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/core_exp_h100",
        help="Directory containing profiling results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/knob_correlation_analysis.json",
        help="Output file for analysis results"
    )
    parser.add_argument(
        "--latency_budgets",
        type=float,
        nargs="+",
        default=[50, 100, 150, 200, 300, 400, 500],
        help="Latency budgets to analyze (ms)"
    )
    
    args = parser.parse_args()
    
    # Load profiling results
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        log.error(f"Results directory not found: {results_dir}")
        return
    
    log.info(f"Loading profiling results from {results_dir}...")
    profiling_results = load_profiling_results(results_dir)
    
    if not profiling_results:
        log.error("No profiling results found!")
        return
    
    log.info(f"Loaded results for {len(profiling_results)} datasets:")
    for dataset_name, results in profiling_results.items():
        log.info(f"  {dataset_name}: {len(results)} configurations")
    
    # Analyze correlation
    analysis_results = analyze_knob_correlation(
        profiling_results,
        latency_budgets=args.latency_budgets
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(analysis_results, f, indent=2)
    
    log.info(f"\nAnalysis results saved to {output_path}")


if __name__ == "__main__":
    main()





