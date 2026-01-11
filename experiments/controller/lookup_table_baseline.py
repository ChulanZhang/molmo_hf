"""
Lookup Table Baseline Controller.

Based on offline profiling results from core_exp, this controller uses a lookup table
to find the best (tier, top_k, num_active_blocks) configuration that satisfies a given
latency budget while maximizing accuracy.

This is a simple baseline that doesn't require training - it directly uses profiling data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np

from experiments.controller.core_exp_data_loader import CoreExpDataLoader

log = logging.getLogger(__name__)


class LookupTableBaselineController:
    """
    Lookup table baseline controller.
    
    Given a latency budget, finds the (tier, top_k, num_active_blocks) configuration
    that satisfies the budget and maximizes accuracy.
    """
    
    def __init__(
        self,
        profiling_results: List[Dict],
        aggregation_method: str = "mean",
        tolerance: float = 0.05,  # 5% tolerance for latency budget
        use_prefill_only: bool = True,  # Use only prefill latency (not including decode)
    ):
        """
        Initialize lookup table controller from profiling results.
        
        Args:
            profiling_results: List of profiling result dictionaries, each containing:
                - tier: str ("low", "medium", "high")
                - top_k: int
                - num_active_blocks: int
                - accuracy: float
                - T_total: float (latency in ms)
            aggregation_method: How to aggregate multiple samples for same config.
                               Options: "mean", "median", "max_accuracy"
            tolerance: Relative tolerance for latency budget matching (0.05 = 5%)
        """
        self.profiling_results = profiling_results
        self.aggregation_method = aggregation_method
        self.tolerance = tolerance
        self.use_prefill_only = use_prefill_only
        
        # Build lookup table: (tier, top_k, num_active_blocks) -> (accuracy, latency)
        self.config_table = self._build_config_table()
        
        # Build latency budget lookup: budget -> best config
        self.lookup_table = self._build_lookup_table()
        
        latency_type = "prefill only" if use_prefill_only else "total (prefill + decode)"
        log.info(f"Built lookup table with {len(self.config_table)} unique configurations")
        log.info(f"Using {latency_type} latency")
        log.info(f"Latency range: [{self.min_latency:.2f}ms, {self.max_latency:.2f}ms]")
    
    def _build_config_table(self) -> Dict[Tuple[str, int, int], Dict]:
        """
        Build configuration table: aggregate results for each (tier, top_k, num_active_blocks).
        
        Returns:
            Dict mapping (tier, top_k, num_active_blocks) -> {
                'accuracy': float,
                'latency': float,
                'num_samples': int,
            }
        """
        config_groups = defaultdict(list)
        
        # Group results by configuration
        for result in self.profiling_results:
            tier = result.get('tier', 'medium')
            top_k = result.get('top_k')
            num_blocks = result.get('num_active_blocks')
            accuracy = result.get('accuracy', 0.0)
            
            # Use prefill-only latency if specified, otherwise use total
            if self.use_prefill_only:
                # Use vision + prefill (excluding decode which is variable)
                T_vision = result.get('T_vision_total', 0.0)
                T_prefill = result.get('T_LLM_prefill', 0.0)
                latency = T_vision + T_prefill
                # Fallback to T_total if prefill components not available
                if latency <= 0:
                    latency = result.get('T_total', 0.0)
            else:
                latency = result.get('T_total', 0.0)
            
            # Skip invalid entries
            if tier is None or top_k is None or num_blocks is None:
                continue
            if latency <= 0:
                continue
            
            key = (tier, top_k, num_blocks)
            config_groups[key].append({
                'accuracy': accuracy,
                'latency': latency,
            })
        
        # Aggregate each configuration
        config_table = {}
        for key, samples in config_groups.items():
            accuracies = [s['accuracy'] for s in samples]
            latencies = [s['latency'] for s in samples]
            
            if self.aggregation_method == "mean":
                avg_accuracy = np.mean(accuracies)
                avg_latency = np.mean(latencies)
            elif self.aggregation_method == "median":
                avg_accuracy = np.median(accuracies)
                avg_latency = np.median(latencies)
            elif self.aggregation_method == "max_accuracy":
                # Use latency of the sample with highest accuracy
                best_idx = np.argmax(accuracies)
                avg_accuracy = accuracies[best_idx]
                avg_latency = latencies[best_idx]
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
            
            config_table[key] = {
                'accuracy': float(avg_accuracy),
                'latency': float(avg_latency),
                'num_samples': len(samples),
                'accuracy_std': float(np.std(accuracies)),
                'latency_std': float(np.std(latencies)),
            }
        
        return config_table
    
    def _build_lookup_table(self) -> Dict[float, Dict]:
        """
        Build lookup table: latency_budget -> best configuration.
        
        For each latency budget, find all configurations that satisfy the budget,
        then select the one with highest accuracy.
        
        Returns:
            Dict mapping latency_budget -> best config dict
        """
        if not self.config_table:
            return {}
        
        # Get latency range
        latencies = [config['latency'] for config in self.config_table.values()]
        self.min_latency = min(latencies)
        self.max_latency = max(latencies)
        
        # Create discrete budget points (every 5ms)
        budget_step = 5.0
        budget_points = np.arange(
            self.min_latency,
            self.max_latency + budget_step,
            budget_step
        )
        
        lookup_table = {}
        
        for budget in budget_points:
            # Find all configurations that satisfy budget (with tolerance)
            budget_max = budget * (1 + self.tolerance)
            
            valid_configs = []
            for (tier, top_k, num_blocks), config_info in self.config_table.items():
                if config_info['latency'] <= budget_max:
                    valid_configs.append({
                        'tier': tier,
                        'top_k': top_k,
                        'num_active_blocks': num_blocks,
                        'accuracy': config_info['accuracy'],
                        'latency': config_info['latency'],
                        'num_samples': config_info['num_samples'],
                    })
            
            if valid_configs:
                # Select config with highest accuracy
                best_config = max(valid_configs, key=lambda x: x['accuracy'])
                lookup_table[float(budget)] = best_config
        
        return lookup_table
    
    def predict(
        self,
        latency_budget: float,
        return_all_candidates: bool = False,
    ) -> Optional[Dict]:
        """
        Predict best configuration for given latency budget.
        
        Args:
            latency_budget: Target latency budget in ms
            return_all_candidates: If True, return all valid configs sorted by accuracy
        
        Returns:
            Configuration dictionary with:
                - tier: str
                - top_k: int
                - num_active_blocks: int
                - accuracy: float (expected)
                - latency: float (expected)
            Or None if no valid configuration found.
        """
        # Find all configurations that satisfy budget
        budget_max = latency_budget * (1 + self.tolerance)
        
        valid_configs = []
        for (tier, top_k, num_blocks), config_info in self.config_table.items():
            if config_info['latency'] <= budget_max:
                valid_configs.append({
                    'tier': tier,
                    'top_k': top_k,
                    'num_active_blocks': num_blocks,
                    'accuracy': config_info['accuracy'],
                    'latency': config_info['latency'],
                    'num_samples': config_info['num_samples'],
                    'accuracy_std': config_info.get('accuracy_std', 0.0),
                    'latency_std': config_info.get('latency_std', 0.0),
                })
        
        if not valid_configs:
            # Fallback: find closest config by latency
            closest_config = min(
                self.config_table.items(),
                key=lambda x: abs(x[1]['latency'] - latency_budget)
            )
            (tier, top_k, num_blocks), config_info = closest_config
            closest_latency = config_info['latency']
            log.warning(
                f"No config satisfies budget {latency_budget:.2f}ms "
                f"(min available: {self.min_latency:.2f}ms). "
                f"Using closest: {tier}_{top_k}_{num_blocks} "
                f"(latency: {closest_latency:.2f}ms, accuracy: {config_info['accuracy']:.4f})"
            )
            return {
                'tier': tier,
                'top_k': top_k,
                'num_active_blocks': num_blocks,
                'accuracy': config_info['accuracy'],
                'latency': config_info['latency'],
                'num_samples': config_info['num_samples'],
            }
        
        # Sort by accuracy (descending)
        valid_configs.sort(key=lambda x: x['accuracy'], reverse=True)
        
        if return_all_candidates:
            return valid_configs
        else:
            # Return best config
            return valid_configs[0]
    
    def get_statistics(self) -> Dict:
        """Get lookup table statistics."""
        if not self.config_table:
            return {}
        
        accuracies = [c['accuracy'] for c in self.config_table.values()]
        latencies = [c['latency'] for c in self.config_table.values()]
        
        # Count configurations by tier, top_k, num_blocks
        tier_counts = defaultdict(int)
        top_k_counts = defaultdict(int)
        num_blocks_counts = defaultdict(int)
        
        for (tier, top_k, num_blocks) in self.config_table.keys():
            tier_counts[tier] += 1
            top_k_counts[top_k] += 1
            num_blocks_counts[num_blocks] += 1
        
        return {
            'num_configs': len(self.config_table),
            'num_lookup_entries': len(self.lookup_table),
            'latency_range': {
                'min': self.min_latency,
                'max': self.max_latency,
            },
            'accuracy_range': {
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies)),
                'mean': float(np.mean(accuracies)),
            },
            'latency_stats': {
                'mean': float(np.mean(latencies)),
                'std': float(np.std(latencies)),
            },
            'config_distribution': {
                'tiers': dict(tier_counts),
                'top_k': dict(top_k_counts),
                'num_active_blocks': dict(num_blocks_counts),
            },
        }
    
    def save(self, output_file: str):
        """Save lookup table to file."""
        output_data = {
            'config_table': {
                f"{tier}_{top_k}_{num_blocks}": config_info
                for (tier, top_k, num_blocks), config_info in self.config_table.items()
            },
            'lookup_table': {
                str(budget): config
                for budget, config in self.lookup_table.items()
            },
            'statistics': self.get_statistics(),
            'metadata': {
                'aggregation_method': self.aggregation_method,
                'tolerance': self.tolerance,
                'use_prefill_only': self.use_prefill_only,
                'num_profiling_results': len(self.profiling_results),
            },
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        log.info(f"Lookup table saved to {output_file}")
    
    @classmethod
    def load(cls, lookup_file: str):
        """Load lookup table from file."""
        with open(lookup_file, 'r') as f:
            data = json.load(f)
        
        # Reconstruct config_table
        config_table = {}
        for key_str, config_info in data['config_table'].items():
            tier, top_k, num_blocks = key_str.split('_')
            config_table[(tier, int(top_k), int(num_blocks))] = config_info
        
        # Reconstruct lookup_table
        lookup_table = {
            float(budget): config
            for budget, config in data['lookup_table'].items()
        }
        
        # Reconstruct controller
        controller = cls.__new__(cls)
        controller.config_table = config_table
        controller.lookup_table = lookup_table
        controller.aggregation_method = data.get('metadata', {}).get('aggregation_method', 'mean')
        controller.tolerance = data.get('metadata', {}).get('tolerance', 0.05)
        controller.use_prefill_only = data.get('metadata', {}).get('use_prefill_only', True)  # Default to True for new tables
        controller.profiling_results = []  # Not needed after loading
        
        # Set latency range from statistics
        stats = data.get('statistics', {})
        latency_range = stats.get('latency_range', {})
        controller.min_latency = latency_range.get('min', 0.0)
        controller.max_latency = latency_range.get('max', 0.0)
        
        return controller


def build_lookup_table_from_core_exp(
    results_dir: str,
    dataset_names: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    aggregation_method: str = "mean",
    tolerance: float = 0.05,
    use_prefill_only: bool = True,  # Use only prefill latency (not including decode)
) -> LookupTableBaselineController:
    """
    Build lookup table controller from core_exp profiling results.
    
    Args:
        results_dir: Directory containing core_exp results (e.g., "./results/core_exp_h100")
        dataset_names: List of dataset names to include. If None, uses all available.
        output_file: Path to save lookup table. If None, doesn't save.
        aggregation_method: How to aggregate multiple samples ("mean", "median", "max_accuracy")
        tolerance: Relative tolerance for latency budget matching
    
    Returns:
        LookupTableBaselineController instance
    """
    log.info(f"Loading profiling results from {results_dir}...")
    
    # Load data using CoreExpDataLoader
    loader = CoreExpDataLoader(results_dir)
    
    if dataset_names is None:
        # Auto-detect available datasets
        dataset_names = []
        for dataset_dir in loader.results_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name.replace("-", "_")
                dataset_names.append(dataset_name)
        log.info(f"Auto-detected {len(dataset_names)} datasets: {dataset_names}")
    
    # Load results from all datasets
    all_results = []
    for dataset_name in dataset_names:
        results = loader.load_dataset_results(dataset_name)
        all_results.extend(results)
        log.info(f"Loaded {len(results)} results from {dataset_name}")
    
    log.info(f"Total profiling results: {len(all_results)}")
    
    if not all_results:
        raise ValueError(f"No profiling results found in {results_dir}")
    
    # Build lookup table controller
    controller = LookupTableBaselineController(
        profiling_results=all_results,
        aggregation_method=aggregation_method,
        tolerance=tolerance,
        use_prefill_only=use_prefill_only,
    )
    
    # Print statistics
    stats = controller.get_statistics()
    log.info("Lookup table statistics:")
    log.info(f"  Unique configurations: {stats['num_configs']}")
    log.info(f"  Latency range: [{stats['latency_range']['min']:.2f}ms, {stats['latency_range']['max']:.2f}ms]")
    log.info(f"  Accuracy range: [{stats['accuracy_range']['min']:.4f}, {stats['accuracy_range']['max']:.4f}]")
    log.info(f"  Mean accuracy: {stats['accuracy_range']['mean']:.4f}")
    
    # Save if output file specified
    if output_file:
        controller.save(output_file)
    
    return controller


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build lookup table baseline controller from core_exp results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/core_exp_h100",
        help="Directory containing core_exp profiling results"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./checkpoints/controller/lookup_table_baseline.json",
        help="Path to save lookup table"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="List of dataset names to include (e.g., coco_2014_vqa text_vqa). If not specified, uses all available."
    )
    parser.add_argument(
        "--aggregation_method",
        type=str,
        default="mean",
        choices=["mean", "median", "max_accuracy"],
        help="How to aggregate multiple samples for same configuration"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Relative tolerance for latency budget matching (0.05 = 5%%)"
    )
    parser.add_argument(
        "--use_prefill_only",
        action="store_true",
        default=True,
        help="Use only prefill latency (vision + prefill, excluding decode). Default: True"
    )
    parser.add_argument(
        "--use_total_latency",
        action="store_true",
        help="Use total latency (vision + prefill + decode). Overrides --use_prefill_only"
    )
    
    args = parser.parse_args()
    
    # Determine which latency to use
    use_prefill_only = not args.use_total_latency
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Build lookup table
    controller = build_lookup_table_from_core_exp(
        results_dir=args.results_dir,
        dataset_names=args.datasets,
        output_file=args.output_file,
        aggregation_method=args.aggregation_method,
        tolerance=args.tolerance,
        use_prefill_only=use_prefill_only,
    )
    
    log.info("Lookup table baseline controller built successfully!")

