"""
Build lookup table controller from profiling results.
Simple baseline: latency_budget -> best_configuration.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


class LookupTableController:
    """
    Simple lookup table controller.
    Given latency budget, find configuration that maximizes accuracy.
    """
    
    def __init__(self, profiling_results: List[Dict], num_bins: int = 50):
        """
        Build lookup table from profiling results.
        
        Args:
            profiling_results: List of profiling result dictionaries
            num_bins: Number of latency budget bins
        """
        self.profiling_results = profiling_results
        self.num_bins = num_bins
        self.lookup_table = self._build_lookup_table()
    
    def _build_lookup_table(self) -> Dict[Tuple[float, float], Dict]:
        """
        Build lookup table: (budget_min, budget_max) -> best_config.
        
        For each latency budget range, find config with highest accuracy.
        """
        if not self.profiling_results:
            return {}
        
        # Get latency range
        latencies = [r.get('T_total', 0.0) for r in self.profiling_results if r.get('T_total', 0) > 0]
        if not latencies:
            return {}
        
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Create latency bins
        latency_bins = np.linspace(min_latency, max_latency, self.num_bins + 1)
        
        lookup = {}
        
        for i in range(len(latency_bins) - 1):
            budget_min = latency_bins[i]
            budget_max = latency_bins[i + 1]
            
            # Find configs within this budget range
            valid_configs = [
                r for r in self.profiling_results
                if budget_min <= r.get('T_total', float('inf')) <= budget_max
                and r.get('accuracy', 0) >= 0  # Valid accuracy
            ]
            
            if valid_configs:
                # Select config with highest accuracy
                best_config = max(valid_configs, key=lambda x: x.get('accuracy', 0))
                
                lookup[(budget_min, budget_max)] = {
                    'tier': best_config.get('tier', 'medium'),
                    'top_k': best_config.get('top_k', 8),
                    'num_blocks': best_config.get('num_blocks', 16),
                    'selected_blocks': best_config.get('selected_blocks', list(range(16))),
                    'accuracy': best_config.get('accuracy', 0.0),
                    'latency': best_config.get('T_total', 0.0),
                }
        
        return lookup
    
    def predict(self, latency_budget: float) -> Optional[Dict]:
        """
        Predict configuration for given latency budget.
        
        Args:
            latency_budget: Target latency budget in ms
        
        Returns:
            config: Configuration dictionary, or None if not found
        """
        # Find matching bin
        for (budget_min, budget_max), config in self.lookup_table.items():
            if budget_min <= latency_budget <= budget_max:
                return config
        
        # Fallback: find closest config
        return self._find_closest_config(latency_budget)
    
    def _find_closest_config(self, latency_budget: float) -> Optional[Dict]:
        """Find configuration with latency closest to budget."""
        if not self.profiling_results:
            return None
        
        # Find config with latency closest to budget
        closest_config = min(
            self.profiling_results,
            key=lambda x: abs(x.get('T_total', float('inf')) - latency_budget)
        )
        
        return {
            'tier': closest_config.get('tier', 'medium'),
            'top_k': closest_config.get('top_k', 8),
            'num_blocks': closest_config.get('num_blocks', 16),
            'selected_blocks': closest_config.get('selected_blocks', list(range(16))),
            'accuracy': closest_config.get('accuracy', 0.0),
            'latency': closest_config.get('T_total', 0.0),
        }
    
    def get_statistics(self) -> Dict:
        """Get lookup table statistics."""
        return {
            'num_bins': self.num_bins,
            'num_entries': len(self.lookup_table),
            'coverage': len(self.lookup_table) / self.num_bins if self.num_bins > 0 else 0.0,
        }
    
    def save(self, output_file: str):
        """Save lookup table to file."""
        output_data = {
            'lookup_table': {
                f"{min_val}_{max_val}": config
                for (min_val, max_val), config in self.lookup_table.items()
            },
            'statistics': self.get_statistics(),
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        log.info(f"Lookup table saved to {output_file}")
    
    @classmethod
    def load(cls, lookup_file: str):
        """Load lookup table from file."""
        with open(lookup_file, 'r') as f:
            data = json.load(f)
        
        # Reconstruct lookup table
        lookup_table = {}
        for key, config in data['lookup_table'].items():
            min_val, max_val = map(float, key.split('_'))
            lookup_table[(min_val, max_val)] = config
        
        controller = cls.__new__(cls)
        controller.lookup_table = lookup_table
        controller.num_bins = data.get('statistics', {}).get('num_bins', 50)
        controller.profiling_results = []  # Not needed after loading
        
        return controller


def build_lookup_table(
    profiling_results_file: str,
    output_file: str,
    num_bins: int = 50,
):
    """
    Build and save lookup table from profiling results.
    
    Args:
        profiling_results_file: Path to profiling results JSON file
        output_file: Path to save lookup table
        num_bins: Number of latency budget bins
    """
    log.info(f"Loading profiling results from {profiling_results_file}...")
    
    with open(profiling_results_file, 'r') as f:
        profiling_results = json.load(f)
    
    log.info(f"Loaded {len(profiling_results)} profiling results")
    
    # Build lookup table
    controller = LookupTableController(profiling_results, num_bins=num_bins)
    
    # Save
    controller.save(output_file)
    
    # Print statistics
    stats = controller.get_statistics()
    log.info(f"Lookup table statistics:")
    log.info(f"  Number of bins: {stats['num_bins']}")
    log.info(f"  Number of entries: {stats['num_entries']}")
    log.info(f"  Coverage: {stats['coverage']:.2%}")
    
    return controller


def main():
    parser = argparse.ArgumentParser(
        description="Build lookup table controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--profiling_results",
        type=str,
        required=True,
        help="Path to profiling results JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="checkpoints/controller/lookup_table.json",
        help="Path to save lookup table"
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=50,
        help="Number of latency budget bins"
    )
    
    args = parser.parse_args()
    
    build_lookup_table(
        profiling_results_file=args.profiling_results,
        output_file=args.output_file,
        num_bins=args.num_bins,
    )


if __name__ == "__main__":
    main()





