"""
Wrapper for LookupTableBaselineController to make it compatible with existing controller interface.

This wrapper allows the lookup table baseline to be used in the same way as the GRPO-trained controller,
but it only uses latency budget (no vision/language features needed).
"""

import logging
from typing import Dict, Optional, Any
import torch

from experiments.controller.lookup_table_baseline import LookupTableBaselineController

log = logging.getLogger(__name__)


class LookupTableControllerWrapper:
    """
    Wrapper for LookupTableBaselineController to match GRPO controller interface.
    
    This allows the lookup table baseline to be used as a drop-in replacement
    for the trained GRPO controller in inference pipelines.
    """
    
    def __init__(self, lookup_table_controller: LookupTableBaselineController):
        """
        Initialize wrapper.
        
        Args:
            lookup_table_controller: LookupTableBaselineController instance
        """
        self.controller = lookup_table_controller
        self.device = "cuda"  # Not used, but kept for compatibility
    
    def forward_stage1(
        self,
        lang_feat: torch.Tensor,
        budget_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 1 forward (for compatibility).
        
        Note: Lookup table doesn't use lang_feat or budget_feat.
        This method is kept for interface compatibility but doesn't do anything.
        
        Args:
            lang_feat: Language features (ignored)
            budget_feat: Budget features (ignored)
        
        Returns:
            Empty dict (for compatibility)
        """
        # Lookup table doesn't need features, but we return empty dict for compatibility
        return {}
    
    def forward_stage2(
        self,
        vision_feat: torch.Tensor,
        lang_feat: torch.Tensor,
        budget_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 2 forward (for compatibility).
        
        Note: Lookup table doesn't use vision_feat, lang_feat, or budget_feat.
        This method is kept for interface compatibility but doesn't do anything.
        
        Args:
            vision_feat: Vision features (ignored)
            lang_feat: Language features (ignored)
            budget_feat: Budget features (ignored)
        
        Returns:
            Empty dict (for compatibility)
        """
        return {}
    
    def predict(
        self,
        latency_budget: float,
        return_all_candidates: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict configuration for given latency budget.
        
        This is the main method to use for lookup table baseline.
        
        Args:
            latency_budget: Target latency budget in ms
            return_all_candidates: If True, return all valid configs sorted by accuracy
        
        Returns:
            Configuration dictionary with:
                - tier: str ("low", "medium", "high")
                - top_k: int
                - num_active_blocks: int
                - accuracy: float (expected)
                - latency: float (expected)
        """
        return self.controller.predict(
            latency_budget=latency_budget,
            return_all_candidates=return_all_candidates,
        )
    
    def predict_stage1(
        self,
        latency_budget: float,
    ) -> Dict[str, Any]:
        """
        Predict Stage 1 configuration (Knob1: tier).
        
        Args:
            latency_budget: Target latency budget in ms
        
        Returns:
            Dict with 'tier' key
        """
        config = self.controller.predict(latency_budget)
        if config:
            return {'tier': config['tier']}
        return {'tier': 'medium'}  # Default fallback
    
    def predict_stage2(
        self,
        latency_budget: float,
    ) -> Dict[str, Any]:
        """
        Predict Stage 2 configuration (Knob2: top_k, Knob3: num_active_blocks).
        
        Args:
            latency_budget: Target latency budget in ms
        
        Returns:
            Dict with 'top_k' and 'num_active_blocks' keys
        """
        config = self.controller.predict(latency_budget)
        if config:
            return {
                'top_k': config['top_k'],
                'num_active_blocks': config['num_active_blocks'],
            }
        return {'top_k': 8, 'num_active_blocks': 16}  # Default fallback
    
    def predict_all(
        self,
        latency_budget: float,
    ) -> Dict[str, Any]:
        """
        Predict all knobs for given latency budget.
        
        Args:
            latency_budget: Target latency budget in ms
        
        Returns:
            Dict with 'tier', 'top_k', 'num_active_blocks', 'accuracy', 'latency'
        """
        return self.controller.predict(latency_budget)
    
    def get_statistics(self) -> Dict:
        """Get lookup table statistics."""
        return self.controller.get_statistics()
    
    def to(self, device: str):
        """Move to device (for compatibility, no-op for lookup table)."""
        self.device = device
        return self
    
    def eval(self):
        """Set to eval mode (for compatibility, no-op for lookup table)."""
        return self


def create_lookup_table_controller(
    lookup_table_path: Optional[str] = None,
    results_dir: Optional[str] = None,
    dataset_names: Optional[list] = None,
    aggregation_method: str = "mean",
    tolerance: float = 0.05,
) -> LookupTableControllerWrapper:
    """
    Create lookup table controller wrapper.
    
    Either load from saved file or build from profiling results.
    
    Args:
        lookup_table_path: Path to saved lookup table JSON file
        results_dir: Directory containing core_exp profiling results (if not loading from file)
        dataset_names: List of dataset names to include (if building from results)
        aggregation_method: How to aggregate samples ("mean", "median", "max_accuracy")
        tolerance: Relative tolerance for latency budget matching
    
    Returns:
        LookupTableControllerWrapper instance
    """
    if lookup_table_path:
        # Load from file
        log.info(f"Loading lookup table from {lookup_table_path}...")
        controller = LookupTableBaselineController.load(lookup_table_path)
    elif results_dir:
        # Build from profiling results
        from experiments.controller.lookup_table_baseline import build_lookup_table_from_core_exp
        log.info(f"Building lookup table from {results_dir}...")
        controller = build_lookup_table_from_core_exp(
            results_dir=results_dir,
            dataset_names=dataset_names,
            aggregation_method=aggregation_method,
            tolerance=tolerance,
        )
    else:
        raise ValueError("Must provide either lookup_table_path or results_dir")
    
    return LookupTableControllerWrapper(controller)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test lookup table baseline controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--lookup_table_path",
        type=str,
        default=None,
        help="Path to saved lookup table JSON file"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/core_exp_h100",
        help="Directory containing core_exp profiling results (if not loading from file)"
    )
    parser.add_argument(
        "--test_budgets",
        type=float,
        nargs="+",
        default=[150.0, 200.0, 250.0, 300.0, 350.0],
        help="Latency budgets to test (in ms)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Create controller
    controller = create_lookup_table_controller(
        lookup_table_path=args.lookup_table_path,
        results_dir=args.results_dir if not args.lookup_table_path else None,
    )
    
    # Test predictions
    log.info("\n" + "=" * 80)
    log.info("Testing Lookup Table Baseline Controller")
    log.info("=" * 80)
    
    for budget in args.test_budgets:
        config = controller.predict(budget)
        if config:
            log.info(f"\nLatency Budget: {budget:.2f}ms")
            log.info(f"  Tier: {config['tier']}")
            log.info(f"  Top-K: {config['top_k']}")
            log.info(f"  Num Active Blocks: {config['num_active_blocks']}")
            log.info(f"  Expected Accuracy: {config['accuracy']:.4f}")
            log.info(f"  Expected Latency: {config['latency']:.2f}ms")
        else:
            log.warning(f"No valid configuration found for budget {budget:.2f}ms")
    
    # Print statistics
    stats = controller.get_statistics()
    log.info("\n" + "=" * 80)
    log.info("Lookup Table Statistics")
    log.info("=" * 80)
    log.info(f"Unique configurations: {stats['num_configs']}")
    log.info(f"Latency range: [{stats['latency_range']['min']:.2f}ms, {stats['latency_range']['max']:.2f}ms]")
    log.info(f"Accuracy range: [{stats['accuracy_range']['min']:.4f}, {stats['accuracy_range']['max']:.4f}]")
    log.info(f"Mean accuracy: {stats['accuracy_range']['mean']:.4f}")

