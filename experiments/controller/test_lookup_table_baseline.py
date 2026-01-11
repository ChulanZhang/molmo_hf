"""
Test script for Lookup Table Baseline Controller.

This script tests the lookup table baseline controller with sample data or real profiling results.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.controller.lookup_table_baseline import (
    LookupTableBaselineController,
    build_lookup_table_from_core_exp,
)
from experiments.controller.lookup_table_wrapper import create_lookup_table_controller

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def test_with_sample_data():
    """Test lookup table controller with sample data."""
    log.info("=" * 80)
    log.info("Test 1: Lookup Table with Sample Data")
    log.info("=" * 80)
    
    # Create sample profiling results
    sample_results = [
        {
            'tier': 'low',
            'top_k': 4,
            'num_active_blocks': 12,
            'accuracy': 0.70,
            'T_total': 120.0,
        },
        {
            'tier': 'low',
            'top_k': 6,
            'num_active_blocks': 12,
            'accuracy': 0.75,
            'T_total': 140.0,
        },
        {
            'tier': 'medium',
            'top_k': 8,
            'num_active_blocks': 14,
            'accuracy': 0.80,
            'T_total': 200.0,
        },
        {
            'tier': 'medium',
            'top_k': 8,
            'num_active_blocks': 16,
            'accuracy': 0.85,
            'T_total': 250.0,
        },
        {
            'tier': 'high',
            'top_k': 12,
            'num_active_blocks': 16,
            'accuracy': 0.90,
            'T_total': 350.0,
        },
    ]
    
    # Create controller
    controller = LookupTableBaselineController(
        profiling_results=sample_results,
        aggregation_method="mean",
        tolerance=0.05,
    )
    
    # Test predictions
    test_budgets = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0]
    
    log.info("\nTesting predictions:")
    for budget in test_budgets:
        config = controller.predict(budget)
        if config:
            log.info(f"  Budget {budget:6.1f}ms -> Tier: {config['tier']:6s}, "
                    f"Top-K: {config['top_k']:2d}, Blocks: {config['num_active_blocks']:2d}, "
                    f"Acc: {config['accuracy']:.3f}, Lat: {config['latency']:.1f}ms")
        else:
            log.warning(f"  Budget {budget:6.1f}ms -> No valid configuration")
    
    # Print statistics
    stats = controller.get_statistics()
    log.info("\nStatistics:")
    log.info(f"  Unique configurations: {stats['num_configs']}")
    log.info(f"  Latency range: [{stats['latency_range']['min']:.1f}ms, {stats['latency_range']['max']:.1f}ms]")
    log.info(f"  Accuracy range: [{stats['accuracy_range']['min']:.3f}, {stats['accuracy_range']['max']:.3f}]")
    
    log.info("\n✓ Test 1 passed!\n")


def test_with_real_data(results_dir: str):
    """Test lookup table controller with real profiling results."""
    log.info("=" * 80)
    log.info("Test 2: Lookup Table with Real Profiling Results")
    log.info("=" * 80)
    
    results_path = Path(results_dir)
    if not results_path.exists():
        log.warning(f"Results directory not found: {results_dir}")
        log.warning("Skipping Test 2")
        return
    
    try:
        # Build controller from real data
        controller = build_lookup_table_from_core_exp(
            results_dir=results_dir,
            dataset_names=None,  # Use all available datasets
            output_file=None,  # Don't save for this test
            aggregation_method="mean",
            tolerance=0.05,
        )
        
        # Test predictions
        test_budgets = [150.0, 200.0, 250.0, 300.0, 350.0]
        
        log.info("\nTesting predictions:")
        for budget in test_budgets:
            config = controller.predict(budget)
            if config:
                log.info(f"  Budget {budget:6.1f}ms -> Tier: {config['tier']:6s}, "
                        f"Top-K: {config['top_k']:2d}, Blocks: {config['num_active_blocks']:2d}, "
                        f"Acc: {config['accuracy']:.3f}, Lat: {config['latency']:.1f}ms")
            else:
                log.warning(f"  Budget {budget:6.1f}ms -> No valid configuration")
        
        # Print statistics
        stats = controller.get_statistics()
        log.info("\nStatistics:")
        log.info(f"  Unique configurations: {stats['num_configs']}")
        log.info(f"  Latency range: [{stats['latency_range']['min']:.1f}ms, {stats['latency_range']['max']:.1f}ms]")
        log.info(f"  Accuracy range: [{stats['accuracy_range']['min']:.3f}, {stats['accuracy_range']['max']:.3f}]")
        log.info(f"  Mean accuracy: {stats['accuracy_range']['mean']:.3f}")
        
        # Test all candidates
        log.info("\nTesting all candidates for budget 200ms:")
        all_configs = controller.predict(200.0, return_all_candidates=True)
        log.info(f"  Found {len(all_configs)} valid configurations:")
        for i, config in enumerate(all_configs[:5]):  # Show top 5
            log.info(f"    {i+1}. Tier: {config['tier']:6s}, Top-K: {config['top_k']:2d}, "
                    f"Blocks: {config['num_active_blocks']:2d}, "
                    f"Acc: {config['accuracy']:.3f}, Lat: {config['latency']:.1f}ms")
        
        log.info("\n✓ Test 2 passed!\n")
        
    except Exception as e:
        log.error(f"Error in Test 2: {e}")
        import traceback
        traceback.print_exc()


def test_wrapper():
    """Test lookup table wrapper."""
    log.info("=" * 80)
    log.info("Test 3: Lookup Table Wrapper")
    log.info("=" * 80)
    
    # Create sample data
    sample_results = [
        {
            'tier': 'medium',
            'top_k': 8,
            'num_active_blocks': 16,
            'accuracy': 0.85,
            'T_total': 200.0,
        },
    ]
    
    controller_base = LookupTableBaselineController(
        profiling_results=sample_results,
        aggregation_method="mean",
        tolerance=0.05,
    )
    
    # Test wrapper
    from experiments.controller.lookup_table_wrapper import LookupTableControllerWrapper
    wrapper = LookupTableControllerWrapper(controller_base)
    
    # Test methods
    config = wrapper.predict_all(200.0)
    log.info(f"predict_all(200.0): {config}")
    
    stage1 = wrapper.predict_stage1(200.0)
    log.info(f"predict_stage1(200.0): {stage1}")
    
    stage2 = wrapper.predict_stage2(200.0)
    log.info(f"predict_stage2(200.0): {stage2}")
    
    log.info("\n✓ Test 3 passed!\n")


def test_save_and_load():
    """Test save and load functionality."""
    log.info("=" * 80)
    log.info("Test 4: Save and Load")
    log.info("=" * 80)
    
    # Create sample data
    sample_results = [
        {
            'tier': 'medium',
            'top_k': 8,
            'num_active_blocks': 16,
            'accuracy': 0.85,
            'T_total': 200.0,
        },
    ]
    
    # Create and save
    controller1 = LookupTableBaselineController(
        profiling_results=sample_results,
        aggregation_method="mean",
        tolerance=0.05,
    )
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        controller1.save(temp_path)
        log.info(f"Saved to {temp_path}")
        
        # Load
        controller2 = LookupTableBaselineController.load(temp_path)
        log.info(f"Loaded from {temp_path}")
        
        # Test that predictions match
        config1 = controller1.predict(200.0)
        config2 = controller2.predict(200.0)
        
        assert config1['tier'] == config2['tier']
        assert config1['top_k'] == config2['top_k']
        assert config1['num_active_blocks'] == config2['num_active_blocks']
        
        log.info("Predictions match after save/load!")
        log.info("\n✓ Test 4 passed!\n")
        
    finally:
        # Clean up
        Path(temp_path).unlink(missing_ok=True)


def main():
    """Run all tests."""
    log.info("Starting Lookup Table Baseline Controller Tests\n")
    
    # Test 1: Sample data
    test_with_sample_data()
    
    # Test 2: Real data (if available)
    results_dir = "./results/core_exp_h100"
    test_with_real_data(results_dir)
    
    # Test 3: Wrapper
    test_wrapper()
    
    # Test 4: Save and load
    test_save_and_load()
    
    log.info("=" * 80)
    log.info("All tests completed!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

