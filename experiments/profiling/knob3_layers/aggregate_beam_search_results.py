#!/usr/bin/env python3
"""
Script to aggregate individual beam search result files into the final summary.
This is useful when the distributed collection failed but individual files exist.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

def aggregate_results(results_dir: Path) -> bool:
    """Aggregate individual beam search result files into final summary.
    
    Args:
        results_dir: Directory containing individual result files
        
    Returns:
        bool: True if aggregation was successful
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return False
    
    # Find all individual beam search result files
    beam_search_files = sorted(results_dir.glob("beam_search_step*_blocks*_removed*.json"))
    
    if not beam_search_files:
        print(f"Warning: No beam search result files found in {results_dir}")
        return False
    
    print(f"Found {len(beam_search_files)} individual result files")
    
    # Load all results
    all_results = []
    for result_file in beam_search_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                all_results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {result_file}: {e}")
            continue
    
    if not all_results:
        print("Error: No valid results loaded")
        return False
    
    print(f"Loaded {len(all_results)} results")
    
    # Check if final results file exists
    final_results_file = results_dir / "exp3_accuracy_sensitivity_v2_results.json"
    
    if final_results_file.exists():
        # Load existing file to preserve other fields
        try:
            with open(final_results_file, 'r') as f:
                final_results = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load existing results file: {e}, creating new one")
            final_results = {}
    else:
        final_results = {}
    
    # Update summary with aggregated results
    final_results["summary"] = all_results
    
    # Ensure config exists
    if "config" not in final_results:
        # Try to infer from first result
        if all_results:
            first_result = all_results[0]
            final_results["config"] = {
                "dataset_name": "unknown",
                "split": "unknown",
                "batch_size": 16,
                "max_new_tokens": 16,
                "num_samples": first_result.get("num_samples", 0),
                "beam_width": 3,
                "max_blocks_to_remove": 4,
                "total_blocks": first_result.get("num_total_blocks", 16),
                "world_size": 1,
            }
    
    # Save updated results
    try:
        with open(final_results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"✅ Successfully aggregated {len(all_results)} results into {final_results_file}")
        return True
    except Exception as e:
        print(f"Error: Failed to save aggregated results: {e}")
        return False


def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("Usage: python aggregate_beam_search_results.py <results_dir>")
        print("Example: python aggregate_beam_search_results.py results/profiling/exp3_beam_search_multi_dataset/text-vqa/train+validation")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    
    if aggregate_results(results_dir):
        print("✅ Aggregation completed successfully")
        sys.exit(0)
    else:
        print("❌ Aggregation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()


