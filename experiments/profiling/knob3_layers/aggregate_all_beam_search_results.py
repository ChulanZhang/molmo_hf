#!/usr/bin/env python3
"""
Script to aggregate all beam search results across all datasets.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import subprocess

def aggregate_results(results_dir: Path) -> bool:
    """Aggregate individual beam search result files into final summary."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return False
    
    beam_search_files = sorted(results_dir.glob("beam_search_step*_blocks*_removed*.json"))
    
    if not beam_search_files:
        return False
    
    all_results = []
    for result_file in beam_search_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                all_results.append(result)
        except Exception as e:
            continue
    
    if not all_results:
        return False
    
    final_results_file = results_dir / "exp3_accuracy_sensitivity_v2_results.json"
    
    if final_results_file.exists():
        try:
            with open(final_results_file, 'r') as f:
                final_results = json.load(f)
        except Exception as e:
            final_results = {}
    else:
        final_results = {}
    
    final_results["summary"] = all_results
    
    if "config" not in final_results:
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
    
    try:
        with open(final_results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        return True
    except Exception as e:
        return False


def main():
    """Main execution"""
    base_dir = Path("results/profiling/exp3_beam_search_multi_dataset")
    
    # Dataset directories
    datasets = [
        "coco-2014-vqa/train+validation",
        "text-vqa/train+validation",
        "okvqa/train+validation",
        "science-qa-img/train+validation",
        "st-qa/train+validation",
        "doc-qa/train+validation",
        "tally-qa/train+test",
        "coco-caption/train+validation",
    ]
    
    success_count = 0
    fail_count = 0
    
    for dataset_path in datasets:
        results_dir = base_dir / dataset_path
        if aggregate_results(results_dir):
            print(f"✅ {dataset_path}: Aggregated successfully")
            success_count += 1
        else:
            print(f"⚠️  {dataset_path}: No results to aggregate or already complete")
            fail_count += 1
    
    print(f"\n{'='*80}")
    print(f"✅ Successfully aggregated: {success_count} dataset(s)")
    print(f"⚠️  Skipped or failed: {fail_count} dataset(s)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


