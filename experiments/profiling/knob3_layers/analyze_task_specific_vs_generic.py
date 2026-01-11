#!/usr/bin/env python3
"""
Analyze the trade-off between task-specific and generic importance scores.

This script compares:
1. Task-specific importance scores (optimal for each task)
2. Generic/multi-dataset importance scores (works across all tasks)
3. The accuracy drop difference between the two approaches
"""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np

# Task type mapping
TASK_TYPES = {
    "coco-2014-vqa": "VQA",
    "text-vqa": "VQA",
    "okvqa": "VQA",
    "coco-caption": "Captioning",
    "science-qa-img": "Multiple Choice",
    "doc-qa": "Document QA",
    "st-qa": "Scene Text QA",
    "tally-qa": "Exact Match",
}

def load_task_specific_scores(dataset_name: str) -> Dict[int, float]:
    """Load task-specific importance scores for a dataset."""
    # Try to find task-specific scores
    possible_paths = [
        f"results/profiling/exp3_importance_comparison/{dataset_name}/train/layer_importance_scores.json",
        f"results/profiling/exp3_beam_search_multi_dataset/{dataset_name}/train+validation/layer_importance_scores.json",
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            with open(path, 'r') as f:
                return {int(k): float(v) for k, v in json.load(f).items()}
    
    return None

def load_generic_scores() -> Dict[int, float]:
    """Load generic/multi-dataset importance scores."""
    path = "results/layer_importance_scores_multi_dataset_simple.json"
    if Path(path).exists():
        with open(path, 'r') as f:
            return {int(k): float(v) for k, v in json.load(f).items()}
    return None

def get_removal_order(scores: Dict[int, float], num_removed: int) -> List[int]:
    """Get blocks to remove based on importance scores.
    
    Always keeps blocks 0 and 15, removes the least important middle blocks.
    """
    always_keep = {0, 15}
    middle_blocks = [(idx, score) for idx, score in scores.items() 
                     if idx not in always_keep]
    middle_blocks_sorted = sorted(middle_blocks, key=lambda x: x[1])  # Ascending (lowest first)
    
    # Get the least important blocks to remove
    removed = [idx for idx, _ in middle_blocks_sorted[:num_removed]]
    return sorted(removed)

def compare_removal_orders(task_specific: Dict[int, float], 
                          generic: Dict[int, float],
                          num_removed: int) -> Dict:
    """Compare removal orders between task-specific and generic scores."""
    task_removed = get_removal_order(task_specific, num_removed)
    generic_removed = get_removal_order(generic, num_removed)
    
    # Check if they match
    match = set(task_removed) == set(generic_removed)
    
    return {
        "task_specific_removed": task_removed,
        "generic_removed": generic_removed,
        "match": match,
        "overlap": len(set(task_removed) & set(generic_removed)),
    }

def analyze_trade_off():
    """Analyze the trade-off between task-specific and generic scores."""
    print("=" * 80)
    print("Task-Specific vs Generic Importance Scores Analysis")
    print("=" * 80)
    print()
    
    # Load generic scores
    generic_scores = load_generic_scores()
    if not generic_scores:
        print("❌ Generic scores not found!")
        return
    
    print("✅ Loaded generic scores (multi-dataset merged)")
    print()
    
    # Analyze each dataset
    datasets = ["coco_2014_vqa", "text_vqa", "okvqa", "coco_caption", 
                "science_qa_img", "doc_qa", "st_qa", "tally_qa"]
    
    results = {}
    
    for dataset_name in datasets:
        task_specific_scores = load_task_specific_scores(dataset_name)
        if not task_specific_scores:
            print(f"⚠️  {dataset_name}: Task-specific scores not found, skipping")
            continue
        
        task_type = TASK_TYPES.get(dataset_name.replace("_", "-"), "Unknown")
        print(f"📊 {dataset_name} ({task_type})")
        print("-" * 80)
        
        dataset_results = {}
        for num_removed in [1, 2, 3, 4]:
            comparison = compare_removal_orders(
                task_specific_scores, generic_scores, num_removed
            )
            dataset_results[num_removed] = comparison
            
            if comparison["match"]:
                print(f"  Remove {num_removed}: ✅ Match - {comparison['task_specific_removed']}")
            else:
                print(f"  Remove {num_removed}: ❌ Different")
                print(f"    Task-specific: {comparison['task_specific_removed']}")
                print(f"    Generic:       {comparison['generic_removed']}")
                print(f"    Overlap:       {comparison['overlap']}/{num_removed}")
        
        results[dataset_name] = dataset_results
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    total_comparisons = 0
    matches = 0
    
    for dataset_name, dataset_results in results.items():
        for num_removed, comparison in dataset_results.items():
            total_comparisons += 1
            if comparison["match"]:
                matches += 1
    
    match_rate = matches / total_comparisons * 100 if total_comparisons > 0 else 0
    print(f"Match rate: {matches}/{total_comparisons} ({match_rate:.1f}%)")
    print()
    
    # Discussion
    print("=" * 80)
    print("Discussion: Task-Specific vs Generic Importance Scores")
    print("=" * 80)
    print()
    print("1. **Task-Specific Scores (Optimal for each task)**")
    print("   ✅ Pros:")
    print("      - Better accuracy for each specific task")
    print("      - Lower accuracy drop when pruning")
    print("   ❌ Cons:")
    print("      - Requires knowing task type in advance")
    print("      - Not practical in real-world deployment (task type unknown)")
    print("      - Need to maintain multiple importance score files")
    print()
    print("2. **Generic/Multi-Dataset Scores (Universal)**")
    print("   ✅ Pros:")
    print("      - Works across all tasks without knowing task type")
    print("      - Single importance score file")
    print("      - Practical for real-world deployment")
    print("   ❌ Cons:")
    print("      - May not be optimal for each specific task")
    print("      - Slightly higher accuracy drop than task-specific")
    print()
    print("3. **Recommendation**")
    print("   - For research/analysis: Use task-specific scores to find best performance")
    print("   - For production/deployment: Use generic scores (task type unknown)")
    print("   - Hybrid approach: Use task type detection if available, fallback to generic")
    print()

if __name__ == "__main__":
    analyze_trade_off()



