"""
Analyze Pareto frontier overlap across different datasets and devices.
"""

import json
import os
from typing import Dict, List, Set, Tuple
from collections import defaultdict


def parse_pareto_info(filepath: str) -> Set[Tuple[int, int, int]]:
    """Parse pareto frontier info file and extract configurations."""
    pareto_configs = set()
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    in_pareto_section = False
    for line in lines:
        if "Pareto Frontier Points" in line:
            in_pareto_section = True
            continue
        
        if in_pareto_section and line.strip() and not line.startswith("-"):
            # Parse lines like "1. max_crops=2, top_k=4, num_active_blocks=12"
            if "max_crops=" in line:
                try:
                    parts = line.split("max_crops=")[1].split(",")
                    max_crops = int(parts[0].strip())
                    top_k = int(parts[1].split("top_k=")[1].strip())
                    num_blocks = int(parts[2].split("num_active_blocks=")[1].strip())
                    pareto_configs.add((max_crops, top_k, num_blocks))
                except (ValueError, IndexError):
                    continue
    
    return pareto_configs


def compute_overlap(set1: Set[Tuple[int, int, int]], set2: Set[Tuple[int, int, int]]) -> Dict[str, float]:
    """Compute overlap metrics between two sets of configurations."""
    intersection = set1 & set2
    union = set1 | set2
    
    if len(union) == 0:
        return {"jaccard": 0.0, "overlap_ratio_1": 0.0, "overlap_ratio_2": 0.0, "intersection_size": 0}
    
    jaccard = len(intersection) / len(union)
    overlap_ratio_1 = len(intersection) / len(set1) if len(set1) > 0 else 0.0
    overlap_ratio_2 = len(intersection) / len(set2) if len(set2) > 0 else 0.0
    
    return {
        "jaccard": jaccard,
        "overlap_ratio_1": overlap_ratio_1,
        "overlap_ratio_2": overlap_ratio_2,
        "intersection_size": len(intersection),
        "set1_size": len(set1),
        "set2_size": len(set2)
    }


def main():
    base_dir = "/home/x-pwang1/ai_project/molmo_hf/results/profiling"
    
    datasets = [
        "doc-qa",
        "okvqa",
        "science-qa-img",
        "st-qa",
        "tally-qa",
        "text-vqa",
        "vqa2",
    ]
    
    # Load pareto frontiers
    pareto_frontiers = {}
    
    # Server datasets
    for dataset in datasets:
        info_file = os.path.join(base_dir, f"exp6_latency_{dataset}", "figures", "pareto_frontier_total_latency_pareto_info.txt")
        if os.path.exists(info_file):
            pareto_frontiers[f"server_{dataset}"] = parse_pareto_info(info_file)
            print(f"Loaded {dataset}: {len(pareto_frontiers[f'server_{dataset}'])} pareto points")
    
    # Orin (VQA2)
    orin_file = os.path.join(base_dir, "orin", "exp6_latency", "figures", "pareto_frontier_total_latency_pareto_info.txt")
    if os.path.exists(orin_file):
        pareto_frontiers["orin_vqa2"] = parse_pareto_info(orin_file)
        print(f"Loaded Orin VQA2: {len(pareto_frontiers['orin_vqa2'])} pareto points")
    
    # Analysis 1: Server vs Orin for VQA2
    print("\n" + "="*80)
    print("Analysis 1: Server vs Orin for VQA2")
    print("="*80)
    
    if "server_vqa2" in pareto_frontiers and "orin_vqa2" in pareto_frontiers:
        server_vqa2 = pareto_frontiers["server_vqa2"]
        orin_vqa2 = pareto_frontiers["orin_vqa2"]
        
        overlap = compute_overlap(server_vqa2, orin_vqa2)
        
        print(f"Server VQA2 pareto points: {overlap['set1_size']}")
        print(f"Orin VQA2 pareto points: {overlap['set2_size']}")
        print(f"Common pareto points: {overlap['intersection_size']}")
        print(f"Jaccard similarity: {overlap['jaccard']:.3f}")
        print(f"Overlap ratio (server): {overlap['overlap_ratio_1']:.3f}")
        print(f"Overlap ratio (orin): {overlap['overlap_ratio_2']:.3f}")
        
        if server_vqa2 == orin_vqa2:
            print("\n✓ Server and Orin have IDENTICAL pareto frontiers!")
        else:
            print("\n✗ Server and Orin have DIFFERENT pareto frontiers")
            print(f"  Only in server: {server_vqa2 - orin_vqa2}")
            print(f"  Only in orin: {orin_vqa2 - server_vqa2}")
    
    # Analysis 2: Cross-dataset overlap on server
    print("\n" + "="*80)
    print("Analysis 2: Cross-dataset Pareto Frontier Overlap (Server)")
    print("="*80)
    
    server_datasets = [d for d in datasets if f"server_{d}" in pareto_frontiers]
    
    overlap_matrix = {}
    for i, dataset1 in enumerate(server_datasets):
        for dataset2 in server_datasets[i+1:]:
            key = f"{dataset1}_vs_{dataset2}"
            overlap = compute_overlap(
                pareto_frontiers[f"server_{dataset1}"],
                pareto_frontiers[f"server_{dataset2}"]
            )
            overlap_matrix[key] = overlap
    
    # Print overlap matrix
    print("\nJaccard Similarity Matrix:")
    print(f"{'Dataset':<20}", end="")
    for d in server_datasets:
        print(f"{d:<15}", end="")
    print()
    
    for i, d1 in enumerate(server_datasets):
        print(f"{d1:<20}", end="")
        for j, d2 in enumerate(server_datasets):
            if i == j:
                print(f"{'1.000':<15}", end="")
            elif i < j:
                key = f"{d1}_vs_{d2}"
                print(f"{overlap_matrix[key]['jaccard']:.3f}", end="")
                print(" " * (15 - len(f"{overlap_matrix[key]['jaccard']:.3f}")), end="")
            else:
                key = f"{d2}_vs_{d1}"
                print(f"{overlap_matrix[key]['jaccard']:.3f}", end="")
                print(" " * (15 - len(f"{overlap_matrix[key]['jaccard']:.3f}")), end="")
        print()
    
    # Find most and least similar pairs
    sorted_overlaps = sorted(overlap_matrix.items(), key=lambda x: x[1]['jaccard'], reverse=True)
    
    print("\nMost Similar Dataset Pairs (Top 5):")
    for key, metrics in sorted_overlaps[:5]:
        print(f"  {key}: Jaccard={metrics['jaccard']:.3f}, Common={metrics['intersection_size']}/{metrics['set1_size']} vs {metrics['set2_size']}")
    
    print("\nLeast Similar Dataset Pairs (Bottom 5):")
    for key, metrics in sorted_overlaps[-5:]:
        print(f"  {key}: Jaccard={metrics['jaccard']:.3f}, Common={metrics['intersection_size']}/{metrics['set1_size']} vs {metrics['set2_size']}")
    
    # Save results
    results = {
        "server_vs_orin_vqa2": {
            "server_pareto": list(server_vqa2) if "server_vqa2" in pareto_frontiers else None,
            "orin_pareto": list(orin_vqa2) if "orin_vqa2" in pareto_frontiers else None,
            "overlap_metrics": overlap if "server_vqa2" in pareto_frontiers and "orin_vqa2" in pareto_frontiers else None,
            "identical": server_vqa2 == orin_vqa2 if "server_vqa2" in pareto_frontiers and "orin_vqa2" in pareto_frontiers else None
        },
        "cross_dataset_overlap": {
            k: {
                "jaccard": v["jaccard"],
                "intersection_size": v["intersection_size"],
                "set1_size": v["set1_size"],
                "set2_size": v["set2_size"]
            }
            for k, v in overlap_matrix.items()
        }
    }
    
    output_file = os.path.join(base_dir, "pareto_overlap_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()







