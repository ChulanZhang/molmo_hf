"""
Compare Pareto frontiers across different latency types (total, prefill, decode).
Generate a table showing which configurations are on the Pareto frontier for each latency type.
"""

import json
import os
import argparse
from typing import Dict, Set, Tuple, List
from collections import defaultdict

def load_exp6_results(exp6_file: str, latency_type: str = "total", latency_metric: str = "mean") -> Dict[Tuple[int, int, int], float]:
    """
    Load exp6 latency results from the merged JSON file.
    
    Args:
        exp6_file: Path to exp6_latency_results.json
        latency_type: Which latency type to use ("total", "prefill", "decode")
        latency_metric: Which latency metric to use ("mean", "P50", "P95", "P99")
    
    Returns:
        Dictionary mapping (max_crops, top_k, num_active_blocks) -> latency_ms
    """
    results = {}
    
    with open(exp6_file, 'r') as f:
        data = json.load(f)
    
    latency_key_map = {
        "total": "latency_total_ms",
        "prefill": "latency_prefill_ms",
        "decode": "latency_decode_ms",
    }
    
    latency_key = latency_key_map.get(latency_type, "latency_total_ms")
    
    if "summary" in data:
        for entry in data["summary"]:
            key = (
                entry["max_crops"],
                entry["top_k"],
                entry["num_active_blocks"]
            )
            
            if latency_key in entry:
                latency_dict = entry[latency_key]
                if latency_metric in latency_dict:
                    latency = latency_dict[latency_metric]
                    results[key] = latency
    
    return results

def load_exp5_results(exp5_dir: str) -> Dict[Tuple[int, int, int], float]:
    """
    Load exp5 accuracy results from all individual JSON files.
    
    Returns:
        Dictionary mapping (max_crops, top_k, num_active_blocks) -> accuracy
    """
    import glob
    
    results = {}
    pattern = os.path.join(exp5_dir, "exp5_accuracy_results_*.json")
    files = glob.glob(pattern)
    
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if "summary" in data and len(data["summary"]) > 0:
                entry = data["summary"][0]
                key = (
                    entry["max_crops"],
                    entry["top_k"],
                    entry["num_active_blocks"]
                )
                accuracy = entry["accuracy"]
                results[key] = accuracy
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return results

def compute_pareto_frontier(
    points: List[Tuple[float, float, Tuple[int, int, int]]]
) -> Set[Tuple[int, int, int]]:
    """
    Compute Pareto frontier points.
    
    Args:
        points: List of (latency, accuracy, config_key) tuples
    
    Returns:
        Set of configuration keys on the Pareto frontier
    """
    if len(points) == 0:
        return set()
    
    sorted_points = sorted(points, key=lambda x: (x[0], -x[1]))
    
    pareto_configs = set()
    
    for i, (latency, accuracy, config) in enumerate(sorted_points):
        is_pareto = True
        
        # Check if this point is dominated by any other point
        for j, (other_latency, other_accuracy, _) in enumerate(sorted_points):
            if i == j:
                continue
            
            # A point dominates if it has >= accuracy AND <= latency, with at least one strict inequality
            if (other_accuracy >= accuracy and other_latency <= latency and 
                (other_accuracy > accuracy or other_latency < latency)):
                is_pareto = False
                break
        
        if is_pareto:
            pareto_configs.add(config)
    
    return pareto_configs

def compare_pareto_frontiers(
    exp5_dir: str,
    exp6_file: str,
    latency_metric: str = "mean",
    output_file: str = None,
):
    """
    Compare Pareto frontiers across different latency types.
    
    Args:
        exp5_dir: Directory containing exp5 accuracy result files
        exp6_file: Path to exp6_latency_results.json
        latency_metric: Which latency metric to use ("mean", "P50", "P95", "P99")
        output_file: Output file path for the comparison table (optional)
    """
    print("Loading exp5 accuracy results...")
    exp5_results = load_exp5_results(exp5_dir)
    
    latency_types = ["total", "prefill", "decode"]
    pareto_frontiers = {}
    all_configs = set()
    
    # Compute Pareto frontier for each latency type
    for latency_type in latency_types:
        print(f"\nComputing Pareto frontier for {latency_type} latency...")
        exp6_results = load_exp6_results(exp6_file, latency_type=latency_type, latency_metric=latency_metric)
        
        # Match configurations
        matched_points = []
        for config_key in exp5_results:
            if config_key in exp6_results:
                accuracy = exp5_results[config_key]
                latency = exp6_results[config_key]
                matched_points.append((latency, accuracy, config_key))
                all_configs.add(config_key)
        
        # Compute Pareto frontier
        pareto_configs = compute_pareto_frontier(matched_points)
        pareto_frontiers[latency_type] = pareto_configs
        print(f"  Found {len(pareto_configs)} Pareto frontier points")
    
    # Generate comparison table
    print("\n" + "="*100)
    print("Pareto Frontier Comparison Table")
    print("="*100)
    
    # Find all configurations that appear in at least one Pareto frontier
    all_pareto_configs = set()
    for pareto_set in pareto_frontiers.values():
        all_pareto_configs.update(pareto_set)
    
    # Sort configurations for consistent output
    sorted_configs = sorted(all_pareto_configs, key=lambda x: (x[0], x[1], x[2]))
    
    # Load latency values for all types
    latency_values = {}
    for latency_type in latency_types:
        latency_values[latency_type] = load_exp6_results(exp6_file, latency_type=latency_type, latency_metric=latency_metric)
    
    # Print table header
    header = (f"{'Configuration':<20} {'Accuracy':<12} {'Total':<12} {'Prefill':<12} {'Decode':<12} "
              f"{'In Total':<10} {'In Prefill':<12} {'In Decode':<12} {'In All':<10}")
    print(header)
    print("-" * 140)
    
    # Count statistics
    in_all = 0
    in_two = 0
    in_one = 0
    
    table_rows = []
    
    for config in sorted_configs:
        max_crops, top_k, num_active_blocks = config
        config_str = f"({max_crops},{top_k},{num_active_blocks})"
        
        # Get accuracy
        accuracy = exp5_results.get(config, 0.0)
        accuracy_str = f"{accuracy:.4f}" if config in exp5_results else "N/A"
        
        # Get latency values
        total_latency = latency_values["total"].get(config, 0.0)
        prefill_latency = latency_values["prefill"].get(config, 0.0)
        decode_latency = latency_values["decode"].get(config, 0.0)
        
        total_latency_str = f"{total_latency:.2f}" if config in latency_values["total"] else "N/A"
        prefill_latency_str = f"{prefill_latency:.2f}" if config in latency_values["prefill"] else "N/A"
        decode_latency_str = f"{decode_latency:.2f}" if config in latency_values["decode"] else "N/A"
        
        in_total = "✓" if config in pareto_frontiers["total"] else " "
        in_prefill = "✓" if config in pareto_frontiers["prefill"] else " "
        in_decode = "✓" if config in pareto_frontiers["decode"] else " "
        
        count = sum([
            config in pareto_frontiers["total"],
            config in pareto_frontiers["prefill"],
            config in pareto_frontiers["decode"]
        ])
        
        if count == 3:
            in_all += 1
            in_all_str = "✓"
        else:
            in_all_str = " "
            if count == 2:
                in_two += 1
            elif count == 1:
                in_one += 1
        
        row = (f"{config_str:<20} {accuracy_str:<12} {total_latency_str:<12} {prefill_latency_str:<12} "
               f"{decode_latency_str:<12} {in_total:<10} {in_prefill:<12} {in_decode:<12} {in_all_str:<10}")
        print(row)
        table_rows.append({
            "config": config_str,
            "max_crops": max_crops,
            "top_k": top_k,
            "num_active_blocks": num_active_blocks,
            "accuracy": accuracy,
            "total_latency": total_latency,
            "prefill_latency": prefill_latency,
            "decode_latency": decode_latency,
            "in_total": config in pareto_frontiers["total"],
            "in_prefill": config in pareto_frontiers["prefill"],
            "in_decode": config in pareto_frontiers["decode"],
            "in_all": count == 3,
        })
    
    print("-" * 140)
    print(f"\nSummary:")
    print(f"  Total unique Pareto configurations: {len(all_pareto_configs)}")
    print(f"  Configurations in all three frontiers: {in_all}")
    print(f"  Configurations in two frontiers: {in_two}")
    print(f"  Configurations in one frontier only: {in_one}")
    print(f"\nBreakdown by latency type:")
    print(f"  Total latency Pareto points: {len(pareto_frontiers['total'])}")
    print(f"  Prefill latency Pareto points: {len(pareto_frontiers['prefill'])}")
    print(f"  Decode latency Pareto points: {len(pareto_frontiers['decode'])}")
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write("Pareto Frontier Comparison Table\n")
            f.write("="*140 + "\n\n")
            f.write(f"Latency metric: {latency_metric}\n")
            f.write(f"Latency values are in milliseconds (ms)\n\n")
            f.write(header + "\n")
            f.write("-" * 140 + "\n")
            for row_data in table_rows:
                config_str = row_data["config"]
                accuracy_str = f"{row_data['accuracy']:.4f}"
                total_latency_str = f"{row_data['total_latency']:.2f}"
                prefill_latency_str = f"{row_data['prefill_latency']:.2f}"
                decode_latency_str = f"{row_data['decode_latency']:.2f}"
                in_total = "✓" if row_data["in_total"] else " "
                in_prefill = "✓" if row_data["in_prefill"] else " "
                in_decode = "✓" if row_data["in_decode"] else " "
                in_all_str = "✓" if row_data["in_all"] else " "
                f.write(f"{config_str:<20} {accuracy_str:<12} {total_latency_str:<12} {prefill_latency_str:<12} "
                       f"{decode_latency_str:<12} {in_total:<10} {in_prefill:<12} {in_decode:<12} {in_all_str:<10}\n")
            f.write("-" * 140 + "\n")
            f.write(f"\nSummary:\n")
            f.write(f"  Total unique Pareto configurations: {len(all_pareto_configs)}\n")
            f.write(f"  Configurations in all three frontiers: {in_all}\n")
            f.write(f"  Configurations in two frontiers: {in_two}\n")
            f.write(f"  Configurations in one frontier only: {in_one}\n")
            f.write(f"\nBreakdown by latency type:\n")
            f.write(f"  Total latency Pareto points: {len(pareto_frontiers['total'])}\n")
            f.write(f"  Prefill latency Pareto points: {len(pareto_frontiers['prefill'])}\n")
            f.write(f"  Decode latency Pareto points: {len(pareto_frontiers['decode'])}\n")
        print(f"\nComparison table saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare Pareto frontiers across different latency types"
    )
    parser.add_argument(
        "--exp5_dir",
        type=str,
        default="./results/profiling/exp5_accuracy",
        help="Directory containing exp5 accuracy result files"
    )
    parser.add_argument(
        "--exp6_file",
        type=str,
        default="./results/profiling/exp6_latency/exp6_latency_results.json",
        help="Path to exp6_latency_results.json"
    )
    parser.add_argument(
        "--latency_metric",
        type=str,
        default="mean",
        choices=["mean", "P50", "P95", "P99"],
        help="Which latency metric to use (default: mean)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results/profiling/pareto_frontier_comparison.txt",
        help="Output file path for the comparison table"
    )
    
    args = parser.parse_args()
    
    compare_pareto_frontiers(
        exp5_dir=args.exp5_dir,
        exp6_file=args.exp6_file,
        latency_metric=args.latency_metric,
        output_file=args.output_file,
    )

if __name__ == "__main__":
    main()

