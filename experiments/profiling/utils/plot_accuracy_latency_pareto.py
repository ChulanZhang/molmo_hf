"""
Plot accuracy vs latency scatter plot with Pareto frontier for exp5 and exp6 results.
"""

import json
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set
from collections import defaultdict

def load_exp5_results(exp5_dir: str) -> Dict[Tuple[int, int, int], float]:
    """
    Load exp5 accuracy results from all individual JSON files.
    
    Returns:
        Dictionary mapping (max_crops, top_k, num_active_blocks) -> accuracy
    """
    results = {}
    
    # Find all exp5 result files
    pattern = os.path.join(exp5_dir, "exp5_accuracy_results_*.json")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} exp5 result files")
    
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
                print(f"  Loaded: crops={key[0]}, top_k={key[1]}, blocks={key[2]}, accuracy={accuracy:.4f}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return results

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
                    print(f"  Loaded: crops={key[0]}, top_k={key[1]}, blocks={key[2]}, {latency_type} latency={latency:.2f}ms ({latency_metric})")
                else:
                    print(f"  Warning: {latency_metric} not found for {key}, available: {list(latency_dict.keys())}")
    
    return results

def compute_pareto_frontier(
    points: List[Tuple[float, float, Tuple[int, int, int]]]
) -> List[Tuple[float, float, Tuple[int, int, int]]]:
    """
    Compute Pareto frontier points.
    
    For accuracy-latency tradeoff:
    - A point is on the Pareto frontier if there's no other point with both higher accuracy AND lower latency
    - Or equivalently: no other point dominates it (has >= accuracy AND <= latency, with at least one strict inequality)
    
    Args:
        points: List of (latency, accuracy, config_key) tuples
    
    Returns:
        List of Pareto frontier points, sorted by latency (ascending)
    """
    if len(points) == 0:
        return []
    
    # Sort by latency (ascending), then by accuracy (descending)
    sorted_points = sorted(points, key=lambda x: (x[0], -x[1]))
    
    pareto_points = []
    
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
            pareto_points.append((latency, accuracy, config))
    
    # Sort Pareto points by latency for plotting
    pareto_points.sort(key=lambda x: x[0])
    
    return pareto_points

def plot_accuracy_latency_pareto(
    exp5_dir: str,
    exp6_file: str,
    output_file: str,
    latency_type: str = "total",
    latency_metric: str = "mean",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
):
    """
    Plot accuracy vs latency scatter plot with Pareto frontier.
    
    Args:
        exp5_dir: Directory containing exp5 accuracy result files
        exp6_file: Path to exp6_latency_results.json
        output_file: Output path for the plot
        latency_type: Which latency type to use ("total", "prefill", "decode")
        latency_metric: Which latency metric to use ("mean", "P50", "P95", "P99")
        figsize: Figure size (width, height)
        dpi: DPI for the figure
    """
    print("Loading exp5 accuracy results...")
    exp5_results = load_exp5_results(exp5_dir)
    
    print(f"\nLoading exp6 {latency_type} latency results (metric: {latency_metric})...")
    exp6_results = load_exp6_results(exp6_file, latency_type=latency_type, latency_metric=latency_metric)
    
    # Match configurations
    print("\nMatching configurations...")
    matched_points = []
    matched_configs = []
    
    for config_key in exp5_results:
        if config_key in exp6_results:
            accuracy = exp5_results[config_key]
            latency = exp6_results[config_key]
            matched_points.append((latency, accuracy, config_key))
            matched_configs.append(config_key)
            print(f"  Matched: crops={config_key[0]}, top_k={config_key[1]}, blocks={config_key[2]}, "
                  f"accuracy={accuracy:.4f}, latency={latency:.2f}ms")
        else:
            print(f"  Warning: Config {config_key} found in exp5 but not in exp6")
    
    for config_key in exp6_results:
        if config_key not in exp5_results:
            print(f"  Warning: Config {config_key} found in exp6 but not in exp5")
    
    if len(matched_points) == 0:
        print("Error: No matching configurations found!")
        return
    
    print(f"\nTotal matched configurations: {len(matched_points)}")
    
    # Compute Pareto frontier
    print("\nComputing Pareto frontier...")
    pareto_points = compute_pareto_frontier(matched_points)
    pareto_configs = {point[2] for point in pareto_points}
    
    print(f"Pareto frontier points ({len(pareto_points)}):")
    for latency, accuracy, config in pareto_points:
        print(f"  crops={config[0]}, top_k={config[1]}, blocks={config[2]}, "
              f"accuracy={accuracy:.4f}, latency={latency:.2f}ms")
    
    # Separate Pareto and non-Pareto points
    pareto_latencies = []
    pareto_accuracies = []
    pareto_labels = []
    non_pareto_latencies = []
    non_pareto_accuracies = []
    non_pareto_labels = []
    
    for latency, accuracy, config in matched_points:
        label = f"crops={config[0]}, top_k={config[1]}, blocks={config[2]}"
        if config in pareto_configs:
            pareto_latencies.append(latency)
            pareto_accuracies.append(accuracy)
            pareto_labels.append(label)
        else:
            non_pareto_latencies.append(latency)
            non_pareto_accuracies.append(accuracy)
            non_pareto_labels.append(label)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot non-Pareto points
    if len(non_pareto_latencies) > 0:
        ax.scatter(
            non_pareto_latencies,
            non_pareto_accuracies,
            c='lightgray',
            s=100,
            alpha=0.6,
            label='Non-Pareto',
            edgecolors='gray',
            linewidths=0.5,
        )
    
    # Plot Pareto points
    if len(pareto_latencies) > 0:
        ax.scatter(
            pareto_latencies,
            pareto_accuracies,
            c='red',
            s=150,
            alpha=0.8,
            label='Pareto Frontier',
            edgecolors='darkred',
            linewidths=1.5,
            marker='*',
            zorder=5,
        )
    
    # Draw Pareto frontier line
    if len(pareto_points) > 1:
        pareto_latencies_sorted = [p[0] for p in pareto_points]
        pareto_accuracies_sorted = [p[1] for p in pareto_points]
        ax.plot(
            pareto_latencies_sorted,
            pareto_accuracies_sorted,
            'r--',
            linewidth=2,
            alpha=0.7,
            label='Pareto Frontier Line',
            zorder=4,
        )
    
    # Define latency label
    latency_label_map = {
        "total": "Total Latency",
        "prefill": "Prefill Latency",
        "decode": "Decode Latency",
    }
    latency_label = latency_label_map.get(latency_type, "Latency")
    
    # Add labels for Pareto points
    for latency, accuracy, config in pareto_points:
        label = f"({config[0]},{config[1]},{config[2]})"
        ax.annotate(
            label,
            (latency, accuracy),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            zorder=6,
        )
    
    ax.set_xlabel(f'{latency_label} (ms, {latency_metric})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Accuracy vs {latency_label} Trade-off (Pareto Frontier)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Add text box explaining the three knobs
    knob_explanation = (
        "Control Knobs:\n"
        "• max_crops: Maximum number of crops per image (controls vision tokens)\n"
        "• top_k: Number of active MoE experts (out of 64)\n"
        "• num_active_blocks: Number of active transformer blocks (out of 16)"
    )
    ax.text(
        0.02, 0.98, knob_explanation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace'
    )
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Also save a text file with Pareto frontier information
    text_output = output_file.replace('.png', '_pareto_info.txt')
    with open(text_output, 'w') as f:
        f.write("Pareto Frontier Configurations\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Latency type: {latency_type}\n")
        f.write(f"Total configurations tested: {len(matched_points)}\n")
        f.write(f"Pareto frontier points: {len(pareto_points)}\n")
        f.write(f"Latency metric: {latency_metric}\n\n")
        f.write("Pareto Frontier Points (sorted by latency):\n")
        f.write("-" * 80 + "\n")
        for i, (latency, accuracy, config) in enumerate(pareto_points, 1):
            f.write(f"{i}. max_crops={config[0]}, top_k={config[1]}, num_active_blocks={config[2]}\n")
            f.write(f"   Accuracy: {accuracy:.4f}\n")
            f.write(f"   {latency_label} ({latency_metric}): {latency:.2f} ms\n\n")
    
    print(f"Pareto frontier info saved to: {text_output}")
    
    plt.close()

def plot_all_latency_types(
    exp5_dir: str,
    exp6_file: str,
    output_dir: str,
    latency_metric: str = "mean",
    dpi: int = 300,
):
    """
    Plot accuracy vs latency scatter plots for all latency types (total, prefill, decode).
    
    Args:
        exp5_dir: Directory containing exp5 accuracy result files
        exp6_file: Path to exp6_latency_results.json
        output_dir: Output directory for the plots
        latency_metric: Which latency metric to use ("mean", "P50", "P95", "P99")
        dpi: DPI for the figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    latency_types = ["total", "prefill", "decode"]
    
    for latency_type in latency_types:
        output_file = os.path.join(output_dir, f"accuracy_vs_{latency_type}_latency_pareto.png")
        print(f"\n{'='*80}")
        print(f"Plotting {latency_type} latency...")
        print(f"{'='*80}")
        plot_accuracy_latency_pareto(
            exp5_dir=exp5_dir,
            exp6_file=exp6_file,
            output_file=output_file,
            latency_type=latency_type,
            latency_metric=latency_metric,
            dpi=dpi,
        )

def main():
    parser = argparse.ArgumentParser(
        description="Plot accuracy vs latency scatter plot with Pareto frontier"
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
        "--output_file",
        type=str,
        default=None,
        help="Output path for a single plot (if not specified, will generate all three: total, prefill, decode)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/profiling",
        help="Output directory for plots (used when generating all three plots)"
    )
    parser.add_argument(
        "--latency_type",
        type=str,
        default=None,
        choices=["total", "prefill", "decode"],
        help="Which latency type to plot (if not specified, will plot all three)"
    )
    parser.add_argument(
        "--latency_metric",
        type=str,
        default="mean",
        choices=["mean", "P50", "P95", "P99"],
        help="Which latency metric to use (default: mean)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for the output figure (default: 300)"
    )
    
    args = parser.parse_args()
    
    # If output_file is specified, plot a single latency type
    if args.output_file is not None:
        latency_type = args.latency_type or "total"
        plot_accuracy_latency_pareto(
            exp5_dir=args.exp5_dir,
            exp6_file=args.exp6_file,
            output_file=args.output_file,
            latency_type=latency_type,
            latency_metric=args.latency_metric,
            dpi=args.dpi,
        )
    # Otherwise, plot all three latency types
    else:
        plot_all_latency_types(
            exp5_dir=args.exp5_dir,
            exp6_file=args.exp6_file,
            output_dir=args.output_dir,
            latency_metric=args.latency_metric,
            dpi=args.dpi,
        )

if __name__ == "__main__":
    main()

