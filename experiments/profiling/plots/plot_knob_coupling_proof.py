"""
Generate "Coupling Proof" plots for E2: Knob coupling + Pareto-front structure.

This script demonstrates that the three knobs are coupled by:
1. Fixing one knob (e.g., max_crops)
2. Computing Pareto frontier over the other two knobs (top_k, num_active_blocks)
3. Showing how the frontier changes materially when the fixed knob changes

Usage:
    python3 experiments/profiling/plots/plot_knob_coupling_proof.py \
        --results-dir /home/x-pwang1/ai_project/molmo_hf/results/core_exp_h100/4run_2000samples \
        --output-dir /home/x-pwang1/ai_project/molmo_hf/analysis_output/e2_knob_coupling
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

ConfigKey = Tuple[int, int, int]  # (max_crops, top_k, num_active_blocks)


def load_task_results(task_dir: Path) -> List[Dict]:
    """Load all result JSON files for a task."""
    # Pattern: <task>_imgsizetier-<tier>_crops<num>_topk<num>_blocks<num>.json
    result_files = sorted(task_dir.glob("*.json"))
    
    if not result_files:
        raise FileNotFoundError(f"No result files found in {task_dir}")
    
    rows: List[Dict] = []
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract config from file data
            max_crops = data.get("max_crops", 0)
            top_k = data.get("top_k", 0)
            num_active_blocks = data.get("num_active_blocks", 0)
            accuracy = data.get("accuracy", 0.0)
            
            # Extract latencies from latency_stats
            latency_stats = data.get("latency_stats", {})
            latency_prefill = latency_stats.get("T_LLM_prefill_p50", 0.0)
            latency_decode = latency_stats.get("T_LLM_decode_p50", 0.0)
            latency_total = latency_stats.get("T_total_p50", 0.0)
            
            if max_crops == 0 or top_k == 0 or num_active_blocks == 0:
                # Skip invalid configs
                continue
            
            rows.append({
                "max_crops": max_crops,
                "top_k": top_k,
                "num_active_blocks": num_active_blocks,
                "accuracy": accuracy,
                "latency_prefill": latency_prefill,
                "latency_decode": latency_decode,
                "latency_total": latency_total,
            })
                
        except Exception as e:
            print(f"Warning: Failed to load {result_file}: {e}")
            continue
    
    return rows


def compute_pareto_frontier(
    points: List[Tuple[float, float, ConfigKey]]
) -> List[Tuple[float, float, ConfigKey]]:
    """
    Compute Pareto frontier points.
    
    For accuracy-latency tradeoff:
    - A point is on the Pareto frontier if there's no other point with both 
      higher accuracy AND lower latency
    - Or equivalently: no other point dominates it (has >= accuracy AND <= latency, 
      with at least one strict inequality)
    
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


def plot_coupling_proof_fixed_knob(
    task_name: str,
    rows: List[Dict],
    fixed_knob: str,  # "max_crops", "top_k", or "num_active_blocks"
    latency_type: str,  # "total", "prefill", or "decode"
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
):
    """
    Create coupling proof plot by fixing one knob and showing Pareto frontiers 
    over the other two knobs for different values of the fixed knob.
    
    Args:
        task_name: Name of the task/dataset
        rows: List of result dictionaries
        fixed_knob: Which knob to fix ("max_crops", "top_k", or "num_active_blocks")
        latency_type: Which latency metric to use ("total", "prefill", "decode")
        output_dir: Directory to save the plot
    """
    latency_key = f"latency_{latency_type}"
    
    # Group rows by the fixed knob value
    grouped_by_fixed = defaultdict(list)
    for row in rows:
        fixed_value = row[fixed_knob]
        grouped_by_fixed[fixed_value].append(row)
    
    # Get sorted fixed knob values
    fixed_values = sorted(grouped_by_fixed.keys())
    
    if len(fixed_values) < 2:
        print(f"  Warning: Need at least 2 different values for {fixed_knob}, found {len(fixed_values)}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Color palette for different fixed knob values
    colors = plt.cm.tab10(np.linspace(0, 1, len(fixed_values)))
    
    # Plot all points (gray, low opacity)
    all_latencies = [r[latency_key] for r in rows]
    all_accuracies = [r["accuracy"] for r in rows]
    ax.scatter(all_latencies, all_accuracies, color="gray", alpha=0.2, s=30, 
               label="All Configurations", zorder=1)
    
    # First pass: compute all Pareto frontiers and store them
    all_pareto_frontiers = {}
    for idx, fixed_val in enumerate(fixed_values):
        subset_rows = grouped_by_fixed[fixed_val]
        
        # Convert to (latency, accuracy, config) tuples
        points = [
            (r[latency_key], r["accuracy"], 
             (r["max_crops"], r["top_k"], r["num_active_blocks"]))
            for r in subset_rows
        ]
        
        # Compute Pareto frontier
        pareto_points = compute_pareto_frontier(points)
        all_pareto_frontiers[fixed_val] = pareto_points
    
    # Second pass: identify consistent vs inconsistent points and plot
    # A point is consistent if it appears in multiple frontiers (within tolerance)
    tolerance_latency = 0.01  # 1% tolerance for latency
    tolerance_accuracy = 0.001  # 0.1% tolerance for accuracy
    
    for idx, fixed_val in enumerate(fixed_values):
        pareto_points = all_pareto_frontiers[fixed_val]
        
        if len(pareto_points) == 0:
            continue
        
        # Extract latencies and accuracies for plotting
        pareto_latencies = [p[0] for p in pareto_points]
        pareto_accuracies = [p[1] for p in pareto_points]
        
        # Determine which points are inconsistent (only in this frontier)
        # A point is inconsistent if the same varying knobs combination doesn't appear
        # in ALL other frontiers' Pareto points
        # Only compare the two varying knobs values, not latency/accuracy
        inconsistent_indices = []
        consistent_indices = []
        
        # Determine which two knobs are varying
        varying_knobs = [k for k in ['max_crops', 'top_k', 'num_active_blocks'] if k != fixed_knob]
        
        for i, (lat, acc, config) in enumerate(pareto_points):
            # Extract the varying knobs values for this point
            if fixed_knob == 'max_crops':
                varying_vals = (config[1], config[2])  # (top_k, num_active_blocks)
            elif fixed_knob == 'top_k':
                varying_vals = (config[0], config[2])  # (max_crops, num_active_blocks)
            else:  # fixed_knob == 'num_active_blocks'
                varying_vals = (config[0], config[1])  # (max_crops, top_k)
            
            # Check if this varying knobs combination appears in ALL other frontiers
            appears_in_all_others = True
            for other_fixed_val, other_pareto_points in all_pareto_frontiers.items():
                if other_fixed_val == fixed_val:
                    continue
                
                found_in_other = False
                for other_lat, other_acc, other_config in other_pareto_points:
                    # Extract varying knobs for other point
                    if fixed_knob == 'max_crops':
                        other_varying_vals = (other_config[1], other_config[2])
                    elif fixed_knob == 'top_k':
                        other_varying_vals = (other_config[0], other_config[2])
                    else:
                        other_varying_vals = (other_config[0], other_config[1])
                    
                    # Only check if varying knobs match (don't check latency/accuracy)
                    if varying_vals == other_varying_vals:
                        found_in_other = True
                        break
                
                if not found_in_other:
                    appears_in_all_others = False
                    break
            
            if appears_in_all_others:
                # This varying knobs combination appears in all frontiers - don't highlight
                consistent_indices.append(i)
            else:
                # This varying knobs combination doesn't appear in all frontiers - highlight it
                inconsistent_indices.append(i)
        
        # Plot the full Pareto frontier line first (normal style)
        ax.plot(
            pareto_latencies,
            pareto_accuracies,
            color=colors[idx],
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=f"{fixed_knob}={fixed_val}",
            zorder=3 + idx,
            alpha=0.8,
        )
        
        # Overlay highlighted segments for inconsistent points
        if inconsistent_indices:
            # Group consecutive inconsistent indices to form segments
            inconsistent_segments = []
            current_segment = []
            
            for i in range(len(pareto_points)):
                if i in inconsistent_indices:
                    current_segment.append(i)
                else:
                    if len(current_segment) > 0:
                        inconsistent_segments.append(current_segment)
                        current_segment = []
            if len(current_segment) > 0:
                inconsistent_segments.append(current_segment)
            
            # Plot each inconsistent segment with highlight
            for segment in inconsistent_segments:
                if len(segment) == 1:
                    # Single point - just highlight the marker
                    i = segment[0]
                    ax.plot(
                        [pareto_latencies[i]],
                        [pareto_accuracies[i]],
                        color=colors[idx],
                        marker="o",
                        linewidth=0,
                        markersize=14,  # Larger marker
                        zorder=6 + idx,
                        alpha=1.0,
                        markeredgewidth=3,
                        markeredgecolor='red',  # Red border for highlight
                    )
                else:
                    # Multiple points - highlight the line segment
                    seg_latencies = [pareto_latencies[i] for i in segment]
                    seg_accuracies = [pareto_accuracies[i] for i in segment]
                    ax.plot(
                        seg_latencies,
                        seg_accuracies,
                        color=colors[idx],
                        marker="o",
                        linewidth=4.5,  # Thicker line for highlight
                        markersize=12,  # Larger markers
                        zorder=5 + idx,  # Higher zorder to be on top
                        alpha=1.0,  # Full opacity
                        markeredgewidth=2,
                        markeredgecolor='red',  # Red border for highlight
                    )
        
        # Annotate all points on the frontier with only the varying knobs
        if len(pareto_points) > 0:
            # Determine which two knobs are varying
            varying_knobs = [k for k in ['max_crops', 'top_k', 'num_active_blocks'] if k != fixed_knob]
            
            for i, (latency, accuracy, config) in enumerate(pareto_points):
                # Create label with only the two varying knobs
                if fixed_knob == 'max_crops':
                    label = f"({config[1]},{config[2]})"  # (top_k, num_active_blocks)
                elif fixed_knob == 'top_k':
                    label = f"({config[0]},{config[2]})"  # (max_crops, num_active_blocks)
                else:  # fixed_knob == 'num_active_blocks'
                    label = f"({config[0]},{config[1]})"  # (max_crops, top_k)
                
                # Use thicker border for inconsistent points
                is_inconsistent = i in inconsistent_indices
                edge_width = 2.0 if is_inconsistent else 1.0
                
                ax.annotate(
                    label,
                    (latency, accuracy),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                             alpha=0.9, edgecolor=colors[idx], linewidth=edge_width),
                    zorder=10,
                )
    
    # Set labels and title
    latency_label_map = {
        "total": "Total Latency",
        "prefill": "Prefill Latency",
        "decode": "Decode Latency",
    }
    latency_label = latency_label_map.get(latency_type, "Latency")
    
    ax.set_xlabel(f'{latency_label} (ms, P50)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(
        f'{task_name}: Coupling Proof - Fixed {fixed_knob}\n'
        f'Pareto Frontiers over (other two knobs) vary with {fixed_knob}',
        fontsize=16, fontweight='bold', pad=15
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Determine varying knobs for explanation
    varying_knobs = [k for k in ['max_crops', 'top_k', 'num_active_blocks'] if k != fixed_knob]
    
    # Place legend in upper left corner
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add explanation text box in lower right corner
    explanation = (
        f"Fixed: {fixed_knob}\n"
        f"Varying: {', '.join(varying_knobs)}\n"
        f"Label: ({varying_knobs[0]}, {varying_knobs[1]})"
    )
    ax.text(
        0.98, 0.02, explanation,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, 
                 edgecolor='black', linewidth=1.0),
        family='monospace',
        zorder=100  # Ensure it's on top
    )
    
    plt.tight_layout()
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"coupling_proof_{task_name}_fixed_{fixed_knob}_{latency_type}.png"
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {output_file}")
    
    return output_file


def plot_all_coupling_proofs(
    task_name: str,
    rows: List[Dict],
    output_dir: Path,
):
    """Generate all coupling proof plots for a task."""
    latency_types = ["total", "prefill", "decode"]
    fixed_knobs = ["max_crops", "top_k", "num_active_blocks"]
    
    output_files = []
    
    for fixed_knob in fixed_knobs:
        for latency_type in latency_types:
            try:
                output_file = plot_coupling_proof_fixed_knob(
                    task_name=task_name,
                    rows=rows,
                    fixed_knob=fixed_knob,
                    latency_type=latency_type,
                    output_dir=output_dir,
                )
                if output_file:
                    output_files.append(output_file)
            except Exception as e:
                print(f"  Error creating plot for {fixed_knob}, {latency_type}: {e}")
                import traceback
                traceback.print_exc()
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate coupling proof plots for knob coupling analysis"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/home/x-pwang1/ai_project/molmo_hf/results/core_exp_h100/4run_2000samples"),
        help="Directory containing task subdirectories with result JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/x-pwang1/ai_project/molmo_hf/analysis_output/e2_knob_coupling"),
        help="Directory to write output plots",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific tasks to process (default: all tasks)",
    )
    parser.add_argument(
        "--latency-type",
        type=str,
        choices=["total", "prefill", "decode"],
        default="total",
        help="Which latency type to use (default: total)",
    )
    parser.add_argument(
        "--fixed-knob",
        type=str,
        choices=["max_crops", "top_k", "num_active_blocks"],
        default=None,
        help="Which knob to fix (default: generate plots for all knobs)",
    )
    args = parser.parse_args()
    
    # Find all task directories
    task_dirs = [d for d in args.results_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if args.tasks:
        task_dirs = [d for d in task_dirs if d.name in args.tasks]
    
    if not task_dirs:
        raise RuntimeError(f"No task directories found under {args.results_dir}")
    
    print(f"Found {len(task_dirs)} task directories")
    
    for task_dir in sorted(task_dirs):
        task_name = task_dir.name
        print(f"\n{'='*80}")
        print(f"Processing task: {task_name}")
        print(f"{'='*80}")
        
        try:
            rows = load_task_results(task_dir)
            if not rows:
                print(f"  Warning: No valid results found for {task_name}")
                continue
            
            print(f"  Loaded {len(rows)} configurations")
            
            # Show config range
            max_crops_vals = sorted(set(r["max_crops"] for r in rows))
            top_k_vals = sorted(set(r["top_k"] for r in rows))
            blocks_vals = sorted(set(r["num_active_blocks"] for r in rows))
            print(f"  max_crops range: {max_crops_vals}")
            print(f"  top_k range: {top_k_vals}")
            print(f"  num_active_blocks range: {blocks_vals}")
            
            # Generate plots
            if args.fixed_knob:
                # Generate plot for specific fixed knob and latency type
                plot_coupling_proof_fixed_knob(
                    task_name=task_name,
                    rows=rows,
                    fixed_knob=args.fixed_knob,
                    latency_type=args.latency_type,
                    output_dir=args.output_dir,
                )
            else:
                # Generate all coupling proof plots
                plot_all_coupling_proofs(
                    task_name=task_name,
                    rows=rows,
                    output_dir=args.output_dir,
                )
            
        except Exception as e:
            print(f"  Error processing {task_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("All coupling proof plots generated!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

