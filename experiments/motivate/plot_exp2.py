#!/usr/bin/env python3
"""
Plot script for Experiment 2: Component Profiling
This script can be run independently to regenerate plots from existing JSON results.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def plot_component_pie_charts(data: Dict, output_dir: Path, dataset_name: str, split: str):
    """Plot pie charts for parameter and latency distributions."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Unified color mapping for components (ggthemes Classic_10 - high saturation, colorblind-friendly)
    # LLM uses green (previously connector's color) since connector is merged into projector
    component_colors = {
        "Vision Encoder": '#1F77B4',  # Deep blue (Classic_10 color 1)
        "Projector": '#FF7F0E',        # Bright orange (Classic_10 color 2)
        "LLM": '#2CA02C',              # Deep green (Classic_10 color 3, previously connector's color)
        "LLM Prefill": '#2CA02C',      # Deep green (same as LLM)
    }

    # Extract data
    if "average_latencies" in data:
        avg_results = data["average_latencies"]
        params = data.get("parameters", {})
    else:
        # Fallback: compute from results
        results = data.get("results", [])
        numeric_keys = [
            k for k in results[0].keys()
            if isinstance(results[0][k], (int, float)) and results[0][k] is not None
        ]
        avg_results = {k: np.mean([r[k] for r in results if r.get(k) is not None]) for k in numeric_keys}
        params = data.get("parameters", {})

    # Plot 1: Parameter Distribution (same style as latency distribution)
    plt.figure(figsize=(8, 6))
    param_labels = ["Vision Encoder", "Projector", "LLM"]
    param_values = [
        params.get("params_vision_encoder", 0),
        params.get("params_projector", 0),
        params.get("params_llm", 0),
    ]
    # Filter out zero values
    param_data = [(l, v) for l, v in zip(param_labels, param_values) if v > 0]
    param_labels_filtered = [l for l, v in param_data]
    param_values_filtered = [v for l, v in param_data]
    param_colors_filtered = [component_colors[l] for l in param_labels_filtered]
    
    # Calculate percentages
    total_params = sum(param_values_filtered)
    param_percentages = [v / total_params * 100 for v in param_values_filtered]
    
    # Draw pie chart without labels (use legend instead)
    wedges, texts, autotexts = plt.pie(
        param_values_filtered,
        labels=None,  # No labels on pie, use legend instead
        colors=param_colors_filtered,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,  # Percentages inside the pie
        textprops={'fontsize': 14},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}  # Thicker border
    )
    
    # Manual distance mapping to avoid overlap
    # Vision Encoder: outermost, LLM: middle, Projector: innermost
    # All moved inward to keep within pie chart
    distance_map = {}
    for label in param_labels_filtered:
        if label == 'Vision Encoder':
            distance_map[label] = 0.85  # Outermost position (moved inward)
        elif label == 'LLM':
            distance_map[label] = 0.75   # Middle position (moved inward)
        elif label == 'Projector':
            distance_map[label] = 0.65   # Innermost position (moved inward)
        else:
            distance_map[label] = 0.75   # Default
    
    # Set text style and manually adjust positions to avoid overlap
    for i, (autotext, wedge, label) in enumerate(zip(autotexts, wedges, param_labels_filtered)):
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
        
        # Get the angle of the wedge center
        theta = (wedge.theta2 + wedge.theta1) / 2
        theta_rad = np.deg2rad(theta)
        
        # Get specified distance for this component
        distance = distance_map.get(label, 0.85)
        
        # Calculate new position
        x = distance * np.cos(theta_rad)
        y = distance * np.sin(theta_rad)
        
        # Set new position
        autotext.set_position((x, y))
    
    # Create legend with values for parameters
    legend_labels = [f'{label}: {val/1e6:.2f}M ({pct:.1f}%)' 
                     for label, val, pct in zip(param_labels_filtered, param_values_filtered, param_percentages)]
    legend = plt.legend(wedges, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.4), 
                       ncol=1, fontsize=14, framealpha=0.9)
    legend.get_frame().set_linewidth(1.5)

    # Remove title as requested
    plt.tight_layout()
    plt.savefig(fig_dir / f"exp2_component_profiling_parameters_{dataset_name}_{split}.png", dpi=300, bbox_inches="tight")
    log.info(f"Plot saved to {fig_dir / f'exp2_component_profiling_parameters_{dataset_name}_{split}.png'}")
    plt.close()

    # Plot 2: Latency Distribution
    plt.figure(figsize=(8, 6))
    latency_labels = ["Vision Encoder", "Projector", "LLM"]
    latency_values = [
        avg_results.get("T_vision_encoder", 0),
        avg_results.get("T_projector", 0),
        avg_results.get("T_LLM_prefill", 0),
    ]
    # Filter out zero values
    latency_data = [(l, v) for l, v in zip(latency_labels, latency_values) if v > 0]
    latency_labels_filtered = [l for l, v in latency_data]
    latency_values_filtered = [v for l, v in latency_data]
    latency_colors_filtered = [component_colors[l] for l in latency_labels_filtered]
    
    # Calculate percentages
    total_latency = sum(latency_values_filtered)
    latency_percentages = [v / total_latency * 100 for v in latency_values_filtered]
    
    # Draw pie chart without labels (use legend instead)
    wedges, texts, autotexts = plt.pie(
        latency_values_filtered,
        labels=None,  # No labels on pie, use legend instead
        colors=latency_colors_filtered,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,  # Percentages inside the pie
        textprops={'fontsize': 14},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}  # Thicker border
    )
    
    # Manual distance mapping to avoid overlap
    # Vision Encoder: outermost, LLM: middle, Projector: innermost
    # All moved inward to keep within pie chart
    distance_map = {}
    for label in latency_labels_filtered:
        if label == 'Vision Encoder':
            distance_map[label] = 0.85  # Outermost position (moved inward)
        elif label == 'LLM':
            distance_map[label] = 0.75   # Middle position (moved inward)
        elif label == 'Projector':
            distance_map[label] = 0.65   # Innermost position (moved inward)
        else:
            distance_map[label] = 0.75   # Default
    
    # Set text style and manually adjust positions to avoid overlap
    for i, (autotext, wedge, label) in enumerate(zip(autotexts, wedges, latency_labels_filtered)):
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
        
        # Get the angle of the wedge center
        theta = (wedge.theta2 + wedge.theta1) / 2
        theta_rad = np.deg2rad(theta)
        
        # Get specified distance for this component
        distance = distance_map.get(label, 0.85)
        
        # Calculate new position
        x = distance * np.cos(theta_rad)
        y = distance * np.sin(theta_rad)
        
        # Set new position
        autotext.set_position((x, y))
    
    # Create legend with values for latencies
    legend_labels = [f'{label}: {val:.2f}ms ({pct:.1f}%)' 
                     for label, val, pct in zip(latency_labels_filtered, latency_values_filtered, latency_percentages)]
    legend = plt.legend(wedges, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.4), 
                       ncol=1, fontsize=14, framealpha=0.9)
    legend.get_frame().set_linewidth(1.5)

    # plt.title("Latency Distribution", fontsize=18)
    plt.tight_layout()
    plt.savefig(fig_dir / f"exp2_component_profiling_latency_{dataset_name}_{split}.png", dpi=300, bbox_inches="tight")
    log.info(f"Plot saved to {fig_dir / f'exp2_component_profiling_latency_{dataset_name}_{split}.png'}")
    plt.close()

    # Print Statistics
    log.info("=" * 60)
    log.info("Average Latencies:")
    log.info(f"  T_vision_encoder:  {avg_results.get('T_vision_encoder', 0):.2f} ms")
    log.info(f"  T_projector:       {avg_results.get('T_projector', 0):.2f} ms")
    log.info(f"  T_LLM_prefill:    {avg_results.get('T_LLM_prefill', 0):.2f} ms")
    log.info(f"  T_total:          {avg_results.get('T_total', 0):.2f} ms")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot Experiment 2: Component Profiling")
    parser.add_argument("--json_file", type=str, required=True, help="Path to exp2_component_profiling.json")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as JSON file)")
    parser.add_argument("--dataset", type=str, default="coco_2014_vqa", help="Dataset name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")

    args = parser.parse_args()

    # Load results
    json_path = Path(args.json_file)
    if not json_path.exists():
        log.error(f"JSON file not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = json_path.parent

    log.info(f"Loaded data from {json_path}")
    log.info(f"Output directory: {output_dir}")

    # Plot
    plot_component_pie_charts(data, output_dir, args.dataset, args.split)
    log.info("Plotting complete!")


if __name__ == "__main__":
    main()
