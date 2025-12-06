#!/usr/bin/env python3
"""
Plot script for Profiling Experiment 4: Output Tokens vs Latency
This script can be run independently to regenerate plots from existing JSON results.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def plot_output_tokens_vs_latency(data: Dict, output_dir: Path):
    """Plot output tokens vs latency using stacked bar chart."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle both old format (list) and new format (dict with summary)
    if isinstance(data, list):
        # Old format: direct list of results
        summary = data
        all_samples = []
    else:
        # New format: dict with summary and all_samples
        summary = data.get("summary", [])
        all_samples = data.get("all_samples", [])
    
    if not summary:
        log.error("No summary data found in JSON file")
        return
    
    # Sort by max_new_tokens
    summary = sorted(summary, key=lambda x: x.get("max_new_tokens", 0))
    
    max_new_tokens = [r.get("max_new_tokens", 0) for r in summary]
    decode_means = [r.get("decode", {}).get("mean", 0) for r in summary]
    total_means = [r.get("total", {}).get("mean", 0) for r in summary]
    groups = {}
    for sample in all_samples:
        mnt = sample.get("max_new_tokens", 0)
        if mnt not in groups:
            groups[mnt] = []
        groups[mnt].append(sample)
    
    # Compute component latencies
    vision_means = []
    llm_prefill_means = []
    llm_decode_means = []
    output_tokens_means = []
    
    for mnt in max_new_tokens:
        if mnt in groups and groups[mnt]:
            # Use per-sample data if available
            samples = groups[mnt]
            vision_means.append(np.mean([s.get("T_vision_total", 0) for s in samples]))
            llm_prefill_means.append(np.mean([s.get("T_LLM_prefill", 0) for s in samples]))
            llm_decode_means.append(np.mean([s.get("T_LLM_decode", 0) for s in samples]))
            output_tokens_means.append(np.mean([s.get("num_output_tokens", 0) for s in samples]))
        else:
            # Fallback to summary statistics
            idx = max_new_tokens.index(mnt)
            summary_entry = summary[idx]
            # Handle both old and new format
            if "decode" in summary_entry and isinstance(summary_entry["decode"], dict):
                decode_mean = summary_entry["decode"].get("mean", 0)
            else:
                decode_mean = summary_entry.get("decode", {}).get("mean", 0) if isinstance(summary_entry.get("decode"), dict) else 0
            
            vision_means.append(0)  # Not available in summary
            llm_prefill_means.append(0)  # Not available in summary
            llm_decode_means.append(decode_mean)
            output_tokens_means.append(mnt)
    
    # Color palette
    colors = {
        'vision': '#1F77B4',        # Deep blue
        'llm_prefill': '#FF7F0E',   # Bright orange
        'llm_decode': '#2CA02C',    # Deep green
        'total': '#D62728',         # Deep red
    }
    
    # Convert to seconds for better visibility
    vision_means_sec = [v / 1000.0 for v in vision_means]
    llm_prefill_means_sec = [v / 1000.0 for v in llm_prefill_means]
    llm_decode_means_sec = [v / 1000.0 for v in llm_decode_means]
    total_means_sec = [v / 1000.0 for v in total_means]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(max_new_tokens))
    width = 0.6
    
    # Stacked bars: Vision + Prefill at bottom, Decode on top
    p1 = plt.bar(x, vision_means_sec, width, label="Vision", 
                color=colors['vision'], edgecolor='black', linewidth=1.0)
    p2 = plt.bar(x, llm_prefill_means_sec, width, bottom=vision_means_sec, 
                label="LLM Prefill", color=colors['llm_prefill'], edgecolor='black', linewidth=1.0)
    p3 = plt.bar(x, llm_decode_means_sec, width, 
                bottom=np.array(vision_means_sec) + np.array(llm_prefill_means_sec), 
                label="LLM Decode", color=colors['llm_decode'], edgecolor='black', linewidth=1.0)
    
    # Add line showing Total Latency
    plt.plot(x, total_means_sec, "o-", label="Total Latency", 
            linewidth=2.5, markersize=8, color=colors['total'], zorder=10)
    
    plt.xlabel("Output Tokens", fontsize=14)
    plt.ylabel("Latency (seconds, log scale)", fontsize=14)
    plt.title("Profiling Exp 4: Output Tokens vs Latency Breakdown", fontsize=16)
    plt.xticks(x, [int(round(v)) for v in output_tokens_means], fontsize=12)
    
    # Use log scale for better visibility
    plt.yscale('log')
    plt.ylim(bottom=0.1, top=1000)
    
    # Format Y-axis labels
    def format_seconds(x, pos):
        if x < 1:
            return f'{x:.2f}'
        elif x < 10:
            return f'{x:.1f}'
        elif x < 100:
            return f'{x:.0f}'
        else:
            return f'{x:.0f}'
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_seconds))
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "exp4_output_tokens_vs_latency_breakdown.png", dpi=300, bbox_inches="tight")
    log.info(f"Plot saved to {fig_dir / 'exp4_output_tokens_vs_latency_breakdown.png'}")
    plt.close()
    
    # Print Statistics
    log.info("=" * 60)
    log.info("Output Tokens Analysis:")
    log.info(f"  Output tokens range: {min(output_tokens_means):.0f} - {max(output_tokens_means):.0f}")
    log.info(f"  Decode latency range: {min(decode_means):.2f} - {max(decode_means):.2f} ms")
    log.info(f"  Total latency range: {min(total_means):.2f} - {max(total_means):.2f} ms")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot Profiling Experiment 4: Output Tokens vs Latency")
    parser.add_argument("--json_file", type=str, required=True, 
                        help="Path to exp4_output_tokens_scaling_results.json")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Output directory (default: same as JSON file)")
    
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
    plot_output_tokens_vs_latency(data, output_dir)
    log.info("Plotting complete!")


if __name__ == "__main__":
    main()

