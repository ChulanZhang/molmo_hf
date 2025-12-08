#!/usr/bin/env python3
"""
Plot script for Profiling Experiment 2: MoE Top-K vs Latency
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


def plot_moe_topk_vs_latency(data: Dict, output_dir: Path):
    """Plot MoE Top-K vs latency using stacked bar chart."""
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
    
    # Sort by top_k
    summary = sorted(summary, key=lambda x: x.get("top_k", 0))
    
    top_k_values = [r.get("top_k", 0) for r in summary]
    prefill_means = [r.get("prefill", {}).get("mean", 0) for r in summary]
    decode_means = [r.get("decode", {}).get("mean", 0) for r in summary]
    
    # Get component latencies from all_samples
    all_samples = data.get("all_samples", [])
    groups = {}
    for sample in all_samples:
        tk = sample.get("top_k", 0)
        if tk not in groups:
            groups[tk] = []
        groups[tk].append(sample)
    
    # Compute component latencies
    vision_means = []
    llm_prefill_means = []
    llm_decode_means = []
    total_means = []
    
    for tk in top_k_values:
        if tk in groups and groups[tk]:
            # Use per-sample data if available
            samples = groups[tk]
            vision_means.append(np.mean([s.get("T_vision_total", 0) for s in samples]))
            llm_prefill_means.append(np.mean([s.get("T_LLM_prefill", 0) for s in samples]))
            llm_decode_means.append(np.mean([s.get("T_LLM_decode", 0) for s in samples]))
            total_means.append(np.mean([s.get("T_total", 0) for s in samples]))
        else:
            # Fallback to summary statistics
            idx = top_k_values.index(tk)
            summary_entry = summary[idx]
            # Handle both old and new format
            if "prefill" in summary_entry and isinstance(summary_entry["prefill"], dict):
                prefill_mean = summary_entry["prefill"].get("mean", 0)
                decode_mean = summary_entry["decode"].get("mean", 0)
            else:
                prefill_mean = summary_entry.get("prefill", {}).get("mean", 0) if isinstance(summary_entry.get("prefill"), dict) else 0
                decode_mean = summary_entry.get("decode", {}).get("mean", 0) if isinstance(summary_entry.get("decode"), dict) else 0
            
            vision_means.append(0)  # Vision not measured separately
            llm_prefill_means.append(prefill_mean)
            llm_decode_means.append(decode_mean)
            total_means.append(prefill_mean + decode_mean)
    
    # Color palette
    colors = {
        'vision': '#1F77B4',        # Deep blue
        'llm_prefill': '#FF7F0E',   # Bright orange
        'llm_decode': '#2CA02C',    # Deep green
        'total': '#D62728',         # Deep red
    }
    
    # Create two subplots: Prefill and Decode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(top_k_values))
    width = 0.6
    
    # Plot 1: Prefill Latency Breakdown
    ax1.bar(x, vision_means, width, label="Vision", 
           color=colors['vision'], edgecolor='black', linewidth=1.0)
    ax1.bar(x, llm_prefill_means, width, bottom=vision_means, 
           label="LLM Prefill", color=colors['llm_prefill'], edgecolor='black', linewidth=1.0)
    ax1.plot(x, [v + p for v, p in zip(vision_means, llm_prefill_means)], 
            "o-", label="Total Prefill", linewidth=2.5, markersize=8, 
            color=colors['total'], zorder=10)
    
    ax1.set_xlabel("Top-K", fontsize=14)
    ax1.set_ylabel("Latency (ms)", fontsize=14)
    ax1.set_title("Prefill Latency Breakdown", fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_k_values, fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, axis="y", alpha=0.3)
    
    # Plot 2: Decode Latency
    ax2.bar(x, llm_decode_means, width, label="LLM Decode", 
           color=colors['llm_decode'], edgecolor='black', linewidth=1.0)
    ax2.plot(x, llm_decode_means, "o-", label="Decode Latency", 
            linewidth=2.5, markersize=8, color=colors['total'], zorder=10)
    
    ax2.set_xlabel("Top-K", fontsize=14)
    ax2.set_ylabel("Latency (ms)", fontsize=14)
    ax2.set_title("Decode Latency", fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_k_values, fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "exp2_moe_topk_vs_latency_breakdown.png", dpi=300, bbox_inches="tight")
    log.info(f"Plot saved to {fig_dir / 'exp2_moe_topk_vs_latency_breakdown.png'}")
    plt.close()
    
    # Print Statistics
    log.info("=" * 60)
    log.info("MoE Top-K Analysis:")
    log.info(f"  Top-K range: {min(top_k_values)} - {max(top_k_values)}")
    log.info(f"  Prefill latency range: {min(prefill_means):.2f} - {max(prefill_means):.2f} ms")
    log.info(f"  Decode latency range: {min(decode_means):.2f} - {max(decode_means):.2f} ms")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot Profiling Experiment 2: MoE Top-K vs Latency")
    parser.add_argument("--json_file", type=str, required=True, 
                        help="Path to exp2_moe_topk_results.json")
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
    plot_moe_topk_vs_latency(data, output_dir)
    log.info("Plotting complete!")


if __name__ == "__main__":
    main()

