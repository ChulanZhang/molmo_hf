#!/usr/bin/env python3
"""
Plot script for Profiling Experiment 1: Vision Tokens vs Latency
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


def plot_vision_tokens_vs_latency(data: Dict, output_dir: Path):
    """Plot vision tokens vs latency using stacked bar chart."""
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
    
    # Sort by vision tokens
    summary = sorted(summary, key=lambda x: x.get("num_vision_tokens", 0))
    
    vision_tokens = [r.get("num_vision_tokens", 0) for r in summary]
    
    # Group samples by vision tokens to compute component means
    groups = {}
    for sample in all_samples:
        vt = sample.get("num_vision_tokens", 0)
        if vt not in groups:
            groups[vt] = []
        groups[vt].append(sample)
    
    # Compute component latencies for each vision token count
    vision_encoder_means = []
    projector_means = []
    prefill_only_means = []
    prefill_total_means = []
    
    for vt in vision_tokens:
        if vt in groups and groups[vt]:
            # Use per-sample data if available
            samples = groups[vt]
            vision_encoder_means.append(np.mean([s.get("T_vision_encoder", 0) for s in samples]))
            projector_means.append(np.mean([s.get("T_projector", 0) for s in samples]))
            prefill_only_means.append(np.mean([s.get("T_LLM_prefill", 0) for s in samples]))
            prefill_total_means.append(
                np.mean([s.get("T_vision_total", 0) + s.get("T_LLM_prefill", 0) for s in samples])
            )
        else:
            # Fallback to summary statistics
            idx = vision_tokens.index(vt)
            summary_entry = summary[idx]
            # Handle both old and new format
            if "prefill" in summary_entry and isinstance(summary_entry["prefill"], dict):
                prefill_mean = summary_entry["prefill"].get("mean", 0)
            else:
                prefill_mean = summary_entry.get("prefill", {}).get("mean", 0) if isinstance(summary_entry.get("prefill"), dict) else 0
            
            vision_encoder_means.append(0)  # Not available in summary
            projector_means.append(0)  # Not available in summary
            prefill_only_means.append(prefill_mean)
            prefill_total_means.append(prefill_mean)
    
    # Unified color palette
    colors = {
        'vision_encoder': '#1F77B4',      # Deep blue
        'projector': '#FF7F0E',          # Bright orange
        'llm_prefill': '#2CA02C',        # Deep green
        'total': '#D62728',              # Deep red
    }
    
    # Stacked bar chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(vision_tokens))
    width = 0.6
    
    # Stack bars: Vision Encoder at bottom, Projector on top, LLM Prefill on top of that
    p1 = plt.bar(x, vision_encoder_means, width, label="Vision Encoder", 
                color=colors['vision_encoder'], edgecolor='black', linewidth=1.0)
    p2 = plt.bar(x, projector_means, width, bottom=vision_encoder_means, 
                label="Projector", color=colors['projector'], edgecolor='black', linewidth=1.0)
    p3 = plt.bar(x, prefill_only_means, width, 
                bottom=np.array(vision_encoder_means) + np.array(projector_means), 
                label="LLM Prefill", color=colors['llm_prefill'], edgecolor='black', linewidth=1.0)
    
    # Add line showing Time to First Token (total prefill)
    plt.plot(x, prefill_total_means, "o-", label="Time to First Token", 
            linewidth=2.5, markersize=8, color=colors['total'], zorder=10)
    
    plt.xlabel("Vision Tokens", fontsize=14)
    plt.ylabel("Latency (ms)", fontsize=14)
    plt.title("Profiling Exp 1: Vision Tokens vs Latency Breakdown", fontsize=16)
    plt.xticks(x, vision_tokens, fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "exp1_vision_tokens_vs_latency_breakdown.png", dpi=300, bbox_inches="tight")
    log.info(f"Plot saved to {fig_dir / 'exp1_vision_tokens_vs_latency_breakdown.png'}")
    plt.close()
    
    # Print Statistics
    log.info("=" * 60)
    log.info("Vision Token Analysis:")
    log.info(f"  Vision tokens range: {min(vision_tokens)} - {max(vision_tokens)}")
    log.info(f"  Prefill latency range: {min(prefill_total_means):.2f} - {max(prefill_total_means):.2f} ms")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot Profiling Experiment 1: Vision Tokens vs Latency")
    parser.add_argument("--json_file", type=str, required=True, 
                        help="Path to exp1_context_scaling_results.json")
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
    plot_vision_tokens_vs_latency(data, output_dir)
    log.info("Plotting complete!")


if __name__ == "__main__":
    main()

