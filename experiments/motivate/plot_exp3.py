#!/usr/bin/env python3
"""
Plot script for Experiment 3: Vision Tokens vs Latency
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


def plot_vision_tokens_vs_latency(results: List[Dict], output_dir: Path):
    """Plot vision tokens vs latency."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    vision_tokens = [r["num_vision_tokens"] for r in results]
    prefill_lats = [
        r.get("T_vision_total", 0) + r.get("T_LLM_prefill", 0) for r in results
    ]
    vision_encoder_lats = [r.get("T_vision_encoder", 0) for r in results]
    projector_lats = [r.get("T_projector", 0) for r in results]
    prefill_only_lats = [r.get("T_LLM_prefill", 0) for r in results]

    # Unified color palette
    colors = {
        'primary': '#1F77B4',      # Deep blue
        'secondary': '#FF7F0E',    # Bright orange
        'tertiary': '#2CA02C',     # Deep green
        'quaternary': '#D62728',   # Deep red
    }
    
    # Combined plot: Stacked bar chart with Time to First Token line on top
    plt.figure(figsize=(8, 6))
    x = np.arange(len(vision_tokens))
    width = 0.6
    
    # Stack bars: Vision Encoder at bottom, Projector on top, LLM Prefill on top of that
    p1 = plt.bar(x, vision_encoder_lats, width, label="Vision Encoder", color=colors['primary'], edgecolor='black', linewidth=1.0)
    p2 = plt.bar(x, projector_lats, width, bottom=vision_encoder_lats, label="Projector", color=colors['secondary'], edgecolor='black', linewidth=1.0)
    p3 = plt.bar(x, prefill_only_lats, width, bottom=np.array(vision_encoder_lats) + np.array(projector_lats), 
                 label="LLM", color=colors['tertiary'], edgecolor='black', linewidth=1.0)
    
    # Add line showing Time to First Token
    plt.plot(x, prefill_lats, "o-", label="Time to First Token", linewidth=2.5, markersize=8, 
            color=colors['quaternary'], zorder=10)
    
    plt.xlabel("Input Vision Tokens", fontsize=16)
    plt.ylabel("Latency (ms)", fontsize=16)
    plt.title("Latency Breakdown", fontsize=18)
    plt.xticks(x, vision_tokens, fontsize=14)
    
    # Set Y-axis to show 400ms tick
    ax = plt.gca()
    current_ticks = ax.get_yticks()
    # Add 400 to the ticks if not already present
    if 400 not in current_ticks:
        new_ticks = sorted(list(current_ticks) + [400])
        ax.set_yticks(new_ticks)
    # Set maximum to 401 to ensure 400 is visible
    current_ylim = ax.get_ylim()
    plt.ylim(bottom=current_ylim[0], top=max(401, current_ylim[1]))
    
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "exp3_vision_tokens_vs_latency_breakdown.png", dpi=300, bbox_inches="tight")
    log.info(f"Plot saved to {fig_dir / 'exp3_vision_tokens_vs_latency_breakdown.png'}")
    plt.close()

    # Print Statistics
    log.info("=" * 60)
    log.info("Vision Token Analysis:")
    log.info(f"  Vision tokens range: {min(vision_tokens)} - {max(vision_tokens)}")
    log.info(f"  Prefill latency range: {min(prefill_lats):.2f} - {max(prefill_lats):.2f} ms")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot Experiment 3: Vision Tokens vs Latency")
    parser.add_argument("--json_file", type=str, required=True, help="Path to exp3_vision_tokens_vs_latency.json")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as JSON file)")

    args = parser.parse_args()

    # Load results
    json_path = Path(args.json_file)
    if not json_path.exists():
        log.error(f"JSON file not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        log.error("No results found in JSON file")
        return

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = json_path.parent

    log.info(f"Loaded {len(results)} results from {json_path}")
    log.info(f"Output directory: {output_dir}")

    # Plot
    plot_vision_tokens_vs_latency(results, output_dir)
    log.info("Plotting complete!")


if __name__ == "__main__":
    main()

