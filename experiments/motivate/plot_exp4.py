#!/usr/bin/env python3
"""
Plot script for Experiment 4: Language Tokens vs Latency
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


def plot_language_tokens_vs_latency(results: List[Dict], output_dir: Path, dataset_name: str, split: str):
    """Plot language tokens vs latency."""
    # Group by max_new_tokens
    groups = {}
    for r in results:
        mt = r.get("max_new_tokens")
        if mt is None:
            continue
        if mt not in groups:
            groups[mt] = []
        groups[mt].append(r)
        
    if not groups:
        log.error("No valid results with max_new_tokens found")
        return

    max_tokens_list = sorted(groups.keys())
    prefill_means = [np.mean([r["T_LLM_prefill"] + r.get("T_vision_total", 0) for r in groups[mt]]) for mt in max_tokens_list]
    decode_means = [np.mean([r["T_LLM_decode"] for r in groups[mt]]) for mt in max_tokens_list]
    decode_stds = [np.std([r["T_LLM_decode"] for r in groups[mt]]) for mt in max_tokens_list]
    total_means = [np.mean([r["T_total"] for r in groups[mt]]) for mt in max_tokens_list]
    output_tokens_means = [np.mean([r["num_output_tokens"] for r in groups[mt]]) for mt in max_tokens_list]
    prefill_means = [np.mean([r["T_LLM_prefill"] + r["T_vision_total"] for r in groups[mt]]) for mt in max_tokens_list]

    # Color palette: blue for prefill, green for decode
    colors = {
        'prefill': '#1F77B4',      # Deep blue for prefill
        'decode': '#2CA02C',       # Deep green for decode
        'total': '#D62728',        # Deep red for total latency line
    }
    
    # Combined plot: Stacked Bar Chart (Prefill + Decode) with Total Latency line
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Convert latencies from ms to seconds for better visibility
    prefill_means_sec = [v / 1000.0 for v in prefill_means]
    decode_means_sec = [v / 1000.0 for v in decode_means]
    total_means_sec = [v / 1000.0 for v in total_means]

    plt.figure(figsize=(8, 6))  # Match other experiment figures
    x = np.arange(len(max_tokens_list))
    width = 0.6

    # In log scale, stacked bars need special handling
    # Prefill is relatively constant (~305ms = 0.305s)
    # Use actual prefill values - they are relatively fixed around 0.3s
    p1 = plt.bar(x, prefill_means_sec, width, label="Prefill (Vision + LLM)", color=colors['prefill'], edgecolor='black', linewidth=1.5)
    p2 = plt.bar(x, decode_means_sec, width, bottom=prefill_means_sec, label="Decode (LLM only)", color=colors['decode'], edgecolor='black', linewidth=1.5)

    # Add line showing Total Latency
    plt.errorbar(
        x,
        total_means_sec,
        yerr=None,
        fmt="o-",
        capsize=5,
        linewidth=2.5,
        markersize=8,
            label="Total Latency",
            color=colors['total'],
            zorder=10
        )

    plt.xlabel("Max Decode Output Tokens", fontsize=16)
    plt.ylabel("Latency (seconds)", fontsize=16, labelpad=3)  # Reduce distance from axis
    plt.title("Latency Breakdown", fontsize=18)
    # Use integer labels for x-axis
    plt.xticks(x, [int(round(v)) for v in output_tokens_means], fontsize=14)
    
    # Use log scale for better visibility of both prefill and decode
    plt.yscale('log')
    plt.ylim(bottom=0.1, top=1000)  # Start from 0.1 seconds, max 1000 seconds
    
    # Format Y-axis labels to be more readable (show as seconds with 1-2 decimal places)
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
    
    # Add a tick at 0.3 seconds (300ms) to highlight prefill latency
    ax = plt.gca()
    current_ticks = ax.get_yticks()
    # Add 0.3 to the ticks if not already present
    if 0.3 not in current_ticks:
        new_ticks = sorted(list(current_ticks) + [0.3])
        ax.set_yticks(new_ticks)
    
    # Ensure Y-axis still starts from 0.1 and ends at 1000 after setting ticks
    plt.ylim(bottom=0.1, top=1000)
    
    plt.yticks(fontsize=14)
    
    plt.legend(fontsize=14, loc='upper left')
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / f"exp4_language_tokens_vs_latency_breakdown_{dataset_name}_{split}.png", dpi=300, bbox_inches="tight")
    log.info(f"Plot saved to {fig_dir / f'exp4_language_tokens_vs_latency_breakdown_{dataset_name}_{split}.png'}")
    plt.close()

    # Print Statistics
    log.info("=" * 60)
    log.info("Language Token Analysis:")
    log.info(f"  Output tokens range: {min(output_tokens_means):.0f} - {max(output_tokens_means):.0f}")
    log.info(f"  Decode latency range: {min(decode_means):.2f} - {max(decode_means):.2f} ms")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot Experiment 4: Language Tokens vs Latency")
    parser.add_argument("--json_file", type=str, required=True, help="Path to exp4_language_tokens_vs_latency.json")
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
    plot_language_tokens_vs_latency(results, output_dir, args.dataset, args.split)
    log.info("Plotting complete!")


if __name__ == "__main__":
    main()

