#!/usr/bin/env python3
"""
Plot script for Experiment 1: Latency Distribution
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


def compute_statistics(latencies: List[float]) -> Dict[str, float]:
    """Compute latency statistics."""
    latencies = np.array(latencies)
    return {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "P50": float(np.percentile(latencies, 50)),
        "P95": float(np.percentile(latencies, 95)),
        "P99": float(np.percentile(latencies, 99)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
    }


def plot_latency_distribution(results: List[Dict], output_dir: Path, dataset_name: str, split: str):
    """Plot latency distribution: histogram and CDF."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    latencies = [r["T_total"] for r in results]
    stats = compute_statistics(latencies)

    # Unified color palette (ggthemes Classic_10 - high saturation, colorblind-friendly)
    colors = {
        'primary': '#1F77B4',      # Deep blue
        'histogram': '#1F77B4',   # Deep blue for histograms
        'p50': '#1F77B4',          # Deep blue for P50
        'p95': '#FF7F0E',          # Bright orange for P95
        'p99': '#D62728',          # Deep red for P99
    }
    
    # Plot 1: Histogram (separate figure)
    plt.figure(figsize=(8, 6))
    plt.hist(latencies, bins=50, color=colors['histogram'], edgecolor='black', linewidth=1.5)
    plt.axvline(stats["P50"], color=colors['p50'], linestyle="--", linewidth=2, label=f"P50: {stats['P50']:.1f}ms")
    plt.axvline(stats["P95"], color=colors['p95'], linestyle="--", linewidth=2, label=f"P95: {stats['P95']:.1f}ms")
    plt.axvline(stats["P99"], color=colors['p99'], linestyle="--", linewidth=2, label=f"P99: {stats['P99']:.1f}ms")
    plt.xlabel("Latency (ms)", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.title("Latency Distribution", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / f"exp1_latency_distribution_histogram_{dataset_name}_{split}.png", dpi=300, bbox_inches="tight")
    log.info(f"Plot saved to {fig_dir / f'exp1_latency_distribution_histogram_{dataset_name}_{split}.png'}")
    plt.close()

    # Plot 2: CDF (separate figure)
    plt.figure(figsize=(8, 6))
    sorted_lats = np.sort(latencies)
    percentiles = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats) * 100
    plt.plot(sorted_lats, percentiles, linewidth=2.5, color=colors['primary'])
    plt.axvline(stats["P50"], color=colors['p50'], linestyle="--", linewidth=2, label=f"P50: {stats['P50']:.1f}ms")
    plt.axvline(stats["P95"], color=colors['p95'], linestyle="--", linewidth=2, label=f"P95: {stats['P95']:.1f}ms")
    plt.axvline(stats["P99"], color=colors['p99'], linestyle="--", linewidth=2, label=f"P99: {stats['P99']:.1f}ms")
    plt.xlabel("Latency (ms)", fontsize=16)
    plt.ylabel("Cumulative Percentage (%)", fontsize=16)
    plt.title("Cumulative Distribution Function", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / f"exp1_latency_distribution_cdf_{dataset_name}_{split}.png", dpi=300, bbox_inches="tight")
    log.info(f"Plot saved to {fig_dir / f'exp1_latency_distribution_cdf_{dataset_name}_{split}.png'}")
    plt.close()

    # Print Statistics
    log.info("=" * 60)
    log.info("Latency Statistics:")
    log.info(f"  Mean: {stats['mean']:.2f} ms")
    log.info(f"  Std:  {stats['std']:.2f} ms")
    log.info(f"  P50:  {stats['P50']:.2f} ms")
    log.info(f"  P95:  {stats['P95']:.2f} ms")
    log.info(f"  P99:  {stats['P99']:.2f} ms")
    log.info(f"  Min:  {stats['min']:.2f} ms")
    log.info(f"  Max:  {stats['max']:.2f} ms")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot Experiment 1: Latency Distribution")
    parser.add_argument("--json_file", type=str, required=True, help="Path to exp1_latency_distribution.json")
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
    plot_latency_distribution(results, output_dir, args.dataset, args.split)
    log.info("Plotting complete!")


if __name__ == "__main__":
    main()

