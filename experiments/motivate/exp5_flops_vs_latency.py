#!/usr/bin/env python3
"""
Experiment 5: FLOPs vs Latency

Goal: Analyze FLOPs-latency correlation to show FLOPs cannot accurately predict single-request latency.
Method: Use results from Exp 3 and Exp 4 to plot FLOPs vs Latency.
Output: Scatter plots with correlation coefficients and R² values.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.stats import linregress

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def format_flops(x, pos):
    """Format FLOPs as numbers without unit (unit is in axis label).
    
    Converts to trillions (T) for display, but only shows the number.
    """
    if x >= 1e12:
        return f'{x/1e12:.1f}'
    elif x >= 1e9:
        return f'{x/1e9:.1f}'
    elif x >= 1e6:
        return f'{x/1e6:.1f}'
    else:
        return f'{x:.0f}'


def analyze_flops_vs_latency(exp3_results_path: str, exp4_results_path: str, output_dir: str, x_axis: str = "flops"):
    """Analyze FLOPs vs Latency from Exp 3 and Exp 4 results.
    
    Args:
        exp3_results_path: Path to Exp 3 results JSON
        exp4_results_path: Path to Exp 4 results JSON
        output_dir: Output directory for plots and results
        x_axis: X-axis type, either "flops" or "tokens" (default: "flops")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        exp3_data = load_json(exp3_results_path)
        exp4_data = load_json(exp4_results_path)
    except FileNotFoundError as e:
        log.error(f"Could not load data: {e}")
        return

    exp3_results = exp3_data.get("results", [])
    exp4_results = exp4_data.get("results", [])

    # Process Exp 3 (Vision Scaling)
    exp3_flops = [r.get("flops_total", 0) for r in exp3_results]
    exp3_lats = [r.get("T_total", 0) for r in exp3_results]
    exp3_vision_tokens = [r.get("num_vision_tokens", 0) for r in exp3_results]

    # Process Exp 4 (Language Scaling)
    # Group by max_new_tokens and average
    exp4_groups = {}
    for r in exp4_results:
        mt = r.get("max_new_tokens", 0)
        if mt not in exp4_groups:
            exp4_groups[mt] = []
        exp4_groups[mt].append(r)

    exp4_max_tokens = sorted(exp4_groups.keys())
    exp4_flops = [np.mean([r.get("flops_total", 0) for r in exp4_groups[mt]]) for mt in exp4_max_tokens]
    exp4_lats = [np.mean([r.get("T_total", 0) for r in exp4_groups[mt]]) for mt in exp4_max_tokens]
    exp4_output_tokens = [np.mean([r.get("num_output_tokens", 0) for r in exp4_groups[mt]]) for mt in exp4_max_tokens]

    # Unified color palette (ggthemes Classic_10 - high saturation, colorblind-friendly)
    # Official Classic_10: https://emilhvitfeldt.github.io/r-color-palettes/discrete/ggthemes/Classic_10/
    colors = {
        'primary': '#1F77B4',      # Deep blue (Classic_10 color 1)
        'secondary': '#FF7F0E',    # Bright orange (Classic_10 color 2)
        'tertiary': '#2CA02C',     # Deep green (Classic_10 color 3)
    }
    
    # Plot 1: Vision Scaling (separate figure)
    # Choose X-axis data
    if x_axis == "tokens":
        exp3_x = exp3_vision_tokens
        exp3_x_label = "Number of input vision tokens"
        exp3_title = None  # No title needed
    else:
        exp3_x = exp3_flops
        # Determine unit based on data range
        max_flops = max(exp3_flops) if exp3_flops else 0
        if max_flops >= 1e12:
            exp3_x_label = "FLOPs (T)"
        elif max_flops >= 1e9:
            exp3_x_label = "FLOPs (B)"
        else:
            exp3_x_label = "FLOPs"
        exp3_title = None  # No title needed
    
    plt.figure(figsize=(8, 6))
    plt.scatter(exp3_x, exp3_lats, s=120, color=colors['primary'], label="Vision Scaling")
    plt.xlabel(exp3_x_label, fontsize=16)
    plt.ylabel("Latency (ms)", fontsize=16)
    if exp3_title:
        plt.title(exp3_title, fontsize=18)
    
    # Format X-axis if showing FLOPs
    if x_axis == "flops":
        plt.gca().xaxis.set_major_formatter(FuncFormatter(format_flops))
    
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=14)

    # Linear fit
    if len(exp3_x) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(exp3_x, exp3_lats)
        x_fit = np.linspace(min(exp3_x), max(exp3_x), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, "r--", linewidth=2.5, label=f"Linear Fit (R²={r_value**2:.3f})")
        plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(fig_dir / "exp5_flops_vs_latency_vision.png", dpi=300, bbox_inches="tight")
    log.info(f"Plot saved to {fig_dir / 'exp5_flops_vs_latency_vision.png'}")
    plt.close()

    # Plot 2: Language Scaling (separate figure)
    # Convert latencies from ms to seconds for language scaling plot
    exp4_lats_seconds = [lat / 1000.0 for lat in exp4_lats]
    
    # Choose X-axis data
    if x_axis == "tokens":
        exp4_x = exp4_output_tokens
        exp4_x_label = "Number of generated language tokens"
        exp4_title = None  # No title needed
    else:
        exp4_x = exp4_flops
        # Determine unit based on data range
        max_flops = max(exp4_flops) if exp4_flops else 0
        if max_flops >= 1e12:
            exp4_x_label = "FLOPs (T)"
        elif max_flops >= 1e9:
            exp4_x_label = "FLOPs (B)"
        else:
            exp4_x_label = "FLOPs"
        exp4_title = None  # No title needed
    
    plt.figure(figsize=(8, 6))
    plt.scatter(exp4_x, exp4_lats_seconds, s=120, color=colors['tertiary'], label="Language Scaling")  # Use green color
    plt.xlabel(exp4_x_label, fontsize=16)
    plt.ylabel("Latency (seconds)", fontsize=16)
    if exp4_title:
        plt.title(exp4_title, fontsize=18)
    
    # Format X-axis
    if x_axis == "flops":
        plt.gca().xaxis.set_major_formatter(FuncFormatter(format_flops))
    elif x_axis == "tokens":
        # Use integer format for tokens
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
    
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=14)

    # Linear fit
    if len(exp4_x) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(exp4_x, exp4_lats_seconds)
        x_fit = np.linspace(min(exp4_x), max(exp4_x), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, "r--", linewidth=2.5, label=f"Linear Fit (R²={r_value**2:.3f})")
        plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(fig_dir / "exp5_flops_vs_latency_language.png", dpi=300, bbox_inches="tight")
    log.info(f"Plot saved to {fig_dir / 'exp5_flops_vs_latency_language.png'}")
    plt.close()

    # Compute and save statistics
    stats = {}
    if len(exp3_x) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(exp3_x, exp3_lats)
        stats["exp3"] = {
            "r_squared": float(r_value**2),
        }

    if len(exp4_x) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(exp4_x, exp4_lats_seconds)
        stats["exp4"] = {
            "r_squared": float(r_value**2),
        }

    # Save results
    results = {
        "exp3": {
            "flops": exp3_flops,
            "latencies": exp3_lats,
            "vision_tokens": exp3_vision_tokens,
        },
        "exp4": {
            "flops": exp4_flops,
            "latencies": exp4_lats,
            "output_tokens": exp4_output_tokens,
        },
        "statistics": stats,
    }

    output_path = output_dir / "exp5_flops_vs_latency.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {output_path}")

    # Print Statistics
    log.info("=" * 60)
    log.info("FLOPs vs Latency Correlation:")
    if "exp3" in stats:
        log.info(f"  Exp 3 (Vision): R² = {stats['exp3']['r_squared']:.3f}")
    if "exp4" in stats:
        log.info(f"  Exp 4 (Language): R² = {stats['exp4']['r_squared']:.3f}")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: FLOPs vs Latency")
    parser.add_argument(
        "--exp3_results",
        type=str,
        required=True,
        help="Path to Exp 3 results JSON file",
    )
    parser.add_argument(
        "--exp4_results",
        type=str,
        required=True,
        help="Path to Exp 4 results JSON file",
    )
    parser.add_argument("--output_dir", type=str, default="./results/motivation/exp5", help="Output directory")
    parser.add_argument(
        "--x_axis",
        type=str,
        default="flops",
        choices=["flops", "tokens"],
        help="X-axis type: 'flops' or 'tokens' (default: 'flops')",
    )

    args = parser.parse_args()

    analyze_flops_vs_latency(args.exp3_results, args.exp4_results, args.output_dir, args.x_axis)


if __name__ == "__main__":
    main()

