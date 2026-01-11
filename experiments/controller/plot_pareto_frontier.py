"""
Plot Pareto Frontier curves from evaluation results.

This script reads Pareto data from evaluation results and creates
visualization plots showing accuracy-latency trade-off curves.

Usage:
    python experiments/controller/plot_pareto_frontier.py \
        --pareto_data ./results/logs_eval/pareto_frontier/pareto_data.json \
        --output_dir ./plots/pareto_frontier/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend
matplotlib.use('Agg')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


def load_pareto_data(pareto_file: Path) -> Dict[str, Any]:
    """Load Pareto data from JSON file."""
    with open(pareto_file, 'r') as f:
        data = json.load(f)
    return data


def plot_dataset_pareto(
    dataset_name: str,
    all_points: List[Dict],
    pareto_points: List[Dict],
    output_file: Path,
    figsize: Tuple[int, int] = (10, 7),
    dpi: int = 300,
):
    """
    Plot Pareto frontier for a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        all_points: All evaluation points
        pareto_points: Pareto frontier points
        output_file: Output file path for the plot
        figsize: Figure size (width, height)
        dpi: DPI for the figure
    """
    # Extract data
    all_latencies = [p['latency'] for p in all_points]
    all_accuracies = [p['accuracy'] for p in all_points]
    
    pareto_latencies = [p['latency'] for p in pareto_points]
    pareto_accuracies = [p['accuracy'] for p in pareto_points]
    
    # Color palette
    colors = {
        'non_pareto': '#7F7F7F',      # Gray
        'pareto': '#D62728',          # Deep red
        'pareto_line': '#D62728',     # Deep red
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot non-Pareto points
    if len(all_points) > len(pareto_points):
        non_pareto_latencies = []
        non_pareto_accuracies = []
        pareto_set = {(p['latency'], p['accuracy']) for p in pareto_points}
        
        for p in all_points:
            if (p['latency'], p['accuracy']) not in pareto_set:
                non_pareto_latencies.append(p['latency'])
                non_pareto_accuracies.append(p['accuracy'])
        
        if non_pareto_latencies:
            ax.scatter(
                non_pareto_latencies,
                non_pareto_accuracies,
                c=colors['non_pareto'],
                s=100,
                alpha=0.5,
                label='Non-Pareto',
                edgecolors='black',
                linewidths=1.0,
                zorder=1,
            )
    
    # Plot Pareto points
    if pareto_points:
        ax.scatter(
            pareto_latencies,
            pareto_accuracies,
            c=colors['pareto'],
            s=200,
            alpha=0.9,
            label='Pareto Frontier',
            edgecolors='black',
            linewidths=1.5,
            marker='*',
            zorder=5,
        )
        
        # Draw Pareto frontier line
        if len(pareto_points) > 1:
            ax.plot(
                pareto_latencies,
                pareto_accuracies,
                color=colors['pareto_line'],
                linestyle='--',
                linewidth=2.5,
                alpha=0.8,
                label='Pareto Frontier Line',
                zorder=4,
            )
        
        # Add annotations for Pareto points
        for point in pareto_points:
            config = point.get('config', {})
            label = f"({config.get('tier', '?')},{config.get('top_k', '?')},{config.get('num_active_blocks', '?')})"
            ax.annotate(
                label,
                (point['latency'], point['accuracy']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    alpha=0.9,
                    edgecolor='black',
                    linewidth=1.0
                ),
                zorder=6,
            )
    
    # Set labels and title
    ax.set_xlabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Pareto Frontier: {dataset_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    log.info(f"Plot saved to: {output_file}")


def plot_all_datasets(
    pareto_data: Dict[str, Any],
    output_dir: Path,
    figsize: Tuple[int, int] = (10, 7),
    dpi: int = 300,
):
    """
    Plot Pareto frontiers for all datasets.
    
    Args:
        pareto_data: Pareto data dictionary
        output_dir: Output directory for plots
        figsize: Figure size (width, height)
        dpi: DPI for the figure
    """
    all_points = pareto_data.get('all_points', {})
    pareto_frontiers = pareto_data.get('pareto_frontiers', {})
    
    # Plot each dataset separately
    for dataset_name in all_points.keys():
        points = all_points[dataset_name]
        frontier = pareto_frontiers.get(dataset_name, [])
        
        output_file = output_dir / f"{dataset_name}_pareto_frontier.png"
        plot_dataset_pareto(
            dataset_name=dataset_name,
            all_points=points,
            pareto_points=frontier,
            output_file=output_file,
            figsize=figsize,
            dpi=dpi,
        )
    
    # Plot all datasets together for comparison
    if len(all_points) > 1:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
        
        # Color palette for multiple datasets
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_points)))
        
        for idx, (dataset_name, points) in enumerate(all_points.items()):
            frontier = pareto_frontiers.get(dataset_name, [])
            
            if frontier:
                latencies = [p['latency'] for p in frontier]
                accuracies = [p['accuracy'] for p in frontier]
                
                ax.plot(
                    latencies,
                    accuracies,
                    marker='*',
                    markersize=10,
                    linewidth=2.5,
                    label=f'{dataset_name} (Pareto)',
                    color=colors[idx],
                    alpha=0.8,
                    zorder=5,
                )
        
        ax.set_xlabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Pareto Frontiers: All Datasets', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        comparison_file = output_dir / "all_datasets_comparison.png"
        plt.savefig(comparison_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        log.info(f"Comparison plot saved to: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Pareto Frontier curves from evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--pareto_data",
        type=str,
        required=True,
        help="Path to Pareto data JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/visualizations/pareto_frontier/",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[10, 7],
        help="Figure size (width, height)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for the figure"
    )
    
    args = parser.parse_args()
    
    pareto_file = Path(args.pareto_data)
    if not pareto_file.exists():
        log.error(f"Pareto data file not found: {pareto_file}")
        return
    
    log.info("=" * 80)
    log.info("Plotting Pareto Frontiers")
    log.info("=" * 80)
    log.info(f"Pareto data: {pareto_file}")
    log.info(f"Output directory: {args.output_dir}")
    log.info("=" * 80)
    
    # Load Pareto data
    log.info("Loading Pareto data...")
    pareto_data = load_pareto_data(pareto_file)
    
    # Print summary
    summary = pareto_data.get('summary', {})
    log.info("\nSummary:")
    for dataset, stats in summary.items():
        log.info(f"  {dataset}:")
        log.info(f"    Total points: {stats['total_points']}")
        log.info(f"    Pareto points: {stats['pareto_points']}")
        log.info(f"    Latency range: {stats['latency_range']}")
        log.info(f"    Accuracy range: {stats['accuracy_range']}")
    
    # Plot
    output_dir = Path(args.output_dir)
    log.info("\nGenerating plots...")
    plot_all_datasets(
        pareto_data=pareto_data,
        output_dir=output_dir,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )
    
    log.info("\n" + "=" * 80)
    log.info("Plotting Complete")
    log.info("=" * 80)
    log.info(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

