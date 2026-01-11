#!/usr/bin/env python3
"""
E1: Stage-Aware Latency Decomposition - Stack Bar Charts
Generate stacked bar charts showing stage time shares for each dataset and configuration.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import re

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_config_from_filename(filename: str) -> Dict[str, str]:
    """Parse configuration from filename.
    
    Example: coco-2014-vqa_imgsizetier-low_crops1_topk4_blocks12.json
    Returns: {
        'dataset': 'coco-2014-vqa',
        'tier': 'low',
        'crops': '1',
        'topk': '4',
        'blocks': '12'
    }
    """
    # Remove .json extension
    basename = filename.replace('.json', '')
    
    # Pattern: dataset_imgsizetier-{tier}_crops{crops}_topk{topk}_blocks{blocks}
    pattern = r'(.+?)_imgsizetier-(\w+)_crops(\d+)_topk(\d+)_blocks(\d+)'
    match = re.match(pattern, basename)
    
    if match:
        return {
            'dataset': match.group(1),
            'tier': match.group(2),
            'crops': match.group(3),
            'topk': match.group(4),
            'blocks': match.group(5)
        }
    else:
        # Try alternative pattern for datasets with different naming
        # e.g., tally-qa_imgsizetier-high_crops13_topk8_blocks16.json
        pattern2 = r'(.+?)_imgsizetier-(\w+)_crops(\d+)_topk(\d+)_blocks(\d+)'
        match2 = re.match(pattern2, basename)
        if match2:
            return {
                'dataset': match2.group(1),
                'tier': match2.group(2),
                'crops': match2.group(3),
                'topk': match2.group(4),
                'blocks': match2.group(5)
            }
        log.warning(f"Could not parse config from filename: {filename}")
        return {}


def load_latency_stats(json_path: Path) -> Dict[str, float]:
    """Load latency statistics from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    latency_stats = data.get('latency_stats', {})
    
    return {
        'T_vision_total': latency_stats.get('T_vision_total_mean', 0.0),
        'T_LLM_prefill': latency_stats.get('T_LLM_prefill_mean', 0.0),
        'T_LLM_decode': latency_stats.get('T_LLM_decode_mean', 0.0),
        'T_total': latency_stats.get('T_total_mean', 0.0),
    }


def create_config_label(config: Dict[str, str]) -> str:
    """Create a readable label for the configuration."""
    return f"Tier:{config['tier']}, Crops:{config['crops']}, TopK:{config['topk']}, Blocks:{config['blocks']}"


def create_short_config_label(config: Dict[str, str]) -> str:
    """Create a short label for the configuration (for x-axis)."""
    tier_short = config['tier'][0].upper()  # L, M, H
    return f"{tier_short}-{config['topk']}-{config['blocks']}"


def sort_configs(configs_with_data: List[Tuple[Dict[str, str], Dict[str, float]]]) -> List[Tuple[Dict[str, str], Dict[str, float]]]:
    """Sort configurations in a logical order for visualization.
    
    Sort by: tier (low, medium, high), then crops, then topk, then blocks.
    """
    tier_order = {'low': 0, 'medium': 1, 'high': 2}
    
    def sort_key(item):
        config, _ = item
        return (
            tier_order.get(config.get('tier', ''), 99),
            int(config.get('crops', 0)),
            int(config.get('topk', 0)),
            int(config.get('blocks', 0))
        )
    
    return sorted(configs_with_data, key=sort_key)


def plot_stage_latency_stack(
    latency_data: Dict[str, float],
    config: Dict[str, str],
    output_path: Path,
    dataset_name: str
):
    """Create a stacked bar chart for a single configuration.
    
    Args:
        latency_data: Dictionary with T_vision_total, T_LLM_prefill, T_LLM_decode, T_total
        config: Configuration dictionary
        output_path: Path to save the plot
        dataset_name: Name of the dataset
    """
    # Extract latency values
    vision_total = latency_data['T_vision_total']
    llm_prefill = latency_data['T_LLM_prefill']
    llm_decode = latency_data['T_LLM_decode']
    total = latency_data['T_total']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color palette (consistent with existing plots)
    colors = {
        'vision': '#1F77B4',      # Deep blue
        'prefill': '#FF7F0E',     # Bright orange
        'decode': '#2CA02C',      # Deep green
    }
    
    # Create single stacked bar
    x = 0
    width = 0.6
    
    # Stack bars: Vision Total at bottom, LLM Prefill in middle, LLM Decode on top
    p1 = ax.bar(
        x, vision_total, width,
        label='Vision Total (Encoder + Projector)',
        color=colors['vision'],
        edgecolor='black',
        linewidth=1.5
    )
    p2 = ax.bar(
        x, llm_prefill, width,
        bottom=vision_total,
        label='LLM Prefill',
        color=colors['prefill'],
        edgecolor='black',
        linewidth=1.5
    )
    p3 = ax.bar(
        x, llm_decode, width,
        bottom=vision_total + llm_prefill,
        label='LLM Decode',
        color=colors['decode'],
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add total latency line/marker on top
    ax.plot(
        [x - width/2, x + width/2],
        [total, total],
        'r-',
        linewidth=2.5,
        label='Total Latency',
        zorder=10
    )
    ax.scatter(
        x, total,
        color='red',
        s=100,
        zorder=11,
        edgecolors='black',
        linewidths=1.5
    )
    
    # Calculate percentages for annotation
    vision_pct = (vision_total / total) * 100 if total > 0 else 0
    prefill_pct = (llm_prefill / total) * 100 if total > 0 else 0
    decode_pct = (llm_decode / total) * 100 if total > 0 else 0
    
    # Add percentage annotations on each segment
    if vision_total > 0:
        ax.text(
            x, vision_total / 2,
            f'{vision_pct:.1f}%',
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='white'
        )
    if llm_prefill > 0:
        ax.text(
            x, vision_total + llm_prefill / 2,
            f'{prefill_pct:.1f}%',
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='white'
        )
    if llm_decode > 0:
        ax.text(
            x, vision_total + llm_prefill + llm_decode / 2,
            f'{decode_pct:.1f}%',
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='white'
        )
    
    # Add total latency value annotation
    ax.text(
        x, total + total * 0.05,
        f'Total: {total:.1f} ms',
        ha='center', va='bottom',
        fontsize=11, fontweight='bold',
        color='red'
    )
    
    # Set labels and title
    config_label = create_config_label(config)
    ax.set_xlabel('Configuration', fontsize=14)
    ax.set_ylabel('Latency (ms)', fontsize=14)
    ax.set_title(
        f'Stage Latency Decomposition\n{dataset_name}\n{config_label}',
        fontsize=14,
        pad=20
    )
    
    # Set x-axis
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([0])
    ax.set_xticklabels([config_label], rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Saved plot: {output_path}")


def plot_all_configs_stacked(
    configs_with_data: List[Tuple[Dict[str, str], Dict[str, float]]],
    output_path: Path,
    dataset_name: str
):
    """Create a stacked bar chart with all configurations for a dataset.
    
    Args:
        configs_with_data: List of (config, latency_data) tuples
        output_path: Path to save the plot
        dataset_name: Name of the dataset
    """
    # Sort configurations
    sorted_configs = sort_configs(configs_with_data)
    
    # Extract data
    configs = [c for c, _ in sorted_configs]
    latency_data_list = [d for _, d in sorted_configs]
    
    vision_totals = [d['T_vision_total'] for d in latency_data_list]
    llm_prefills = [d['T_LLM_prefill'] for d in latency_data_list]
    llm_decodes = [d['T_LLM_decode'] for d in latency_data_list]
    totals = [d['T_total'] for d in latency_data_list]
    
    # Create figure with appropriate size for 27 bars
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Color palette
    colors = {
        'vision': '#1F77B4',      # Deep blue
        'prefill': '#FF7F0E',     # Bright orange
        'decode': '#2CA02C',      # Deep green
    }
    
    # X positions for bars
    x = np.arange(len(configs))
    width = 0.7
    
    # Stack bars: Vision Total at bottom, LLM Prefill in middle, LLM Decode on top
    p1 = ax.bar(
        x, vision_totals, width,
        label='Vision Total (Encoder + Projector)',
        color=colors['vision'],
        edgecolor='black',
        linewidth=0.8
    )
    p2 = ax.bar(
        x, llm_prefills, width,
        bottom=vision_totals,
        label='LLM Prefill',
        color=colors['prefill'],
        edgecolor='black',
        linewidth=0.8
    )
    p3 = ax.bar(
        x, llm_decodes, width,
        bottom=np.array(vision_totals) + np.array(llm_prefills),
        label='LLM Decode',
        color=colors['decode'],
        edgecolor='black',
        linewidth=0.8
    )
    
    # Add total latency line on top
    ax.plot(
        x, totals,
        'ro-',
        linewidth=2.0,
        markersize=4,
        label='Total Latency',
        zorder=10
    )
    
    # Set labels and title
    ax.set_xlabel('Configuration', fontsize=14)
    ax.set_ylabel('Latency (ms)', fontsize=14)
    ax.set_title(
        f'Stage Latency Decomposition - All Configurations\n{dataset_name}',
        fontsize=16,
        pad=20
    )
    
    # Set x-axis labels
    short_labels = [create_short_config_label(c) for c in configs]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlim(-0.5, len(configs) - 0.5)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Saved combined plot: {output_path}")


def process_dataset(
    dataset_dir: Path,
    output_base_dir: Path
):
    """Process all JSON files in a dataset directory."""
    dataset_name = dataset_dir.name
    log.info(f"Processing dataset: {dataset_name}")
    
    # Find all JSON files
    json_files = sorted(dataset_dir.glob("*.json"))
    
    if not json_files:
        log.warning(f"No JSON files found in {dataset_dir}")
        return
    
    log.info(f"Found {len(json_files)} configuration files")
    
    # Create output directory for this dataset
    dataset_output_dir = output_base_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all configurations and their data
    configs_with_data = []
    
    # Process each configuration
    for json_file in json_files:
        try:
            # Parse configuration from filename
            config = parse_config_from_filename(json_file.name)
            if not config:
                log.warning(f"Skipping {json_file.name} - could not parse config")
                continue
            
            # Load latency stats
            latency_data = load_latency_stats(json_file)
            
            # Store for combined plot
            configs_with_data.append((config, latency_data))
            
            # Create individual plot (optional - can be disabled)
            # Create output filename
            config_str = f"{config['tier']}_crops{config['crops']}_topk{config['topk']}_blocks{config['blocks']}"
            output_filename = f"stage_latency_stack_{config_str}.png"
            output_path = dataset_output_dir / output_filename
            
            # Create individual plot
            plot_stage_latency_stack(
                latency_data,
                config,
                output_path,
                dataset_name
            )
            
        except Exception as e:
            log.error(f"Error processing {json_file}: {e}", exc_info=True)
    
    # Create combined plot with all configurations
    if configs_with_data:
        combined_output_path = dataset_output_dir / "stage_latency_stack_all_configs.png"
        plot_all_configs_stacked(
            configs_with_data,
            combined_output_path,
            dataset_name
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate E1 stage latency decomposition stack bar charts"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/home/x-pwang1/ai_project/molmo_hf/results/core_exp_h100/4run_2000samples"),
        help="Directory containing dataset subdirectories with JSON results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/x-pwang1/ai_project/molmo_hf/results/analysis_output/e1_stage_latency_stacks"),
        help="Output directory for plots"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Specific datasets to process (default: all)"
    )
    
    args = parser.parse_args()
    
    # Find all dataset directories
    if args.datasets:
        dataset_dirs = [args.results_dir / d for d in args.datasets]
    else:
        dataset_dirs = [
            d for d in args.results_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.') and d.name != 'logs'
        ]
    
    log.info(f"Found {len(dataset_dirs)} datasets to process")
    
    # Process each dataset
    for dataset_dir in sorted(dataset_dirs):
        if not dataset_dir.exists():
            log.warning(f"Dataset directory does not exist: {dataset_dir}")
            continue
        process_dataset(dataset_dir, args.output_dir)
    
    log.info(f"All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

