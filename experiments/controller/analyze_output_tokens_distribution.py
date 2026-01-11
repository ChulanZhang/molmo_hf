"""
Analyze output tokens distribution across all datasets.
This helps understand if short answers (e.g., 1-2 tokens) are affecting decode per-token latency statistics.
"""

import argparse
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def load_all_samples(results_dir: Path, dataset_names: list = None):
    """Load all per-sample results from JSON files."""
    all_samples = []
    
    if dataset_names is None:
        # Auto-detect all datasets
        dataset_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name != 'logs']
        dataset_names = [d.name.replace('-', '_') for d in dataset_dirs]
        log.info(f"Auto-detected datasets: {dataset_names}")
    
    for dataset_name in dataset_names:
        # Convert dataset name to directory format (underscore to hyphen)
        dataset_dir_name = dataset_name.replace('_', '-')
        dataset_path = results_dir / dataset_dir_name
        
        if not dataset_path.exists():
            log.warning(f"Dataset directory not found: {dataset_path}")
            continue
        
        log.info(f"Loading samples from {dataset_name}...")
        json_files = list(dataset_path.glob("*.json"))
        
        if not json_files:
            log.warning(f"No JSON files found in {dataset_path}")
            continue
        
        dataset_samples = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'per_sample_results' in data:
                        for sample in data['per_sample_results']:
                            sample['dataset'] = dataset_name
                            dataset_samples.append(sample)
            except Exception as e:
                log.warning(f"Error loading {json_file}: {e}")
                continue
        
        log.info(f"  Loaded {len(dataset_samples)} samples from {dataset_name}")
        all_samples.extend(dataset_samples)
    
    log.info(f"Total samples loaded: {len(all_samples)}")
    return all_samples


def analyze_output_tokens_distribution(samples: list, output_dir: Path):
    """Analyze output tokens distribution and its relationship with decode latency."""
    
    # Extract output tokens
    output_tokens_list = [s.get('output_tokens', 0) for s in samples]
    output_tokens_array = np.array(output_tokens_list)
    
    # Basic statistics
    log.info("=" * 80)
    log.info("Output Tokens Distribution Analysis")
    log.info("=" * 80)
    log.info(f"Total samples: {len(samples)}")
    log.info(f"Output tokens - Min: {output_tokens_array.min()}, Max: {output_tokens_array.max()}")
    log.info(f"Output tokens - Mean: {output_tokens_array.mean():.2f}, Median: {np.median(output_tokens_array):.2f}")
    log.info(f"Output tokens - Std: {output_tokens_array.std():.2f}")
    log.info(f"Output tokens - P25: {np.percentile(output_tokens_array, 25):.2f}, P75: {np.percentile(output_tokens_array, 75):.2f}")
    log.info(f"Output tokens - P95: {np.percentile(output_tokens_array, 95):.2f}, P99: {np.percentile(output_tokens_array, 99):.2f}")
    
    # Count by ranges
    log.info("\nOutput Tokens Distribution by Ranges:")
    ranges = [
        (1, 1, "Exactly 1 token"),
        (2, 2, "Exactly 2 tokens"),
        (1, 2, "1-2 tokens (short answers)"),
        (3, 5, "3-5 tokens"),
        (6, 10, "6-10 tokens"),
        (11, 20, "11-20 tokens"),
        (21, 50, "21-50 tokens"),
        (51, float('inf'), "51+ tokens"),
    ]
    
    range_counts = {}
    for min_val, max_val, label in ranges:
        if max_val == float('inf'):
            count = np.sum((output_tokens_array >= min_val))
        else:
            count = np.sum((output_tokens_array >= min_val) & (output_tokens_array <= max_val))
        percentage = (count / len(output_tokens_array)) * 100
        range_counts[label] = (count, percentage)
        log.info(f"  {label:30s}: {count:6d} samples ({percentage:5.2f}%)")
    
    # Analyze decode latency by output tokens
    log.info("\n" + "=" * 80)
    log.info("Decode Per-Token Latency Analysis by Output Tokens")
    log.info("=" * 80)
    
    # Group by output tokens ranges
    decode_latency_by_range = defaultdict(list)
    decode_per_token_by_range = defaultdict(list)
    
    for sample in samples:
        output_tokens = sample.get('output_tokens', 0)
        T_LLM_decode = sample.get('T_LLM_decode', 0.0)
        T_decode_per_token = sample.get('T_decode_per_token', 0.0)
        
        if output_tokens == 0:
            continue
        
        # Determine range
        if output_tokens == 1:
            range_key = "1 token"
        elif output_tokens == 2:
            range_key = "2 tokens"
        elif output_tokens <= 5:
            range_key = "3-5 tokens"
        elif output_tokens <= 10:
            range_key = "6-10 tokens"
        elif output_tokens <= 20:
            range_key = "11-20 tokens"
        else:
            range_key = "21+ tokens"
        
        decode_latency_by_range[range_key].append(T_LLM_decode)
        decode_per_token_by_range[range_key].append(T_decode_per_token)
    
    log.info("\nDecode Per-Token Latency Statistics by Output Tokens Range:")
    for range_key in sorted(decode_per_token_by_range.keys(), key=lambda x: (
        1 if '1 token' in x else 2 if '2 tokens' in x else 3 if '3-5' in x else 4 if '6-10' in x else 5 if '11-20' in x else 6
    )):
        values = decode_per_token_by_range[range_key]
        if len(values) > 0:
            values_array = np.array(values)
            # Filter out outliers (> 60ms/token)
            filtered_values = values_array[values_array <= 60.0]
            log.info(f"\n  {range_key}:")
            log.info(f"    Samples: {len(values)} (after filtering >60ms: {len(filtered_values)})")
            if len(filtered_values) > 0:
                log.info(f"    Mean: {filtered_values.mean():.2f} ms/token")
                log.info(f"    Median: {np.median(filtered_values):.2f} ms/token")
                log.info(f"    Std: {filtered_values.std():.2f} ms/token")
                log.info(f"    Min: {filtered_values.min():.2f} ms/token, Max: {filtered_values.max():.2f} ms/token")
                log.info(f"    P25: {np.percentile(filtered_values, 25):.2f} ms/token, P75: {np.percentile(filtered_values, 75):.2f} ms/token")
                if len(values) != len(filtered_values):
                    log.info(f"    Outliers (>60ms/token): {len(values) - len(filtered_values)} samples")
    
    # Dataset-wise analysis
    log.info("\n" + "=" * 80)
    log.info("Dataset-wise Output Tokens Distribution")
    log.info("=" * 80)
    
    dataset_stats = defaultdict(lambda: {'samples': [], 'output_tokens': []})
    for sample in samples:
        dataset = sample.get('dataset', 'unknown')
        output_tokens = sample.get('output_tokens', 0)
        dataset_stats[dataset]['samples'].append(sample)
        dataset_stats[dataset]['output_tokens'].append(output_tokens)
    
    for dataset in sorted(dataset_stats.keys()):
        tokens_array = np.array(dataset_stats[dataset]['output_tokens'])
        short_answers = np.sum(tokens_array <= 2)
        short_percentage = (short_answers / len(tokens_array)) * 100
        
        log.info(f"\n  {dataset}:")
        log.info(f"    Total samples: {len(tokens_array)}")
        log.info(f"    Output tokens - Mean: {tokens_array.mean():.2f}, Median: {np.median(tokens_array):.2f}")
        log.info(f"    Short answers (<=2 tokens): {short_answers} ({short_percentage:.2f}%)")
        log.info(f"    Very short (1 token): {np.sum(tokens_array == 1)} ({np.sum(tokens_array == 1)/len(tokens_array)*100:.2f}%)")
    
    # Create visualizations
    log.info("\nCreating visualizations...")
    
    # 1. Output tokens histogram
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    # Histogram of all output tokens
    ax = axes[0, 0]
    ax.hist(output_tokens_array, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
    ax.set_xlabel('Output Tokens', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Output Tokens (All Datasets)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('white')
    ax.axvline(np.median(output_tokens_array), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(output_tokens_array):.1f}')
    ax.legend()
    
    # Histogram of short answers (1-10 tokens)
    ax = axes[0, 1]
    short_tokens = output_tokens_array[output_tokens_array <= 10]
    ax.hist(short_tokens, bins=10, edgecolor='black', alpha=0.7, color='#e74c3c')
    ax.set_xlabel('Output Tokens', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Short Answers (≤10 tokens)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('white')
    
    # Decode per-token latency vs output tokens (scatter)
    ax = axes[1, 0]
    decode_per_token_list = [s.get('T_decode_per_token', 0.0) for s in samples]
    decode_per_token_array = np.array(decode_per_token_list)
    
    # Filter outliers
    valid_mask = (decode_per_token_array <= 60.0) & (output_tokens_array > 0)
    valid_output_tokens = output_tokens_array[valid_mask]
    valid_decode_per_token = decode_per_token_array[valid_mask]
    
    ax.scatter(valid_output_tokens, valid_decode_per_token, alpha=0.3, s=10, color='#2ecc71')
    ax.set_xlabel('Output Tokens', fontsize=12, fontweight='bold')
    ax.set_ylabel('Decode Per-Token Latency (ms/token)', fontsize=12, fontweight='bold')
    ax.set_title('Decode Per-Token Latency vs Output Tokens', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('white')
    
    # Box plot: decode per-token latency by output tokens range
    ax = axes[1, 1]
    range_labels = []
    range_data = []
    for range_key in sorted(decode_per_token_by_range.keys(), key=lambda x: (
        1 if '1 token' in x else 2 if '2 tokens' in x else 3 if '3-5' in x else 4 if '6-10' in x else 5 if '11-20' in x else 6
    )):
        values = np.array(decode_per_token_by_range[range_key])
        filtered_values = values[values <= 60.0]
        if len(filtered_values) > 0:
            range_labels.append(range_key)
            range_data.append(filtered_values)
    
    if range_data:
        bp = ax.boxplot(range_data, labels=range_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#3498db')
            patch.set_alpha(0.7)
        ax.set_ylabel('Decode Per-Token Latency (ms/token)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Output Tokens Range', fontsize=12, fontweight='bold')
        ax.set_title('Decode Per-Token Latency Distribution by Output Tokens Range', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('white')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = output_dir / 'output_tokens_distribution.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none', pad_inches=0.2,
               format='png', transparent=False,
               pil_kwargs={'mode': 'RGB'})
    plt.close(fig)
    log.info(f"Saved visualization to {output_path}")
    
    # Save statistics to JSON
    stats = {
        'total_samples': len(samples),
        'output_tokens_stats': {
            'min': float(output_tokens_array.min()),
            'max': float(output_tokens_array.max()),
            'mean': float(output_tokens_array.mean()),
            'median': float(np.median(output_tokens_array)),
            'std': float(output_tokens_array.std()),
            'p25': float(np.percentile(output_tokens_array, 25)),
            'p75': float(np.percentile(output_tokens_array, 75)),
            'p95': float(np.percentile(output_tokens_array, 95)),
            'p99': float(np.percentile(output_tokens_array, 99)),
        },
        'range_counts': {label: {'count': int(count), 'percentage': float(percentage)} 
                        for label, (count, percentage) in range_counts.items()},
        'decode_per_token_by_range': {
            range_key: {
                'count': len(values),
                'mean': float(np.mean(filtered_values)) if len(filtered_values) > 0 else 0.0,
                'median': float(np.median(filtered_values)) if len(filtered_values) > 0 else 0.0,
                'std': float(np.std(filtered_values)) if len(filtered_values) > 0 else 0.0,
            }
            for range_key, values in decode_per_token_by_range.items()
            if len(values) > 0
            for filtered_values in [np.array(values)[np.array(values) <= 60.0]]
        },
        'dataset_stats': {
            dataset: {
                'total_samples': len(stats['samples']),
                'mean_output_tokens': float(np.mean(np.array(stats['output_tokens']))),
                'median_output_tokens': float(np.median(np.array(stats['output_tokens']))),
                'short_answers_count': int(np.sum(np.array(stats['output_tokens']) <= 2)),
                'short_answers_percentage': float(np.sum(np.array(stats['output_tokens']) <= 2) / len(stats['output_tokens']) * 100),
                'one_token_count': int(np.sum(np.array(stats['output_tokens']) == 1)),
                'one_token_percentage': float(np.sum(np.array(stats['output_tokens']) == 1) / len(stats['output_tokens']) * 100),
            }
            for dataset, stats in dataset_stats.items()
        }
    }
    
    stats_path = output_dir / 'output_tokens_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    log.info(f"Saved statistics to {stats_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze output tokens distribution across datasets")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing core experiment results")
    parser.add_argument("--dataset_names", type=str, nargs="+", default=None,
                       help="List of dataset names (if not provided, auto-detect all)")
    parser.add_argument("--output_dir", type=str, default="./results/analysis_output",
                       help="Output directory for statistics and visualizations")
    parser.add_argument("--use_all_datasets", action="store_true",
                       help="Use all datasets in results_dir (overrides dataset_names)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        log.error(f"Results directory not found: {results_dir}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load samples
    if args.use_all_datasets or not args.dataset_names:
        dataset_names = None  # Auto-detect
    else:
        dataset_names = args.dataset_names
    
    samples = load_all_samples(results_dir, dataset_names)
    
    if len(samples) == 0:
        log.error("No samples loaded!")
        return
    
    # Analyze
    stats = analyze_output_tokens_distribution(samples, output_dir)
    
    log.info("\n" + "=" * 80)
    log.info("Analysis complete!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

