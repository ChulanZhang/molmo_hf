#!/usr/bin/env python3
"""
Compare importance scores from train set vs validation set.
This script runs sensitivity analysis on both splits and computes Spearman correlation.

Usage:
    python experiments/profiling/knob3_layers/compare_train_val_importance.py \
        --dataset_name coco_2014_vqa \
        --model_path checkpoints \
        --output_dir ./results/profiling/exp3_importance_comparison
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def run_sensitivity_analysis(
    dataset_name: str,
    split: str,
    model_path: str,
    output_dir: str,
    num_gpus: int,
    batch_size: int = 16,  # Lower default batch size to avoid OOM
    num_samples: int = None,
    max_new_tokens: int = 16,
) -> Dict[int, float]:
    """Run sensitivity analysis and return importance scores."""
    
    log.info(f"Running sensitivity analysis on {dataset_name} ({split} split)...")
    
    # Create split-specific output directory
    split_output_dir = os.path.join(output_dir, dataset_name, split)
    os.makedirs(split_output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "torchrun",
        f"--nproc-per-node={num_gpus}",
        "experiments/profiling/knob3_layers/exp3_accuracy_sensitivity_v2.py",
        "--model_path", model_path,
        "--output_dir", split_output_dir,
        "--dataset_name", dataset_name,
        "--split", split,
        "--batch_size", str(batch_size),
        "--max_new_tokens", str(max_new_tokens),
        # Note: --skip_sensitivity is a flag (action="store_true"), 
        # so we don't pass it to run sensitivity analysis
        "--beam_width", "3",
        "--max_blocks_to_remove", "0",  # Skip beam search, only do sensitivity analysis
        "--auto_adjust_batch_size",  # Enable batch size optimization
    ]
    
    if num_samples is not None:
        cmd.extend(["--num_samples", str(num_samples)])
    
    # Run command with real-time output streaming (like run_multi_datasets_h100.py)
    # This allows tqdm progress bars to display properly
    try:
        # Create environment with warnings suppressed
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
        env['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'  # Suppress torchrun warnings
        
        # Use Popen to stream output in real-time
        # Let stderr go directly to terminal (for tqdm progress bars)
        # Capture stdout for error analysis if needed
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,  # Capture stdout for error analysis
            stderr=None,  # Let stderr go directly to terminal (for tqdm)
            text=True,
            bufsize=1,  # Line buffered
            env=env
        )
        
        # Stream stdout to terminal in real-time, filtering torchrun warnings
        stdout_lines = []
        for line in process.stdout:
            # Filter out torchrun OMP_NUM_THREADS warnings
            if 'OMP_NUM_THREADS' in line or 'Setting OMP_NUM_THREADS' in line or '*****************************************' in line:
                continue
            # Write to terminal in real-time
            sys.stdout.write(line)
            sys.stdout.flush()
            # Also save for error analysis
            stdout_lines.append(line)
        
        # Wait for process to complete
        process.wait()
        
        # Check if successful
        if process.returncode == 0:
            log.info(f"Sensitivity analysis completed for {dataset_name} ({split})")
        else:
            # Process failed, analyze error output
            error_output = ''.join(stdout_lines)
            log.error(f"Failed to run sensitivity analysis for {dataset_name} ({split}): Command returned non-zero exit status {process.returncode}")
            if error_output:
                # Extract key error messages
                error_lines = error_output.split('\n')
                key_errors = [line for line in error_lines if any(keyword in line for keyword in [
                    'FileNotFoundError', 'CUDA error', 'invalid configuration', 'OOM', 'out of memory',
                    'RuntimeError', 'Traceback', 'Error', 'UnboundLocalError', 'device-side assert'
                ])]
                if key_errors:
                    log.error(f"Key errors (showing first 20 lines):")
                    for err in key_errors[:20]:  # Show first 20 key errors
                        log.error(f"  {err}")
                
                # Always show last 50 lines for debugging
                log.error(f"\nFull error output (last 50 lines):")
                all_lines = error_output.split('\n')
                for line in all_lines[-50:]:
                    if line.strip():
                        log.error(f"  {line}")
            
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
    except subprocess.CalledProcessError as e:
        raise
    except Exception as e:
        log.error(f"Unexpected error running sensitivity analysis: {e}")
        raise
    
    # Load importance scores
    importance_file = os.path.join(split_output_dir, "layer_importance_scores.json")
    if not os.path.exists(importance_file):
        raise FileNotFoundError(f"Importance scores file not found: {importance_file}")
    
    with open(importance_file, 'r') as f:
        importance_scores = json.load(f)
    
    # Convert keys to int
    importance_scores = {int(k): float(v) for k, v in importance_scores.items()}
    
    return importance_scores


def compute_spearman_correlation(
    scores_train: Dict[int, float],
    scores_val: Dict[int, float]
) -> Tuple[float, float]:
    """
    Compute Spearman correlation between train and validation importance scores.
    
    Returns:
        (correlation, p-value)
    """
    # Get common block indices
    common_blocks = sorted(set(scores_train.keys()) & set(scores_val.keys()))
    
    if len(common_blocks) < 2:
        raise ValueError("Need at least 2 common blocks to compute correlation")
    
    # Extract scores in the same order
    train_scores = [scores_train[b] for b in common_blocks]
    val_scores = [scores_val[b] for b in common_blocks]
    
    # Compute Spearman correlation
    correlation, p_value = spearmanr(train_scores, val_scores)
    
    return float(correlation), float(p_value)


def visualize_importance_comparison(
    scores_train: Dict[int, float],
    scores_val: Dict[int, float],
    dataset_name: str,
    output_dir: str,
    correlation: float = None,
    p_value: float = None,
):
    """
    Create a bar chart comparing importance scores between train and validation sets.
    
    Args:
        scores_train: Importance scores from train set
        scores_val: Importance scores from validation set
        dataset_name: Name of the dataset
        output_dir: Output directory for saving the plot
        correlation: Spearman correlation coefficient (optional)
        p_value: P-value of correlation (optional)
    """
    # Get all blocks and sort by index
    all_blocks = sorted(set(scores_train.keys()) | set(scores_val.keys()))
    
    # Extract scores in block order
    train_scores = [scores_train.get(block, 0.0) for block in all_blocks]
    val_scores = [scores_val.get(block, 0.0) for block in all_blocks]
    
    # Create figure with two subplots: side-by-side bars and grouped bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Side-by-side comparison
    x = np.arange(len(all_blocks))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_scores, width, label='Train', alpha=0.8, color='#2E86AB')
    bars2 = ax1.bar(x + width/2, val_scores, width, label='Validation', alpha=0.8, color='#A23B72')
    
    ax1.set_xlabel('Block Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Importance Score (Accuracy Drop)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Block Importance Comparison: {dataset_name}\n(Side-by-Side)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Block {b}' for b in all_blocks], rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # Add correlation info if provided
    if correlation is not None and p_value is not None:
        ax1.text(0.02, 0.98, f'Spearman ρ = {correlation:.4f}\np-value = {p_value:.4f}',
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Grouped bars (stacked view for easier comparison)
    x2 = np.arange(len(all_blocks))
    bars3 = ax2.bar(x2, train_scores, width=0.6, label='Train', alpha=0.7, color='#2E86AB', edgecolor='black', linewidth=0.5)
    bars4 = ax2.bar(x2, val_scores, width=0.6, label='Validation', alpha=0.7, color='#A23B72', 
                    bottom=train_scores, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Block Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Importance Score (Accuracy Drop)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Block Importance Comparison: {dataset_name}\n(Stacked)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f'Block {b}' for b in all_blocks], rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)
    
    # Highlight Block 0 if it has very high importance (outlier)
    max_score = max(max(train_scores), max(val_scores))
    if max_score > 0.5:  # Block 0 is an outlier
        ax1.axhline(y=max_score * 0.8, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(y=max_score * 0.8, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(len(all_blocks) - 1, max_score * 0.8, 'Block 0 outlier', 
                ha='right', va='bottom', fontsize=9, color='red', style='italic')
        ax2.text(len(all_blocks) - 1, max_score * 0.8, 'Block 0 outlier', 
                ha='right', va='bottom', fontsize=9, color='red', style='italic')
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, dataset_name, f"importance_comparison_{dataset_name}.png")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    log.info(f"Visualization saved to: {output_file}")
    
    plt.close()
    
    # Also create a simpler single plot version (just side-by-side)
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(len(all_blocks))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_scores, width, label='Train', alpha=0.8, color='#2E86AB', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, val_scores, width, label='Validation', alpha=0.8, color='#A23B72', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Block Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Importance Score (Accuracy Drop)', fontsize=12, fontweight='bold')
    title = f'Block Importance Comparison: {dataset_name}'
    if correlation is not None:
        title += f'\nSpearman ρ = {correlation:.4f} (p = {p_value:.4f})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Block {b}' for b in all_blocks], rotation=45, ha='right')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add value labels on bars (only for smaller values to avoid clutter)
    for i, (train_val, val_val) in enumerate(zip(train_scores, val_scores)):
        if train_val < 0.1:  # Only label small values
            ax.text(i - width/2, train_val + 0.002, f'{train_val:.3f}', 
                   ha='center', va='bottom', fontsize=8, rotation=90)
        if val_val < 0.1:
            ax.text(i + width/2, val_val + 0.002, f'{val_val:.3f}', 
                   ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    
    # Save simpler version
    output_file_simple = os.path.join(output_dir, dataset_name, f"importance_comparison_{dataset_name}_simple.png")
    plt.savefig(output_file_simple, dpi=300, bbox_inches='tight')
    log.info(f"Simple visualization saved to: {output_file_simple}")
    
    plt.close()


def compare_importance_scores(
    dataset_name: str,
    model_path: str,
    output_dir: str,
    num_gpus: int = 4,
    batch_size: int = 64,
    num_samples: int = None,
    max_new_tokens: int = 16,
) -> Dict:
    """Compare importance scores from train and validation sets."""
    
    log.info("=" * 80)
    log.info(f"Comparing importance scores: {dataset_name}")
    log.info("=" * 80)
    
    # Run sensitivity analysis on train set
    scores_train = None
    try:
        scores_train = run_sensitivity_analysis(
            dataset_name=dataset_name,
            split="train",
            model_path=model_path,
            output_dir=output_dir,
            num_gpus=num_gpus,
            batch_size=batch_size,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
        )
        log.info(f"✅ Successfully computed importance scores on train set")
    except Exception as e:
        error_msg = str(e)
        # Check if it's a data availability issue (missing train images)
        if "FileNotFoundError" in error_msg or "train2014" in error_msg or "No such file" in error_msg:
            log.warning(f"⚠️  Train set images not available for {dataset_name}")
            log.warning(f"   Error: {error_msg[:200]}...")
            log.warning(f"   This is likely because train2014 images are not downloaded.")
            log.warning(f"   Will use validation set instead.")
        else:
            log.error(f"Failed to run sensitivity analysis on train set: {e}")
            raise
    
    # Run sensitivity analysis on validation set
    try:
        scores_val = run_sensitivity_analysis(
            dataset_name=dataset_name,
            split="validation",
            model_path=model_path,
            output_dir=output_dir,
            num_gpus=num_gpus,
            batch_size=batch_size,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
        )
    except Exception as e:
        log.error(f"Failed to run sensitivity analysis on validation set: {e}")
        scores_val = None
    
    if scores_train is None and scores_val is None:
        raise ValueError("Could not get importance scores from either split")
    
    # If train set is not available, we can only use validation set
    if scores_train is None:
        log.warning("=" * 80)
        log.warning("⚠️  Train set not available - cannot compare train vs validation")
        log.warning("=" * 80)
        log.warning(f"Train set images are missing for {dataset_name}.")
        log.warning(f"This is likely because train2014 images are not downloaded.")
        log.warning(f"")
        log.warning(f"Options:")
        log.warning(f"1. Download train2014 images and re-run")
        log.warning(f"2. Use validation set for sensitivity analysis (acceptable)")
        log.warning(f"")
        log.warning(f"Since train set is not available, we'll use validation set scores only.")
        log.warning(f"Note: This means we cannot verify consistency between splits.")
        log.warning("=" * 80)
        
        # Return validation set scores only
        comparison_result = {
            "dataset_name": dataset_name,
            "train_available": False,
            "validation_available": True,
            "note": "Train set images not available, using validation set only",
            "validation_scores": {str(k): v for k, v in scores_val.items()},
            "block_ranking_val": sorted(scores_val.items(), key=lambda x: x[1]),
        }
        
        # Save results
        output_file = os.path.join(output_dir, dataset_name, f"importance_comparison_{dataset_name}.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(comparison_result, f, indent=2)
        
        log.info(f"\nResults saved to: {output_file}")
        log.info(f"\n⚠️  Cannot compare train vs validation - train set not available")
        log.info(f"   Using validation set scores only.")
        
        return comparison_result
    
    if scores_val is None:
        raise ValueError("Could not get importance scores from validation set")
    
    # Compute correlation
    correlation, p_value = compute_spearman_correlation(scores_train, scores_val)
    
    # Prepare results
    common_blocks = sorted(set(scores_train.keys()) & set(scores_val.keys()))
    
    comparison_result = {
        "dataset_name": dataset_name,
        "num_common_blocks": len(common_blocks),
        "spearman_correlation": correlation,
        "p_value": p_value,
        "is_consistent": correlation > 0.9,  # Threshold: 0.9
        "train_scores": {str(k): v for k, v in scores_train.items()},
        "validation_scores": {str(k): v for k, v in scores_val.items()},
        "block_ranking_train": sorted(scores_train.items(), key=lambda x: x[1]),  # Least important first
        "block_ranking_val": sorted(scores_val.items(), key=lambda x: x[1]),  # Least important first
    }
    
    # Log results
    log.info("=" * 80)
    log.info("Comparison Results:")
    log.info("=" * 80)
    log.info(f"Dataset: {dataset_name}")
    log.info(f"Number of common blocks: {len(common_blocks)}")
    log.info(f"Spearman correlation: {correlation:.4f}")
    log.info(f"P-value: {p_value:.4f}")
    log.info(f"Consistency (correlation > 0.9): {'✅ YES' if correlation > 0.9 else '❌ NO'}")
    
    log.info("\nBlock Importance Ranking (Train):")
    for i, (block_idx, score) in enumerate(comparison_result["block_ranking_train"], 1):
        log.info(f"  {i}. Block {block_idx}: {score:.4f}")
    
    log.info("\nBlock Importance Ranking (Validation):")
    for i, (block_idx, score) in enumerate(comparison_result["block_ranking_val"], 1):
        log.info(f"  {i}. Block {block_idx}: {score:.4f}")
    
    # Save results
    output_file = os.path.join(output_dir, dataset_name, f"importance_comparison_{dataset_name}.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(comparison_result, f, indent=2)
    
    log.info(f"\nResults saved to: {output_file}")
    
    # Create visualization
    try:
        visualize_importance_comparison(
            scores_train=scores_train,
            scores_val=scores_val,
            dataset_name=dataset_name,
            output_dir=output_dir,
            correlation=correlation,
            p_value=p_value,
        )
    except Exception as e:
        log.warning(f"Failed to create visualization: {e}")
        log.warning("Continuing without visualization...")
    
    return comparison_result


# Supported datasets for importance comparison
# These are datasets that typically have both train and validation splits
SUPPORTED_DATASETS = [
    "coco_2014_vqa",
    "text_vqa",
    "okvqa",
    "science_qa_img",
    "doc_qa",
    "chart_qa",
    "info_qa",
    "plot_qa",
    "figure_qa",
    "dv_qa",
    "mmmu",
    "coco_caption",
    "coco_captioning",  # Alias for coco_caption
    "tally_qa",
    "st_qa",  # SceneTextQa
]


def main():
    parser = argparse.ArgumentParser(
        description="Compare importance scores from train vs validation sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    # Single dataset:
    python experiments/profiling/knob3_layers/compare_train_val_importance.py \\
        --dataset_name coco_2014_vqa
    
    # Multiple datasets:
    python experiments/profiling/knob3_layers/compare_train_val_importance.py \\
        --dataset_name coco_2014_vqa text_vqa okvqa
    
    # All supported datasets:
    python experiments/profiling/knob3_layers/compare_train_val_importance.py \\
        --dataset_name all

Supported datasets: {', '.join(SUPPORTED_DATASETS)}
        """
    )
    parser.add_argument("--dataset_name", type=str, nargs="+", required=True,
                       help="Dataset name(s) (e.g., coco_2014_vqa) or 'all' for all supported datasets. "
                            "Can specify multiple: --dataset_name coco_2014_vqa text_vqa")
    parser.add_argument("--model_path", type=str, default="checkpoints",
                       help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str,
                       default="./results/profiling/exp3_importance_comparison",
                       help="Output directory")
    parser.add_argument("--num_gpus", type=int, default=4,
                       help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size (default: 16, lower if OOM)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to use (None = use all)")
    parser.add_argument("--max_new_tokens", type=int, default=16,
                       help="Maximum new tokens")
    
    args = parser.parse_args()
    
    # Auto-detect number of GPUs if not specified
    if args.num_gpus == 4:
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'],
                                   capture_output=True, text=True, check=True)
            num_gpus = len(result.stdout.strip().split('\n'))
            args.num_gpus = num_gpus
            log.info(f"Auto-detected {num_gpus} GPUs")
        except Exception:
            pass
    
    # Parse dataset names
    dataset_names = args.dataset_name
    if len(dataset_names) == 1 and dataset_names[0].lower() == "all":
        dataset_names = SUPPORTED_DATASETS
        log.info(f"Running comparison on all {len(dataset_names)} supported datasets")
    else:
        # Validate dataset names
        invalid_datasets = [d for d in dataset_names if d not in SUPPORTED_DATASETS]
        if invalid_datasets:
            log.warning(f"Warning: The following datasets may not be fully supported: {invalid_datasets}")
            log.warning(f"Supported datasets: {', '.join(SUPPORTED_DATASETS)}")
            log.warning("Continuing anyway (they may still work if they have train/validation splits)...")
    
    # Run comparison for each dataset
    all_results = {}
    failed_datasets = []
    
    log.info("=" * 80)
    log.info(f"Running importance comparison on {len(dataset_names)} dataset(s)")
    log.info("=" * 80)
    
    for i, dataset_name in enumerate(dataset_names, 1):
        log.info("")
        log.info("=" * 80)
        log.info(f"Dataset {i}/{len(dataset_names)}: {dataset_name}")
        log.info("=" * 80)
        
        try:
            result = compare_importance_scores(
                dataset_name=dataset_name,
                model_path=args.model_path,
                output_dir=args.output_dir,
                num_gpus=args.num_gpus,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
            )
            
            all_results[dataset_name] = result
            
            # Print summary for this dataset
            log.info("")
            log.info("=" * 80)
            log.info(f"SUMMARY - {dataset_name}")
            log.info("=" * 80)
            if 'spearman_correlation' in result:
                log.info(f"Spearman Correlation: {result['spearman_correlation']:.4f}")
                log.info(f"P-value: {result['p_value']:.4f}")
                log.info(f"Consistency: {'✅ YES' if result['is_consistent'] else '❌ NO'} (correlation {'>' if result['is_consistent'] else '<='} 0.9)")
            else:
                log.info("⚠️  Could not compare train vs validation (train set not available)")
            log.info("=" * 80)
            
        except Exception as e:
            log.error(f"Failed to compare importance scores for {dataset_name}: {e}", exc_info=True)
            failed_datasets.append(dataset_name)
            all_results[dataset_name] = {"error": str(e)}
            # Continue with next dataset
            log.warning(f"Continuing to next dataset...")
    
    # Print final summary
    log.info("")
    log.info("=" * 80)
    log.info("FINAL SUMMARY")
    log.info("=" * 80)
    log.info(f"Total datasets processed: {len(dataset_names)}")
    log.info(f"Successful: {len(dataset_names) - len(failed_datasets)}")
    log.info(f"Failed: {len(failed_datasets)}")
    
    if failed_datasets:
        log.warning(f"Failed datasets: {', '.join(failed_datasets)}")
    
    # Summary table of correlations
    if len(all_results) > 1:
        log.info("")
        log.info("Correlation Summary:")
        log.info("-" * 80)
        log.info(f"{'Dataset':<25} {'Correlation':<15} {'P-value':<15} {'Consistent':<12}")
        log.info("-" * 80)
        for dataset_name, result in all_results.items():
            if 'spearman_correlation' in result:
                corr = result['spearman_correlation']
                pval = result['p_value']
                consistent = '✅ YES' if result['is_consistent'] else '❌ NO'
                log.info(f"{dataset_name:<25} {corr:<15.4f} {pval:<15.4f} {consistent:<12}")
            elif 'error' in result:
                log.info(f"{dataset_name:<25} {'ERROR':<15} {'-':<15} {'-':<12}")
            else:
                log.info(f"{dataset_name:<25} {'N/A':<15} {'-':<15} {'-':<12}")
        log.info("-" * 80)
    
    log.info("=" * 80)
    
    # Exit with error if any datasets failed
    if failed_datasets:
        sys.exit(1)


if __name__ == "__main__":
    main()

