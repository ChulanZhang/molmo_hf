"""
Visualize Latency Estimator performance.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set matplotlib style for better visibility
plt.style.use('default')
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor'] = 'white'
matplotlib.rcParams['savefig.facecolor'] = 'white'
matplotlib.rcParams['savefig.edgecolor'] = 'none'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.controller.core_exp_data_loader import CoreExpDataLoader
from experiments.controller.latency_estimator import LatencyEstimator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


class LatencyEstimatorDataset:
    """Dataset for latency estimator visualization."""
    
    def __init__(self, samples: List[Dict]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        tier = sample.get('tier', 'medium')
        tier_map = {'low': 0, 'medium': 1, 'high': 2}
        tier_idx = tier_map.get(tier.lower(), 1)
        
        T_vision_total = sample.get('T_vision_total', 0.0)
        T_LLM_prefill = sample.get('T_LLM_prefill', 0.0)
        T_prefill_total = T_vision_total + T_LLM_prefill
        
        T_LLM_decode = sample.get('T_LLM_decode', 0.0)
        output_tokens = sample.get('output_tokens', 1)
        T_decode_per_token_avg = T_LLM_decode / max(output_tokens, 1)
        
        return {
            'vision_tokens': torch.tensor(sample.get('vision_tokens', 0), dtype=torch.long),
            'text_tokens': torch.tensor(sample.get('text_tokens', 0), dtype=torch.long),
            'tier_idx': torch.tensor(tier_idx, dtype=torch.long),
            'top_k': torch.tensor(sample.get('top_k', 8), dtype=torch.long),
            'num_active_blocks': torch.tensor(sample.get('num_active_blocks', 16), dtype=torch.long),
            'output_tokens': torch.tensor(output_tokens, dtype=torch.long),  # For positioned prediction
            'T_prefill_total': torch.tensor(T_prefill_total, dtype=torch.float32),
            'T_LLM_decode': torch.tensor(T_LLM_decode, dtype=torch.float32),  # Total decode latency
            'T_decode_per_token_avg': torch.tensor(T_decode_per_token_avg, dtype=torch.float32),  # Average (for reference)
            'tier': tier,
            'top_k_value': sample.get('top_k', 8),
            'num_blocks_value': sample.get('num_active_blocks', 16),
        }


def collect_predictions(
    model: LatencyEstimator,
    data_loader: DataLoader,
    device: str = "cuda",
    max_samples: int = 10000,
) -> Dict:
    """Collect predictions and targets."""
    model.eval()
    
    all_pred_prefill = []
    all_target_prefill = []
    all_pred_decode = []
    all_target_decode = []
    all_configs = []
    
    sample_count = 0
    total_batches = len(data_loader)
    
    log.info(f"Starting prediction collection: {total_batches} batches, max_samples={max_samples}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Collecting predictions", total=total_batches)):
            if sample_count >= max_samples:
                log.info(f"Reached max_samples={max_samples}, stopping")
                break
            
            if (batch_idx + 1) % 100 == 0:
                log.info(f"  Processed {batch_idx+1}/{total_batches} batches, {sample_count} samples collected")
                
            batch_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Predict prefill (position doesn't matter)
            pred_prefill_tensor = model(
                vision_tokens=batch_device['vision_tokens'],
                text_tokens=batch_device['text_tokens'],
                tier_idx=batch_device['tier_idx'],
                top_k=batch_device['top_k'],
                num_active_blocks=batch_device['num_active_blocks'],
                token_position=torch.ones_like(batch_device['vision_tokens']),
            )['T_prefill_total']
            pred_prefill = pred_prefill_tensor.cpu().numpy()
            target_prefill = batch_device['T_prefill_total'].cpu().numpy()
            
            # Predict decode total by summing positioned latencies
            B = batch_device['vision_tokens'].shape[0]
            output_tokens = batch_device['output_tokens']
            max_tokens = int(output_tokens.max().item())
            
            # Predict at all positions
            positions = torch.arange(1, max_tokens + 1, device=device).float()
            decode_latencies_all = model.predict_decode_at_positions(
                vision_tokens=batch_device['vision_tokens'],
                text_tokens=batch_device['text_tokens'],
                tier_idx=batch_device['tier_idx'],
                top_k=batch_device['top_k'],
                num_active_blocks=batch_device['num_active_blocks'],
                positions=positions,
            ) # (B, max_tokens)
            
            # Sum up to output_tokens for each sample
            pred_decode_total = torch.zeros(B, device=device, dtype=torch.float32)
            for i in range(B):
                num_tokens = int(output_tokens[i].item())
                if num_tokens > 0 and num_tokens <= max_tokens:
                    pred_decode_total[i] = decode_latencies_all[i, :num_tokens].sum()
            
            # Also compute average per-token latency for comparison
            pred_decode_avg = (pred_decode_total / output_tokens.float()).cpu().numpy()
            target_decode_total = batch_device['T_LLM_decode'].cpu().numpy()
            target_decode_avg = batch_device['T_decode_per_token_avg'].cpu().numpy()
            
            all_pred_prefill.extend(pred_prefill)
            all_target_prefill.extend(target_prefill)
            all_pred_decode.extend(pred_decode_avg)  # Use average for visualization compatibility
            all_target_decode.extend(target_decode_avg)  # Use average for visualization compatibility
            
            # Store config info
            for i in range(len(pred_prefill)):
                # Convert to Python native types
                top_k_val = batch['top_k_value'][i]
                num_blocks_val = batch['num_blocks_value'][i]
                tier_val = batch['tier'][i]
                
                # Handle tensor conversion
                if hasattr(top_k_val, 'item'):
                    top_k_val = int(top_k_val.item())
                elif isinstance(top_k_val, torch.Tensor):
                    top_k_val = int(top_k_val.cpu().item())
                else:
                    top_k_val = int(top_k_val)
                
                if hasattr(num_blocks_val, 'item'):
                    num_blocks_val = int(num_blocks_val.item())
                elif isinstance(num_blocks_val, torch.Tensor):
                    num_blocks_val = int(num_blocks_val.cpu().item())
                else:
                    num_blocks_val = int(num_blocks_val)
                
                if isinstance(tier_val, torch.Tensor):
                    tier_val = tier_val.cpu().item() if tier_val.numel() == 1 else str(tier_val.cpu().numpy())
                elif not isinstance(tier_val, str):
                    tier_val = str(tier_val)
                
                all_configs.append({
                    'tier': tier_val,
                    'top_k': top_k_val,
                    'num_blocks': num_blocks_val,
                })
            
            sample_count += len(pred_prefill)
    
    log.info(f"Collected predictions for {sample_count} samples")
    log.info(f"  Prefill: {len(all_pred_prefill)} predictions")
    log.info(f"  Decode: {len(all_pred_decode)} predictions")
    
    return {
        'prefill': {
            'pred': np.array(all_pred_prefill),
            'target': np.array(all_target_prefill),
        },
        'decode': {
            'pred': np.array(all_pred_decode),
            'target': np.array(all_target_decode),
        },
        'configs': all_configs,
    }


def plot_scatter(pred: np.ndarray, target: np.ndarray, title: str, xlabel: str, ylabel: str, ax=None):
    """Plot scatter plot of predictions vs targets."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot with better visibility
    ax.scatter(target, pred, alpha=0.5, s=15, color='#3498db', edgecolors='black', linewidths=0.5)
    
    # Perfect prediction line (y=x)
    min_val = min(target.min(), pred.min())
    max_val = max(target.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    
    # Calculate metrics
    mae = np.mean(np.abs(pred - target))
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    r2 = 1 - np.sum((target - pred) ** 2) / np.sum((target - np.mean(target)) ** 2)
    
    # Add metrics to plot
    ax.text(0.05, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_error_distribution(pred: np.ndarray, target: np.ndarray, title: str, ax=None):
    """Plot error distribution."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
    
    errors = pred - target
    abs_errors = np.abs(errors)
    
    ax.hist(abs_errors, bins=50, alpha=0.8, edgecolor='black', linewidth=1.5, color='#3498db')
    ax.axvline(np.mean(abs_errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(abs_errors):.2f}')
    ax.axvline(np.median(abs_errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(abs_errors):.2f}')
    
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_by_config(data: Dict, output_dir: Path):
    """Plot errors grouped by configuration."""
    configs = data['configs']
    log.info(f"Total configs: {len(configs)}")
    if len(configs) > 0:
        log.info(f"Sample config: {configs[0]}")
        # Check unique values
        unique_topk = set(c.get('top_k') for c in configs)
        unique_blocks = set(c.get('num_blocks') for c in configs)
        log.info(f"Unique top_k values: {sorted(unique_topk)}")
        log.info(f"Unique num_blocks values: {sorted(unique_blocks)}")
    
    prefill_errors = np.abs(data['prefill']['pred'] - data['prefill']['target'])
    decode_errors = np.abs(data['decode']['pred'] - data['decode']['target'])
    
    log.info(f"Prefill errors range: [{prefill_errors.min():.2f}, {prefill_errors.max():.2f}]")
    log.info(f"Decode errors range: [{decode_errors.min():.2f}, {decode_errors.max():.2f}]")
    
    # Group by tier
    tier_errors = {'low': [], 'medium': [], 'high': []}
    for i, config in enumerate(configs):
        tier = config.get('tier', 'medium')
        if tier in tier_errors:
            tier_errors[tier].append({
                'prefill': prefill_errors[i],
                'decode': decode_errors[i],
            })
    
    # Filter out empty tiers
    tier_errors = {k: v for k, v in tier_errors.items() if len(v) > 0}
    
    if not tier_errors:
        log.warning("No tier data found, skipping tier plot")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('white')
        
        tiers = sorted(tier_errors.keys())
        tier_colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}  # Green, Orange, Red
        
        # Prefill errors by tier
        prefill_means = [np.mean([e['prefill'] for e in tier_errors[t]]) for t in tiers]
        prefill_stds = [np.std([e['prefill'] for e in tier_errors[t]]) for t in tiers]
        colors = [tier_colors.get(t, '#3498db') for t in tiers]
        
        bars0 = axes[0].bar(tiers, prefill_means, yerr=prefill_stds, capsize=5, 
                           alpha=0.8, edgecolor='black', linewidth=1.5, color=colors)
        axes[0].set_ylabel('MAE (ms)', fontsize=12, fontweight='bold')
        axes[0].set_title('Prefill Latency Error by Tier', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0].set_facecolor('white')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars0, prefill_means, prefill_stds)):
            axes[0].text(bar.get_x() + bar.get_width()/2, mean + std + 0.5,
                        f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Decode errors by tier
        decode_means = [np.mean([e['decode'] for e in tier_errors[t]]) for t in tiers]
        decode_stds = [np.std([e['decode'] for e in tier_errors[t]]) for t in tiers]
        
        bars1 = axes[1].bar(tiers, decode_means, yerr=decode_stds, capsize=5,
                           alpha=0.8, edgecolor='black', linewidth=1.5, color=colors)
        axes[1].set_ylabel('MAE (ms/token)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Tier', fontsize=12, fontweight='bold')
        axes[1].set_title('Decode Per-Token Latency Error by Tier', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[1].set_facecolor('white')
        axes[1].tick_params(axis='x', labelsize=11)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars1, decode_means, decode_stds)):
            axes[1].text(bar.get_x() + bar.get_width()/2, mean + std + 0.5,
                        f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        # Ensure white background is saved
        fig.patch.set_facecolor('white')
        for ax in axes:
            ax.set_facecolor('white')
        plt.savefig(output_dir / 'errors_by_tier.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        plt.close(fig)
        log.info(f"Saved errors_by_tier.png with {len(tiers)} tiers")
    
    # Group by top_k
    topk_errors = {}
    for i, config in enumerate(configs):
        topk = config.get('top_k', 8)
        if topk not in topk_errors:
            topk_errors[topk] = {'prefill': [], 'decode': []}
        topk_errors[topk]['prefill'].append(prefill_errors[i])
        topk_errors[topk]['decode'].append(decode_errors[i])
    
    if not topk_errors:
        log.warning("No top_k data found, skipping top_k plot")
    else:
        # Increase figure width for better spacing
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('white')
        
        topks = sorted(topk_errors.keys())
        log.info(f"Top-K values found: {topks}, with sample counts: {[len(topk_errors[k]['prefill']) for k in topks]}")
        
        # Use bright, distinct colors for different top_k values
        color_palette = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']  # Blue, Green, Orange, Red, Purple, Teal
        colors = [color_palette[i % len(color_palette)] for i in range(len(topks))]
        
        prefill_means = [np.mean(topk_errors[k]['prefill']) for k in topks]
        prefill_stds = [np.std(topk_errors[k]['prefill']) for k in topks]
        
        # Use wider bars with spacing
        x_pos = np.arange(len(topks))
        width = 0.6  # Bar width
        bars0 = axes[0].bar(x_pos, prefill_means, width=width, yerr=prefill_stds, capsize=5,
                           alpha=0.9, edgecolor='black', linewidth=2, color=colors)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([f'{k}' for k in topks], fontsize=12, rotation=0, ha='center')
        axes[0].tick_params(axis='x', labelsize=11, pad=8)
        axes[0].set_ylabel('MAE (ms)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Top-K', fontsize=12, fontweight='bold')
        axes[0].set_title('Prefill Latency Error by Top-K', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0].set_facecolor('white')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars0, prefill_means, prefill_stds)):
            axes[0].text(bar.get_x() + bar.get_width()/2, mean + std + 0.5,
                        f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        decode_means = [np.mean(topk_errors[k]['decode']) for k in topks]
        decode_stds = [np.std(topk_errors[k]['decode']) for k in topks]
        
        bars1 = axes[1].bar(x_pos, decode_means, width=width, yerr=decode_stds, capsize=5,
                           alpha=0.9, edgecolor='black', linewidth=2, color=colors)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([f'{k}' for k in topks], fontsize=12, rotation=0, ha='center')
        axes[1].tick_params(axis='x', labelsize=11, pad=8)
        axes[1].set_ylabel('MAE (ms/token)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Top-K', fontsize=12, fontweight='bold')
        axes[1].set_title('Decode Per-Token Latency Error by Top-K', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[1].set_facecolor('white')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars1, decode_means, decode_stds)):
            axes[1].text(bar.get_x() + bar.get_width()/2, mean + std + 0.5,
                        f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Ensure white background and proper styling
        fig.patch.set_facecolor('white')
        for ax in axes:
            ax.set_facecolor('white')
            # Style spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#333333')
            ax.spines['left'].set_color('#333333')
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            # Ensure tick colors are visible
            ax.tick_params(colors='#333333', which='both')
        
        plt.tight_layout()
        # Save with explicit parameters - force RGB mode
        output_path = output_dir / 'errors_by_topk.png'
        # Use explicit RGB conversion to avoid any transparency issues
        fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.2,
                   format='png', transparent=False, 
                   pil_kwargs={'mode': 'RGB'})
        plt.close(fig)
        log.info(f"Saved errors_by_topk.png with {len(topks)} top_k values: {topks}")
        if output_path.exists():
            log.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Group by num_blocks
    blocks_errors = {}
    for i, config in enumerate(configs):
        blocks = config.get('num_blocks', 16)
        # Convert to int if it's a tensor or other type
        if hasattr(blocks, 'item'):
            blocks = int(blocks.item())
        elif not isinstance(blocks, int):
            blocks = int(blocks)
        
        if blocks not in blocks_errors:
            blocks_errors[blocks] = {'prefill': [], 'decode': []}
        blocks_errors[blocks]['prefill'].append(float(prefill_errors[i]))
        blocks_errors[blocks]['decode'].append(float(decode_errors[i]))
    
    if not blocks_errors:
        log.warning("No num_blocks data found, skipping blocks plot")
    else:
        # Increase figure width for better spacing
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('white')
        
        blocks_list = sorted(blocks_errors.keys())
        log.info(f"Block counts found: {blocks_list}, with sample counts: {[len(blocks_errors[k]['prefill']) for k in blocks_list]}")
        
        # Verify we have data
        for k in blocks_list:
            if len(blocks_errors[k]['prefill']) == 0:
                log.warning(f"Blocks {k} has no prefill data!")
            if len(blocks_errors[k]['decode']) == 0:
                log.warning(f"Blocks {k} has no decode data!")
        
        # Use bright, distinct colors for different block counts
        color_palette = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']  # Blue, Green, Orange, Red, Purple, Teal, Dark Orange, Dark Blue
        colors = [color_palette[i % len(color_palette)] for i in range(len(blocks_list))]
        log.info(f"Using colors: {colors[:len(blocks_list)]}")
        
        prefill_means = [np.mean(blocks_errors[k]['prefill']) for k in blocks_list]
        prefill_stds = [np.std(blocks_errors[k]['prefill']) for k in blocks_list]
        log.info(f"Prefill means: {prefill_means}, stds: {prefill_stds}")
        
        # Use wider bars with spacing
        x_pos = np.arange(len(blocks_list))
        width = 0.6  # Bar width
        bars0 = axes[0].bar(x_pos, prefill_means, width=width, yerr=prefill_stds, capsize=5,
                           alpha=0.9, edgecolor='black', linewidth=2, color=colors)
        log.info(f"Created {len(bars0)} bars for prefill")
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([f'{k}' for k in blocks_list], fontsize=12, rotation=0, ha='center')
        axes[0].set_xlabel('Number of Blocks', fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', labelsize=11, pad=8)
        axes[0].set_ylabel('MAE (ms)', fontsize=12, fontweight='bold')
        axes[0].set_title('Prefill Latency Error by Number of Blocks', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0].set_facecolor('white')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars0, prefill_means, prefill_stds)):
            axes[0].text(bar.get_x() + bar.get_width()/2, mean + std + 0.5,
                        f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        decode_means = [np.mean(blocks_errors[k]['decode']) for k in blocks_list]
        decode_stds = [np.std(blocks_errors[k]['decode']) for k in blocks_list]
        
        bars1 = axes[1].bar(x_pos, decode_means, width=width, yerr=decode_stds, capsize=5,
                           alpha=0.9, edgecolor='black', linewidth=2, color=colors)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([f'{k}' for k in blocks_list], fontsize=12, rotation=0, ha='center')
        axes[1].set_xlabel('Number of Blocks', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', labelsize=11, pad=8)
        axes[1].set_ylabel('MAE (ms/token)', fontsize=12, fontweight='bold')
        axes[1].set_title('Decode Per-Token Latency Error by Number of Blocks', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[1].set_facecolor('white')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars1, decode_means, decode_stds)):
            axes[1].text(bar.get_x() + bar.get_width()/2, mean + std + 0.5,
                        f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Ensure white background and proper styling
        fig.patch.set_facecolor('white')
        for ax in axes:
            ax.set_facecolor('white')
            # Style spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#333333')
            ax.spines['left'].set_color('#333333')
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            # Ensure tick colors are visible
            ax.tick_params(colors='#333333', which='both')
        
        plt.tight_layout()
        # Save with explicit parameters - force RGB mode
        output_path = output_dir / 'errors_by_blocks.png'
        # Use explicit RGB conversion to avoid any transparency issues
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.2,
                   format='png', transparent=False,
                   pil_kwargs={'mode': 'RGB'})
        plt.close(fig)
        log.info(f"Saved errors_by_blocks.png with {len(blocks_list)} block counts: {blocks_list}")
        if output_path.exists():
            log.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Latency Estimator Performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing core experiment results"
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        default=None,
        help="Dataset names to visualize. If not specified, will auto-detect all available datasets"
    )
    parser.add_argument(
        "--use_all_datasets",
        action="store_true",
        help="Use all available datasets in results_dir (auto-detects, ignores --dataset_names)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/visualizations/latency_estimator",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Maximum samples to use for visualization"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    log.info(f"Loading model from {args.checkpoint_path}")
    model = LatencyEstimator()
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    # Auto-detect datasets if needed
    if args.use_all_datasets or args.dataset_names is None:
        log.info("Auto-detecting available datasets...")
        results_path = Path(args.results_dir)
        available_datasets = []
        
        for item in results_path.iterdir():
            if item.is_dir() and item.name != "logs":
                # Check if it contains JSON files
                json_files = list(item.glob("*.json"))
                if json_files:
                    # Convert directory name to dataset name (e.g., "text-vqa" -> "text_vqa")
                    dataset_name = item.name.replace("-", "_")
                    available_datasets.append(dataset_name)
        
        if available_datasets:
            args.dataset_names = sorted(available_datasets)
            log.info(f"Found {len(args.dataset_names)} datasets: {', '.join(args.dataset_names)}")
        else:
            raise ValueError(f"No datasets found in {args.results_dir}")
    else:
        log.info(f"Using specified datasets: {', '.join(args.dataset_names)}")
    
    # Load data
    log.info("Loading evaluation data...")
    data_loader = CoreExpDataLoader(args.results_dir)
    samples = data_loader.load_multiple_datasets(args.dataset_names)
    log.info(f"Loaded {len(samples)} total samples")
    
    # Filter valid samples
    log.info("Filtering valid samples...")
    valid_samples = []
    filtered_outliers = 0
    for i, sample in enumerate(samples):
        if (i + 1) % 10000 == 0:
            log.info(f"  Processed {i+1}/{len(samples)} samples, found {len(valid_samples)} valid")
        T_vision_total = sample.get('T_vision_total', 0.0)
        T_LLM_prefill = sample.get('T_LLM_prefill', 0.0)
        T_LLM_decode = sample.get('T_LLM_decode', 0.0)
        output_tokens = sample.get('output_tokens', 1)
        
        # Calculate decode per-token latency
        decode_per_token = T_LLM_decode / max(output_tokens, 1)
        
        # Filter out outliers: decode per-token latency > 60ms/token
        if decode_per_token > 60.0:
            filtered_outliers += 1
            continue
        
        if (sample.get('vision_tokens', 0) > 0 and
            sample.get('text_tokens', 0) > 0 and
            T_vision_total > 0 and
            T_LLM_prefill > 0 and
            T_LLM_decode > 0 and
            output_tokens > 0):
            valid_samples.append(sample)
    
    if filtered_outliers > 0:
        log.info(f"Filtered out {filtered_outliers} outliers (decode per-token latency > 60ms/token)")
    
    log.info(f"Found {len(valid_samples)} valid samples")
    
    # Limit samples if needed
    if len(valid_samples) > args.max_samples:
        log.info(f"Limiting to {args.max_samples} samples (from {len(valid_samples)} valid samples)")
        valid_samples = valid_samples[:args.max_samples]
    
    # Create dataset and dataloader
    log.info(f"Creating dataset with {len(valid_samples)} samples...")
    dataset = LatencyEstimatorDataset(valid_samples)
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False,  # Disable pin_memory for stability
    )
    
    # Collect predictions
    log.info(f"Collecting predictions for {len(dataset)} samples (batch_size={args.batch_size})...")
    data = collect_predictions(model, eval_loader, args.device, args.max_samples)
    
    log.info(f"Collected {len(data['prefill']['pred'])} predictions")
    
    # Create visualizations
    log.info("Creating visualizations...")
    
    # 1. Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    plot_scatter(
        data['prefill']['pred'],
        data['prefill']['target'],
        'Prefill Latency: Predicted vs Actual',
        'Actual Prefill Latency (ms)',
        'Predicted Prefill Latency (ms)',
        axes[0]
    )
    
    plot_scatter(
        data['decode']['pred'],
        data['decode']['target'],
        'Decode Per-Token Latency: Predicted vs Actual',
        'Actual Decode Per-Token Latency (ms/token)',
        'Predicted Decode Per-Token Latency (ms/token)',
        axes[1]
    )
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Saved scatter plots to {output_dir / 'scatter_plots.png'}")
    
    # 2. Error distributions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    plot_error_distribution(
        data['prefill']['pred'],
        data['prefill']['target'],
        'Prefill Latency Error Distribution',
        axes[0]
    )
    
    plot_error_distribution(
        data['decode']['pred'],
        data['decode']['target'],
        'Decode Per-Token Latency Error Distribution',
        axes[1]
    )
    
    plt.tight_layout()
    fig.patch.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('white')
    plt.savefig(output_dir / 'error_distributions.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close(fig)
    log.info(f"Saved error distributions to {output_dir / 'error_distributions.png'}")
    
    # 3. Errors by configuration
    plot_by_config(data, output_dir)
    log.info(f"Saved configuration-based plots to {output_dir}")
    
    # 4. Summary statistics
    prefill_mae = np.mean(np.abs(data['prefill']['pred'] - data['prefill']['target']))
    prefill_rmse = np.sqrt(np.mean((data['prefill']['pred'] - data['prefill']['target']) ** 2))
    decode_mae = np.mean(np.abs(data['decode']['pred'] - data['decode']['target']))
    decode_rmse = np.sqrt(np.mean((data['decode']['pred'] - data['decode']['target']) ** 2))
    
    summary = {
        'prefill': {
            'mae': float(prefill_mae),
            'rmse': float(prefill_rmse),
        },
        'decode': {
            'mae': float(decode_mae),
            'rmse': float(decode_rmse),
        },
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    log.info("\n" + "=" * 80)
    log.info("Visualization Summary")
    log.info("=" * 80)
    log.info(f"Prefill Latency - MAE: {prefill_mae:.2f}ms, RMSE: {prefill_rmse:.2f}ms")
    log.info(f"Decode Per-Token Latency - MAE: {decode_mae:.3f}ms/token, RMSE: {decode_rmse:.3f}ms/token")
    log.info(f"\nAll visualizations saved to: {output_dir}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

