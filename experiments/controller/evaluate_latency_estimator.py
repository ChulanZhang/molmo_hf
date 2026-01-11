"""
Evaluate trained Latency Estimator model.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.controller.core_exp_data_loader import CoreExpDataLoader
from experiments.controller.latency_estimator import LatencyEstimator, LatencyEstimatorTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


class LatencyEstimatorDataset:
    """Dataset for latency estimator evaluation."""
    
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
            'T_LLM_decode': torch.tensor(T_LLM_decode, dtype=torch.float32),  # Total decode latency (target)
            'T_decode_per_token_avg': torch.tensor(T_decode_per_token_avg, dtype=torch.float32),  # Average (for reference)
            'tier': tier,
            'top_k_value': sample.get('top_k', 8),
            'num_blocks_value': sample.get('num_active_blocks', 16),
        }


def evaluate_model(
    model: LatencyEstimator,
    data_loader: DataLoader,
    device: str = "cuda",
) -> Dict:
    """Evaluate model on dataset."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_errors = []
    errors_by_config = {}
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
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
            
            # Compute errors
            error_prefill = np.abs(pred_prefill - target_prefill)
            error_decode_total = np.abs(pred_decode_total.cpu().numpy() - target_decode_total)
            error_decode_avg = np.abs(pred_decode_avg - target_decode_avg)
            
            all_predictions.append({
                'prefill': pred_prefill,
                'decode_total': pred_decode_total.cpu().numpy(),
                'decode_avg': pred_decode_avg,
            })
            all_targets.append({
                'prefill': target_prefill,
                'decode_total': target_decode_total,
                'decode_avg': target_decode_avg,
            })
            all_errors.append({
                'prefill': error_prefill,
                'decode_total': error_decode_total,
                'decode_avg': error_decode_avg,
            })
            
            # Group by configuration
            tiers = batch['tier']
            top_ks = batch['top_k_value']
            num_blocks = batch['num_blocks_value']
            
            for i in range(len(tiers)):
                config_key = f"{tiers[i]}_topk{top_ks[i]}_blocks{num_blocks[i]}"
                if config_key not in errors_by_config:
                    errors_by_config[config_key] = {
                        'prefill_errors': [],
                        'decode_total_errors': [],
                        'decode_avg_errors': [],
                        'count': 0,
                    }
                errors_by_config[config_key]['prefill_errors'].append(error_prefill[i])
                errors_by_config[config_key]['decode_total_errors'].append(error_decode_total[i])
                errors_by_config[config_key]['decode_avg_errors'].append(error_decode_avg[i])
                errors_by_config[config_key]['count'] += 1
    
    # Aggregate metrics
    all_prefill_pred = np.concatenate([p['prefill'] for p in all_predictions])
    all_prefill_target = np.concatenate([t['prefill'] for t in all_targets])
    all_decode_total_pred = np.concatenate([p['decode_total'] for p in all_predictions])
    all_decode_total_target = np.concatenate([t['decode_total'] for t in all_targets])
    all_decode_avg_pred = np.concatenate([p['decode_avg'] for p in all_predictions])
    all_decode_avg_target = np.concatenate([t['decode_avg'] for t in all_targets])
    
    all_prefill_error = np.concatenate([e['prefill'] for e in all_errors])
    all_decode_total_error = np.concatenate([e['decode_total'] for e in all_errors])
    all_decode_avg_error = np.concatenate([e['decode_avg'] for e in all_errors])
    
    # Overall metrics
    metrics = {
        'prefill': {
            'mae': float(np.mean(all_prefill_error)),
            'rmse': float(np.sqrt(np.mean(all_prefill_error ** 2))),
            'mape': float(np.mean(np.abs(all_prefill_error / (all_prefill_target + 1e-6)) * 100)),
            'r2': float(1 - np.sum((all_prefill_target - all_prefill_pred) ** 2) / 
                        np.sum((all_prefill_target - np.mean(all_prefill_target)) ** 2)),
            'mean_target': float(np.mean(all_prefill_target)),
            'std_target': float(np.std(all_prefill_target)),
        },
        'decode_total': {
            'mae': float(np.mean(all_decode_total_error)),
            'rmse': float(np.sqrt(np.mean(all_decode_total_error ** 2))),
            'mape': float(np.mean(np.abs(all_decode_total_error / (all_decode_total_target + 1e-6)) * 100)),
            'r2': float(1 - np.sum((all_decode_total_target - all_decode_total_pred) ** 2) / 
                        np.sum((all_decode_total_target - np.mean(all_decode_total_target)) ** 2)),
            'mean_target': float(np.mean(all_decode_total_target)),
            'std_target': float(np.std(all_decode_total_target)),
        },
        'decode_avg': {
            'mae': float(np.mean(all_decode_avg_error)),
            'rmse': float(np.sqrt(np.mean(all_decode_avg_error ** 2))),
            'mape': float(np.mean(np.abs(all_decode_avg_error / (all_decode_avg_target + 1e-6)) * 100)),
            'r2': float(1 - np.sum((all_decode_avg_target - all_decode_avg_pred) ** 2) / 
                        np.sum((all_decode_avg_target - np.mean(all_decode_avg_target)) ** 2)),
            'mean_target': float(np.mean(all_decode_avg_target)),
            'std_target': float(np.std(all_decode_avg_target)),
        },
    }
    
    # Per-configuration metrics
    config_metrics = {}
    for config_key, config_data in errors_by_config.items():
        config_metrics[config_key] = {
            'count': config_data['count'],
            'prefill_mae': float(np.mean(config_data['prefill_errors'])),
            'decode_total_mae': float(np.mean(config_data['decode_total_errors'])),
            'decode_avg_mae': float(np.mean(config_data['decode_avg_errors'])),
        }
    
    return {
        'overall_metrics': metrics,
        'config_metrics': config_metrics,
    }


def load_checkpoint_metrics(checkpoint_path: str) -> Dict:
    """Load metrics from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint.get('val_metrics', {})


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Latency Estimator",
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
        help="Directory containing core experiment results for evaluation"
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        default=None,
        help="Dataset names to evaluate on. If not specified, will auto-detect all available datasets"
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
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save evaluation results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Load model
    log.info(f"Loading model from {args.checkpoint_path}")
    model = LatencyEstimator()
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    # Load checkpoint metrics
    checkpoint_metrics = load_checkpoint_metrics(args.checkpoint_path)
    if checkpoint_metrics:
        log.info("Checkpoint validation metrics:")
        for key, value in checkpoint_metrics.items():
            log.info(f"  {key}: {value:.4f}")
    
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
    
    # Load evaluation data
    log.info("Loading evaluation data...")
    data_loader = CoreExpDataLoader(args.results_dir)
    samples = data_loader.load_multiple_datasets(args.dataset_names)
    
    # Filter valid samples
    valid_samples = []
    filtered_outliers = 0
    for sample in samples:
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
    
    log.info(f"Found {len(valid_samples)} valid samples for evaluation")
    
    # Create dataset and dataloader
    dataset = LatencyEstimatorDataset(valid_samples)
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Evaluate
    log.info("Evaluating model...")
    results = evaluate_model(model, eval_loader, args.device)
    
    # Print results
    log.info("=" * 80)
    log.info("Evaluation Results")
    log.info("=" * 80)
    
    log.info("\nOverall Metrics:")
    log.info("Prefill Latency:")
    log.info(f"  MAE: {results['overall_metrics']['prefill']['mae']:.2f}ms")
    log.info(f"  RMSE: {results['overall_metrics']['prefill']['rmse']:.2f}ms")
    log.info(f"  MAPE: {results['overall_metrics']['prefill']['mape']:.2f}%")
    log.info(f"  R²: {results['overall_metrics']['prefill']['r2']:.4f}")
    log.info(f"  Mean Target: {results['overall_metrics']['prefill']['mean_target']:.2f}ms")
    log.info(f"  Std Target: {results['overall_metrics']['prefill']['std_target']:.2f}ms")
    
    # Calculate relative error for prefill
    prefill_rel_error = (results['overall_metrics']['prefill']['mae'] / 
                        results['overall_metrics']['prefill']['mean_target'] * 100)
    log.info(f"  Relative Error: {prefill_rel_error:.2f}%")
    
    log.info("\nDecode Total Latency (Sum of Positioned Latencies):")
    decode_total_metrics = results['overall_metrics']['decode_total']
    log.info(f"  MAE: {decode_total_metrics['mae']:.2f}ms")
    log.info(f"  RMSE: {decode_total_metrics['rmse']:.2f}ms")
    log.info(f"  MAPE: {decode_total_metrics['mape']:.2f}%")
    log.info(f"  R²: {decode_total_metrics['r2']:.4f}")
    log.info(f"  Mean Target: {decode_total_metrics['mean_target']:.2f}ms")
    log.info(f"  Std Target: {decode_total_metrics['std_target']:.2f}ms")
    
    # Calculate relative errors
    decode_total_rel_error = (decode_total_metrics['mae'] / 
                              decode_total_metrics['mean_target'] * 100)
    log.info(f"  Relative Error: {decode_total_rel_error:.2f}%")
    
    log.info("\nDecode Average Per-Token Latency (Reference):")
    decode_avg_metrics = results['overall_metrics']['decode_avg']
    log.info(f"  MAE: {decode_avg_metrics['mae']:.3f}ms/token")
    log.info(f"  RMSE: {decode_avg_metrics['rmse']:.3f}ms/token")
    log.info(f"  MAPE: {decode_avg_metrics['mape']:.2f}%")
    log.info(f"  R²: {decode_avg_metrics['r2']:.4f}")
    log.info(f"  Mean Target: {decode_avg_metrics['mean_target']:.3f}ms/token")
    log.info(f"  Std Target: {decode_avg_metrics['std_target']:.3f}ms/token")
    
    decode_avg_rel_error = (decode_avg_metrics['mae'] / 
                            decode_avg_metrics['mean_target'] * 100)
    log.info(f"  Relative Error: {decode_avg_rel_error:.2f}%")
    
    log.info("\n" + "=" * 80)
    log.info("Performance Assessment:")
    log.info("=" * 80)
    log.info("Prefill Latency (Primary Metric):")
    if prefill_rel_error < 5.0:
        log.info(f"  ✓ Excellent: Relative error {prefill_rel_error:.2f}% < 5%")
    elif prefill_rel_error < 10.0:
        log.info(f"  ✓ Good: Relative error {prefill_rel_error:.2f}% < 10%")
    else:
        log.info(f"  ⚠ Needs improvement: Relative error {prefill_rel_error:.2f}% >= 10%")
    
    log.info("Decode Total Latency (Sum of Positioned Latencies):")
    if decode_total_rel_error < 10.0:
        log.info(f"  ✓ Excellent: Relative error {decode_total_rel_error:.2f}% < 10%")
    elif decode_total_rel_error < 20.0:
        log.info(f"  ✓ Good: Relative error {decode_total_rel_error:.2f}% < 20%")
    else:
        log.info(f"  ⚠ Acceptable: Relative error {decode_total_rel_error:.2f}% (may need improvement)")
    
    log.info(f"Decode Average Per-Token (Reference): MAPE {decode_avg_rel_error:.2f}%")
    
    log.info("\nPer-Configuration Metrics (Top 10 by sample count):")
    sorted_configs = sorted(
        results['config_metrics'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )[:10]
    
    for config_key, config_metrics in sorted_configs:
        log.info(f"  {config_key}:")
        log.info(f"    Count: {config_metrics['count']}")
        log.info(f"    Prefill MAE: {config_metrics['prefill_mae']:.2f}ms")
        log.info(f"    Decode Total MAE: {config_metrics['decode_total_mae']:.2f}ms")
        log.info(f"    Decode Average MAE: {config_metrics['decode_avg_mae']:.3f}ms/token")
    
    # Save results
    if args.output_file:
        output_data = {
            'checkpoint_path': args.checkpoint_path,
            'checkpoint_metrics': checkpoint_metrics,
            'evaluation_results': results,
        }
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        log.info(f"\nResults saved to {args.output_file}")
    
    log.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()

