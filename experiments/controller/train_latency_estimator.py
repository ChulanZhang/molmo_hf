"""
Train Latency Estimator on core experiment data.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.controller.core_exp_data_loader import CoreExpDataLoader
from experiments.controller.latency_estimator import LatencyEstimator, LatencyEstimatorTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


class LatencyEstimatorDataset(Dataset):
    """Dataset for latency estimator training."""
    
    def __init__(self, samples: List[Dict]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert tier to index
        tier = sample.get('tier', 'medium')
        tier_map = {'low': 0, 'medium': 1, 'high': 2}
        tier_idx = tier_map.get(tier.lower(), 1)
        
        return {
            'vision_tokens': torch.tensor(sample.get('vision_tokens', 0), dtype=torch.long),
            'text_tokens': torch.tensor(sample.get('text_tokens', 0), dtype=torch.long),
            'output_tokens': torch.tensor(sample.get('output_tokens', 0), dtype=torch.long),
            'tier_idx': torch.tensor(tier_idx, dtype=torch.long),
            'top_k': torch.tensor(sample.get('top_k', 8), dtype=torch.long),
            'num_active_blocks': torch.tensor(sample.get('num_active_blocks', 16), dtype=torch.long),
            'T_vision_total': torch.tensor(sample.get('T_vision_total', 0.0), dtype=torch.float32),
            'T_LLM_prefill': torch.tensor(sample.get('T_LLM_prefill', 0.0), dtype=torch.float32),
            'T_LLM_decode': torch.tensor(sample.get('T_LLM_decode', 0.0), dtype=torch.float32),
            'T_total': torch.tensor(sample.get('T_total', 0.0), dtype=torch.float32),
        }


def prepare_training_data(
    results_dir: str,
    dataset_names: List[str],
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    Prepare training data from core experiment results.
    
    Args:
        results_dir: Directory containing core experiment results
        dataset_names: List of dataset names
        max_samples: Maximum samples per dataset
    
    Returns:
        List of training samples
    """
    log.info("Loading core experiment results...")
    
    data_loader = CoreExpDataLoader(results_dir)
    samples = data_loader.load_multiple_datasets(dataset_names)
    
    if not samples:
        raise ValueError("No training samples found!")
    
    # Filter valid samples
    valid_samples = []
    for sample in samples:
        # Check required fields
        if (sample.get('vision_tokens', 0) > 0 and
            sample.get('text_tokens', 0) > 0 and
            sample.get('T_total', 0) > 0):
            valid_samples.append(sample)
    
    log.info(f"Found {len(valid_samples)} valid samples")
    
    # Limit samples if needed
    if max_samples and len(valid_samples) > max_samples:
        random.shuffle(valid_samples)
        valid_samples = valid_samples[:max_samples]
        log.info(f"Limited to {max_samples} samples")
    
    return valid_samples


def train_latency_estimator(
    training_data: List[Dict],
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    num_epochs: int = 50,
    lr: float = 1e-3,
    train_split: float = 0.8,
    seed: int = 42,
):
    """
    Train latency estimator.
    
    Args:
        training_data: List of training samples
        output_dir: Directory to save checkpoints
        device: Device to use
        batch_size: Batch size
        num_epochs: Number of epochs
        lr: Learning rate
        train_split: Train/val split ratio
        seed: Random seed
    """
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    log.info(f"Training latency estimator with {len(training_data)} samples")
    
    # Split train/val
    random.shuffle(training_data)
    split_idx = int(len(training_data) * train_split)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    log.info(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Create datasets
    train_dataset = LatencyEstimatorDataset(train_data)
    val_dataset = LatencyEstimatorDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    model = LatencyEstimator(
        hidden_dim=256,
        num_layers=2,
        use_output_tokens=True,
    )
    
    # Create trainer
    trainer = LatencyEstimatorTrainer(
        model=model,
        device=device,
        lr=lr,
        weight_decay=1e-5,
    )
    
    # Training loop
    best_val_loss = float('inf')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_metrics = {
            'loss': [],
            'mae_total': [],
            'rel_error_total': [],
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            metrics = trainer.train_step(batch)
            
            for key in train_metrics:
                if key in metrics:
                    train_metrics[key].append(metrics[key])
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'mae': f"{metrics['mae_total']:.2f}ms",
            })
        
        avg_train_metrics = {k: np.mean(v) for k, v in train_metrics.items() if v}
        log.info(f"Train Epoch {epoch+1}: {avg_train_metrics}")
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        log.info(f"Val Epoch {epoch+1}: {val_metrics}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_metrics': val_metrics,
            }
            torch.save(checkpoint, output_path / 'best_latency_estimator.pt')
            log.info(f"Saved best model (val_loss={best_val_loss:.4f})")
        
        # Periodic save
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_metrics': val_metrics,
            }
            torch.save(checkpoint, output_path / f'latency_estimator_epoch_{epoch+1}.pt')
    
    log.info(f"Training completed! Best model saved to {output_path / 'best_latency_estimator.pt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Latency Estimator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        default=["text_vqa", "coco_2014_vqa"],
        help="Dataset names to load"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/latency_estimator",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Train/val split ratio"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Prepare data
    training_data = prepare_training_data(
        results_dir=args.results_dir,
        dataset_names=args.dataset_names,
        max_samples=args.max_samples,
    )
    
    # Train
    train_latency_estimator(
        training_data=training_data,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        train_split=args.train_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

