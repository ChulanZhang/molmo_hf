"""
Main script to train the GRPO Controller.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project path
sys.path.append(os.getcwd())

from experiments.controller.data_preparation import (
    ExpDataLoader,
    TrainingDataBuilder,
    save_training_data,
    load_training_data,
)
from experiments.controller.controller_model import GRPOController, RewardFunction
from experiments.controller.grpo_trainer import GRPOTrainer, ControllerDataset

# Model-related imports
from experiments.base_experiment import BaseExperiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def prepare_data(
    exp5_dir: str,
    exp6_dir: str,
    dataset_name: str,
    model_path: str,
    output_data_path: str,
    device: str = "cuda",
    batch_size: int = 32,
    force_recompute: bool = False,
):
    """Prepare training data."""
    
    # Reuse cached data if present
    if os.path.exists(output_data_path) and not force_recompute:
        log.info(f"Loading existing training data from {output_data_path}")
        return load_training_data(output_data_path)
    
    log.info("Preparing training data from exp5 and exp6 results...")
    
    # Load exp5 and exp6 results
    data_loader = ExpDataLoader(exp5_dir, exp6_dir)
    exp5_results = data_loader.load_exp5_results(dataset_name)
    exp6_results = data_loader.load_exp6_results(dataset_name)
    
    # Merge data
    merged_results = data_loader.merge_exp5_exp6(exp5_results, exp6_results)
    
    if not merged_results:
        raise ValueError("No data found in exp5 and exp6 results!")
    
    # Load model for feature extraction
    log.info("Loading model for feature extraction...")
    experiment = BaseExperiment(
        model_path=model_path,
        device=device,
    )
    
    # Load dataset
    from molmo.data import get_dataset_by_name
    dataset = get_dataset_by_name(dataset_name, split="validation")
    
    # Extract features
    log.info("Extracting features from dataset...")
    builder = TrainingDataBuilder(
        model=experiment.model,
        tokenizer=experiment.tokenizer,
        device=device,
    )
    
    # Collect all sample_ids
    sample_ids = list(set([r.get('sample_id') for r in merged_results if r.get('sample_id') is not None]))
    log.info(f"Extracting features for {len(sample_ids)} samples...")
    
    features = builder.extract_features(dataset, sample_ids, batch_size=batch_size)
    
    # Build training data
    log.info("Building training data...")
    training_data = builder.build_training_data(merged_results, features)
    
    # Save data
    log.info(f"Saving training data to {output_data_path}...")
    save_training_data(training_data, output_data_path)
    
    return training_data


def train_controller(
    training_data: list,
    model_path: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    num_epochs: int = 100,
    lr: float = 1e-4,
    group_size: int = 8,
    train_split: float = 0.8,
    reward_alpha: float = 1.0,
    reward_beta: float = 0.5,
    reward_gamma: float = 10.0,
    reward_delta: float = 0.1,
    reward_epsilon: float = 0.05,
):
    """Train the controller."""
    
    log.info(f"Training controller with {len(training_data)} samples")
    
    # Split train/val
    import random
    random.seed(42)
    random.shuffle(training_data)
    
    split_idx = int(len(training_data) * train_split)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    log.info(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Create datasets
    train_dataset = ControllerDataset(train_data)
    val_dataset = ControllerDataset(val_data)
    
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
    
    # Build controller from feature dims
    sample = train_data[0]
    image_feat_dim = sample['image_feature'].shape[0]
    lang_feat_dim = sample['language_feature'].shape[0]
    
    log.info(f"Image feature dim: {image_feat_dim}, Language feature dim: {lang_feat_dim}")
    
    controller = GRPOController(
        image_feat_dim=image_feat_dim,
        lang_feat_dim=lang_feat_dim,
        budget_dim=32,
        hidden_dim=256,
        num_max_crops=6,
        num_top_k=8,
        num_blocks=9,
        dropout=0.1,
    )
    
    # Build reward function
    reward_fn = RewardFunction(
        alpha=reward_alpha,
        beta=reward_beta,
        gamma=reward_gamma,
        delta=reward_delta,
        epsilon=reward_epsilon,
    )
    
    # Build trainer
    trainer = GRPOTrainer(
        controller=controller,
        reward_fn=reward_fn,
        device=device,
        group_size=group_size,
        lr=lr,
        weight_decay=1e-5,
    )
    
    # Train
    os.makedirs(output_dir, exist_ok=True)
    trainer.train(
        train_loader=train_loader,
        num_epochs=num_epochs,
        val_loader=val_loader,
        save_dir=output_dir,
        save_every=10,
    )
    
    log.info(f"Training completed! Checkpoints saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train GRPO Controller")
    
    # Data args
    parser.add_argument("--exp5_dir", type=str, required=True,
                       help="Directory containing exp5 results")
    parser.add_argument("--exp6_dir", type=str, required=True,
                       help="Directory containing exp6 results")
    parser.add_argument("--dataset_name", type=str, default="text_vqa",
                       help="Dataset name")
    parser.add_argument("--model_path", type=str, default="checkpoints",
                       help="Path to model checkpoint")
    parser.add_argument("--output_data_path", type=str, default="./data/controller_training_data.json",
                       help="Path to save/load training data")
    parser.add_argument("--force_recompute", action="store_true",
                       help="Force recompute features even if data exists")
    
    # Training args
    parser.add_argument("--output_dir", type=str, default="./checkpoints/controller",
                       help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--group_size", type=int, default=8,
                       help="Group size for GRPO")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Train/val split ratio")
    
    # Reward args
    parser.add_argument("--reward_alpha", type=float, default=1.0,
                       help="Accuracy weight")
    parser.add_argument("--reward_beta", type=float, default=0.5,
                       help="Latency penalty weight")
    parser.add_argument("--reward_gamma", type=float, default=10.0,
                       help="Budget violation penalty weight")
    parser.add_argument("--reward_delta", type=float, default=0.1,
                       help="Efficiency bonus weight")
    parser.add_argument("--reward_epsilon", type=float, default=0.05,
                       help="Complexity penalty weight")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Prepare data
    training_data = prepare_data(
        exp5_dir=args.exp5_dir,
        exp6_dir=args.exp6_dir,
        dataset_name=args.dataset_name,
        model_path=args.model_path,
        output_data_path=args.output_data_path,
        device=args.device,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
    )
    
    # Train controller
    train_controller(
        training_data=training_data,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        group_size=args.group_size,
        train_split=args.train_split,
        reward_alpha=args.reward_alpha,
        reward_beta=args.reward_beta,
        reward_gamma=args.reward_gamma,
        reward_delta=args.reward_delta,
        reward_epsilon=args.reward_epsilon,
    )


if __name__ == "__main__":
    main()

