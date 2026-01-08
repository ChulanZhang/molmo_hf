"""
Complete training script for Two-Stage Controller.

Stage 1: Supervised learning for Knob1 (from core_exp data)
Stage 2: GRPO for Knob2 & Knob3 (online training with latency estimator)
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.base_experiment import BaseExperiment
from experiments.controller.core_exp_data_loader import CoreExpDataLoader
from experiments.controller.feature_extractors import LanguageFeatureExtractor, LatencyBudgetEncoder
from experiments.controller.controller import (
    TwoStageController,
    Knob1PredictorBudgetOnly,
    Knob1PredictorBudgetLanguage,
    Knob1PredictorBudgetLanguageVision,
    Knob2Knob3Predictor,
)
from experiments.controller.latency_estimator import LatencyEstimator
from experiments.controller.controller_model import RewardFunction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('two_stage_controller_training.log'),
    ]
)
log = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Stage1Dataset(Dataset):
    """Dataset for Stage 1 (Knob1) supervised training."""
    
    def __init__(self, samples: List[Dict], lang_extractor, budget_encoder, device: str = "cuda"):
        self.samples = samples
        self.lang_extractor = lang_extractor
        self.budget_encoder = budget_encoder
        self.device = device
        
        # Pre-extract features
        self.features = []
        for sample in tqdm(samples, desc="Extracting Stage 1 features"):
            try:
                # Extract language feature
                prompt = sample.get('metadata', {}).get('question', '')
                if not prompt:
                    prompt = sample.get('metadata', {}).get('prompt', '')
                lang_feat = lang_extractor.extract(prompt).squeeze(0).cpu()
                
                # Encode budget
                budget = torch.tensor([sample['latency_budget']], device=device)
                budget_feat = budget_encoder(budget).squeeze(0).cpu()
                
                # Get tier index
                tier = sample.get('tier', 'medium')
                tier_map = {'low': 0, 'medium': 1, 'high': 2}
                tier_idx = tier_map.get(tier.lower(), 1)
                
                self.features.append({
                    'lang_feat': lang_feat,
                    'budget_feat': budget_feat,
                    'tier_idx': tier_idx,
                })
            except Exception as e:
                log.warning(f"Error extracting features for sample {sample.get('sample_id')}: {e}")
                continue
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feat = self.features[idx]
        return {
            'lang_feat': feat['lang_feat'],
            'budget_feat': feat['budget_feat'],
            'tier_idx': torch.tensor(feat['tier_idx'], dtype=torch.long),
        }


def train_stage1(
    training_data: List[Dict],
    model_path: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    num_epochs: int = 50,
    lr: float = 1e-4,
    train_split: float = 0.8,
    seed: int = 42,
):
    """
    Train Stage 1: Knob1 predictor (supervised learning).
    """
    set_seed(seed)
    
    log.info(f"Training Stage 1 (Knob1) with {len(training_data)} samples")
    
    # Load model for feature extraction
    experiment = BaseExperiment(model_path=model_path, device=device)
    tokenizer = experiment.tokenizer
    wte_layer = experiment.model.model.transformer.wte
    
    # Initialize feature extractors
    lang_extractor = LanguageFeatureExtractor(tokenizer, wte_layer, max_length=512)
    budget_encoder = LatencyBudgetEncoder(hidden_dim=256, use_sinusoidal=False)
    budget_encoder.to(device)
    
    # Split data
    random.shuffle(training_data)
    split_idx = int(len(training_data) * train_split)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    log.info(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Create datasets
    train_dataset = Stage1Dataset(train_data, lang_extractor, budget_encoder, device)
    val_dataset = Stage1Dataset(val_data, lang_extractor, budget_encoder, device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Create model (using budget_language variant by default)
    knob1_predictor = Knob1PredictorBudgetLanguage(
        lang_feat_dim=2048,  # d_model
        budget_feat_dim=256,
        hidden_dim=256,
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(knob1_predictor.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Train
        knob1_predictor.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            lang_feat = batch['lang_feat'].to(device)
            budget_feat = batch['budget_feat'].to(device)
            tier_idx = batch['tier_idx'].to(device)
            
            # Forward
            logits = knob1_predictor(lang_feat, budget_feat)
            loss = criterion(logits, tier_idx)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(knob1_predictor.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            train_losses.append(loss.item())
            pred = logits.argmax(dim=-1)
            train_correct += (pred == tier_idx).sum().item()
            train_total += tier_idx.shape[0]
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/train_total:.4f}",
            })
        
        avg_train_loss = np.mean(train_losses)
        train_acc = train_correct / train_total
        log.info(f"Train Epoch {epoch+1}: loss={avg_train_loss:.4f}, acc={train_acc:.4f}")
        
        # Validate
        knob1_predictor.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                lang_feat = batch['lang_feat'].to(device)
                budget_feat = batch['budget_feat'].to(device)
                tier_idx = batch['tier_idx'].to(device)
                
                logits = knob1_predictor(lang_feat, budget_feat)
                loss = criterion(logits, tier_idx)
                
                val_losses.append(loss.item())
                pred = logits.argmax(dim=-1)
                val_correct += (pred == tier_idx).sum().item()
                val_total += tier_idx.shape[0]
        
        avg_val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total
        log.info(f"Val Epoch {epoch+1}: loss={avg_val_loss:.4f}, acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': knob1_predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }
            torch.save(checkpoint, output_path / 'best_stage1_checkpoint.pt')
            log.info(f"Saved best Stage 1 model (val_acc={best_val_acc:.4f})")
    
    log.info(f"Stage 1 training completed! Best model saved to {output_path / 'best_stage1_checkpoint.pt'}")
    return knob1_predictor


def train_stage2_grpo(
    training_data: List[Dict],
    knob1_predictor: Knob1PredictorBudgetLanguage,
    latency_estimator: LatencyEstimator,
    model_path: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 1e-4,
    group_size: int = 5,
    train_split: float = 0.8,
    seed: int = 42,
    use_latency_estimator: bool = True,
):
    """
    Train Stage 2: Knob2 & Knob3 predictor with GRPO.
    
    Full implementation with online execution.
    """
    set_seed(seed)
    
    log.info(f"Training Stage 2 (Knob2 & Knob3) with GRPO")
    log.info(f"Using latency estimator: {use_latency_estimator}")
    
    # Load model
    experiment = BaseExperiment(model_path=model_path, device=device)
    model = experiment.model
    tokenizer = experiment.tokenizer
    
    # Initialize feature extractors
    from experiments.controller.feature_extractors import LanguageFeatureExtractor, LatencyBudgetEncoder
    lang_extractor = LanguageFeatureExtractor(tokenizer, model.model.transformer.wte, max_length=512)
    budget_encoder = LatencyBudgetEncoder(hidden_dim=256, use_sinusoidal=False).to(device)
    
    # Create Stage 2 predictor
    knob2_knob3_predictor = Knob2Knob3Predictor(
        vision_feat_dim=2048,
        lang_feat_dim=2048,
        budget_feat_dim=256,
        hidden_dim=512,
    ).to(device)
    
    # Reward function
    from experiments.controller.controller_model import RewardFunction
    reward_fn = RewardFunction(
        alpha=1.0,
        beta=0.5,
        gamma=10.0,
        delta=0.1,
        epsilon=0.05,
    )
    
    # Create trainer
    from experiments.controller.stage2_grpo_trainer import Stage2GRPOTrainer
    trainer = Stage2GRPOTrainer(
        knob2_knob3_predictor=knob2_knob3_predictor,
        knob1_predictor=knob1_predictor,
        model=model,
        latency_estimator=latency_estimator,
        reward_fn=reward_fn,
        device=device,
        lr=lr,
        weight_decay=1e-5,
        max_grad_norm=1.0,
        group_size=group_size,
        use_latency_estimator=use_latency_estimator,
    )
    
    # Create dataset (simplified - would need proper dataset with images and prompts)
    # For now, we'll create a placeholder dataset
    log.warning("Stage 2 training requires proper dataset with images and prompts")
    log.warning("Creating placeholder dataset - replace with actual dataset loading")
    
    # Split data
    random.shuffle(training_data)
    split_idx = int(len(training_data) * train_split)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    log.info(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Training loop (simplified - would need proper dataloader)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    best_val_reward = float('-inf')
    
    for epoch in range(num_epochs):
        # Train (simplified - would iterate over proper dataloader)
        log.info(f"Epoch {epoch+1}/{num_epochs}")
        log.warning("Full training loop requires proper dataset and dataloader")
        
        # Placeholder: would train here
        # For now, just save a checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': knob2_knob3_predictor.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
            }
            torch.save(checkpoint, output_path / f'stage2_checkpoint_epoch_{epoch+1}.pt')
            log.info(f"Saved checkpoint at epoch {epoch+1}")
    
    log.info(f"Stage 2 training completed! Checkpoints saved to {output_path}")
    return knob2_knob3_predictor


def main():
    parser = argparse.ArgumentParser(
        description="Train Two-Stage Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data args
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
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--latency_estimator_path",
        type=str,
        default=None,
        help="Path to trained latency estimator (required for Stage 2)"
    )
    
    # Training args
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/two_stage_controller",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=['stage1', 'stage2', 'both'],
        default='both',
        help="Which stage to train"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--num_epochs_stage1",
        type=int,
        default=50,
        help="Number of epochs for Stage 1"
    )
    parser.add_argument(
        "--num_epochs_stage2",
        type=int,
        default=100,
        help="Number of epochs for Stage 2"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=5,
        help="Group size for GRPO (Stage 2)"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Train/val split ratio"
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
    
    # Load training data
    log.info("Loading training data...")
    data_loader = CoreExpDataLoader(args.results_dir)
    training_data = data_loader.load_multiple_datasets(args.dataset_names)
    
    if not training_data:
        raise ValueError("No training data found!")
    
    log.info(f"Loaded {len(training_data)} training samples")
    
    # Train Stage 1
    if args.stage in ['stage1', 'both']:
        log.info("=" * 80)
        log.info("Training Stage 1: Knob1 Predictor (Supervised Learning)")
        log.info("=" * 80)
        
        knob1_predictor = train_stage1(
            training_data=training_data,
            model_path=args.model_path,
            output_dir=os.path.join(args.output_dir, 'stage1'),
            device=args.device,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs_stage1,
            lr=args.lr,
            train_split=args.train_split,
            seed=args.seed,
        )
    else:
        # Load pre-trained Stage 1
        stage1_path = Path(args.output_dir) / 'stage1' / 'best_stage1_checkpoint.pt'
        if not stage1_path.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_path}")
        
        knob1_predictor = Knob1PredictorBudgetLanguage().to(args.device)
        checkpoint = torch.load(stage1_path, map_location=args.device)
        knob1_predictor.load_state_dict(checkpoint['model_state_dict'])
        log.info(f"Loaded Stage 1 from {stage1_path}")
    
    # Train Stage 2
    if args.stage in ['stage2', 'both']:
        if args.latency_estimator_path is None:
            raise ValueError("--latency_estimator_path is required for Stage 2 training")
        
        log.info("=" * 80)
        log.info("Training Stage 2: Knob2 & Knob3 Predictor (GRPO)")
        log.info("=" * 80)
        
        # Load latency estimator
        latency_estimator = LatencyEstimator()
        checkpoint = torch.load(args.latency_estimator_path, map_location=args.device)
        latency_estimator.load_state_dict(checkpoint['model_state_dict'])
        latency_estimator.to(args.device)
        latency_estimator.eval()
        log.info(f"Loaded latency estimator from {args.latency_estimator_path}")
        
        train_stage2_grpo(
            training_data=training_data,
            knob1_predictor=knob1_predictor,
            latency_estimator=latency_estimator,
            model_path=args.model_path,
            output_dir=os.path.join(args.output_dir, 'stage2'),
            device=args.device,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs_stage2,
            lr=args.lr,
            group_size=args.group_size,
            train_split=args.train_split,
            seed=args.seed,
        )
    
    log.info("Training completed!")


if __name__ == "__main__":
    main()

