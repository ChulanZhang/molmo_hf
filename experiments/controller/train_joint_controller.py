"""
One-stage controller training script.
Trains a single controller that predicts all knobs upfront.

Key design:
1. One-Stage Controller: Predicts tier, block mask, and per-block top-k upfront
2. Direct Latency Measurement: Uses hooks to measure actual latency (no estimator)
3. Budget Token: Encoded as d_model-dim token, concatenated to input sequence
4. Decode Phase: Uses prefill configuration, no controller re-run
5. Block Activation Quota: At least 12 blocks, at most 16 blocks (including block0)
6. Block0 Top-K: Fixed at 8

Training process:
- Batch size = 1 per sample (for accurate latency measurement)
- Latency budget sampled from [170ms, 380ms] uniformly
- Reward: accuracy + latency constraint + budget violation penalty
- Controller + budget encoder MLP are trainable
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
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel, DistributedDataParallel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.controller.model_loader import load_model_and_tokenizer
from experiments.controller.core_exp_data_loader import CoreExpDataLoader
from experiments.controller.feature_extractors import LanguageFeatureExtractor, LatencyBudgetEncoder
from experiments.controller.controller import (
    OneStageControllerPredictor,
)
from experiments.controller.controller_model import RewardFunction
from experiments.controller.joint_grpo_trainer import JointGRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('results/logs/training/joint_controller_training.log'),
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


def train_joint_controller(
    training_data: List[Dict],  # Not used for online training, kept for compatibility
    model_path: str,
    output_dir: str,
    dataset_names: List[str],  # Dataset names for online training
    device: str = "cuda",
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 1e-4,
    group_size: int = 5,
    train_split: float = 0.8,
    seed: int = 42,
    load_stage1_checkpoint: Optional[str] = None,  # Optional: load pre-trained controller
    latency_budget_min: float = 170.0,
    latency_budget_max: float = 380.0,
    importance_scores_file: Optional[str] = None,
    use_multi_gpu: bool = False,  # If True, use DataParallel for multi-GPU training
    use_wandb: bool = False,  # If True, enable Weights & Biases logging
    wandb_project: Optional[str] = None,  # W&B project name (default: "molmo-controller")
    wandb_entity: Optional[str] = None,  # W&B entity/team name
    wandb_name: Optional[str] = None,  # W&B run name (default: auto-generated)
):
    """
    Train Stage1 and Stage2 jointly with GRPO.
    
    Args:
        training_data: Training data from core_exp results
        model_path: Path to model checkpoint
        output_dir: Output directory for checkpoints
        device: Device to use
        batch_size: Batch size
        num_epochs: Number of epochs
        lr: Learning rate for Stage2
        stage1_lr_ratio: Learning rate ratio for Stage1 (relative to Stage2)
        group_size: Group size for GRPO
        train_split: Train/val split ratio
        seed: Random seed
        load_stage1_checkpoint: Optional path to pre-trained Stage1 checkpoint
        latency_budget_min: Minimum latency budget
        latency_budget_max: Maximum latency budget
        importance_scores_file: Path to importance scores JSON file
    """
    set_seed(seed)
    
    log.info(f"Training One-Stage Controller with {len(training_data)} samples")
    
    # Load model
    model, tokenizer, processor = load_model_and_tokenizer(model_path=model_path, device=device)
    wte_layer = model.model.transformer.wte
    
    # Initialize feature extractors
    lang_extractor = LanguageFeatureExtractor(tokenizer, wte_layer, max_length=512)
    # Latency budget encoder: outputs d_model dimension token (not 256-dim feature)
    budget_encoder = LatencyBudgetEncoder(d_model=2048, use_sinusoidal=True).to(device)
    
    # Create one-stage controller
    one_stage_controller = OneStageControllerPredictor(
        vision_dim=768,  # CLIP vision encoder output dimension
        lang_dim=2048,   # d_model for language features
        budget_dim=2048, # d_model for budget features
        hidden_dim=256,
        dropout=0.1,
        total_blocks=16,
    ).to(device)
    
    # Load pre-trained checkpoint if provided
    if load_stage1_checkpoint:
        checkpoint = torch.load(load_stage1_checkpoint, map_location=device)
        if 'one_stage_controller_state_dict' in checkpoint:
            one_stage_controller.load_state_dict(checkpoint['one_stage_controller_state_dict'])
            log.info(f"Loaded pre-trained one-stage controller from {load_stage1_checkpoint}")
        elif 'model_state_dict' in checkpoint:
            one_stage_controller.load_state_dict(checkpoint['model_state_dict'])
            log.info(f"Loaded pre-trained controller from {load_stage1_checkpoint}")
        else:
            log.warning(f"Could not find controller state dict in {load_stage1_checkpoint}")
    
    # Reward function
    reward_fn = RewardFunction(
        alpha=1.0,      # accuracy weight
        beta=0.5,       # latency penalty weight
        gamma=10.0,     # budget violation penalty weight
        delta=0.1,      # efficiency bonus weight
        epsilon=0.05,   # complexity penalty weight
    )
    
    # Load importance scores for block selection (if provided)
    if importance_scores_file and not os.path.exists(importance_scores_file):
        log.warning(f"Importance scores file not found: {importance_scores_file}, will use prefix blocks")
        importance_scores_file = None
    
    # Multi-GPU support: Use DataParallel if multiple GPUs available
    num_gpus = torch.cuda.device_count()
    if use_multi_gpu and num_gpus > 1:
        log.info(f"Using {num_gpus} GPUs with DataParallel")
        # Note: We wrap the model, but controllers are small and may not benefit much
        # The main benefit is parallelizing the model forward passes for different samples
        model = DataParallel(model)
        # Controllers are small, but we can still wrap them for consistency
        one_stage_controller = DataParallel(one_stage_controller)
        # When using DataParallel, the main device is cuda:0
        device = "cuda:0"
        # Set batch_size to number of GPUs (per device batch size = 1, global batch size = num_gpus)
        if batch_size != num_gpus:
            log.info(f"Adjusting batch_size from {batch_size} to {num_gpus} (number of GPUs)")
            batch_size = num_gpus
    else:
        log.info(f"Using single GPU: {device}")
        # For single GPU, per device batch size = 1
        if batch_size != 1:
            log.info(f"Adjusting batch_size from {batch_size} to 1 (single GPU, per device batch size = 1)")
            batch_size = 1
    
    # Create trainer
    trainer = JointGRPOTrainer(
        one_stage_controller=one_stage_controller,
        model=model,
        reward_fn=reward_fn,
        budget_encoder=budget_encoder,  # Pass budget_encoder for model forward
        device=device,
        lr=lr,
        weight_decay=1e-5,
        max_grad_norm=1.0,
        group_size=group_size,
        importance_scores_file=importance_scores_file,
        min_active_blocks=12,  # Minimum active blocks (including block0)
        max_active_blocks=16,  # Maximum active blocks (including block0)
    )
    
    # Create online training dataset
    from experiments.controller.online_training_dataset import OnlineTrainingDataset, collate_online_training_batch
    
    # Load datasets for training
    # Note: We limit train samples to avoid very long loading times
    # You can increase this or set to None for full dataset
    train_datasets = []
    val_datasets = []
    
    # Debug: Use only first dataset for now
    log.info("Loading training datasets (DEBUG: using only first dataset)...")
    dataset_name = dataset_names[0]  # Use only first dataset for debugging
    
    # Training dataset - increase data size for better training
    train_dataset = OnlineTrainingDataset(
        dataset_name=dataset_name,
        split="train",
        model_path=None,  # Don't load model in dataset
        device=device,
        num_samples=200,  # Reduced for debugging
        latency_budget_min=latency_budget_min,
        latency_budget_max=latency_budget_max,
        seed=seed,
    )
    train_datasets.append(train_dataset)
    
    # Validation dataset
    log.info(f"Loading validation dataset {dataset_name}...")
    val_dataset = OnlineTrainingDataset(
        dataset_name=dataset_name,
        split="validation",
        model_path=None,  # Don't load model in dataset
        device=device,
        num_samples=50,  # Reduced for debugging
        latency_budget_min=latency_budget_min,
        latency_budget_max=latency_budget_max,
        seed=seed + 1,
    )
    val_datasets.append(val_dataset)
    
    # Combine datasets
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    
    log.info(f"Train dataset: {len(train_dataset)} samples")
    log.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    # Note: collate_fn returns CPU tensors, we'll move to device in train_step
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False,  # Disable pin_memory to avoid issues
        collate_fn=lambda batch: collate_online_training_batch(batch, processor, "cpu"),  # Return CPU tensors
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,  # Disable pin_memory
        collate_fn=lambda batch: collate_online_training_batch(batch, processor, "cpu"),  # Return CPU tensors
    ) if len(val_dataset) > 0 else None
    
    # Training loop
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard writer (always enabled)
    tensorboard_dir = output_path / 'tensorboard'
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    log.info(f"TensorBoard logs will be saved to {tensorboard_dir}")
    log.info(f"To view TensorBoard, run: tensorboard --logdir={tensorboard_dir}")
    
    # Initialize Weights & Biases (optional)
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            
            # Default project name
            project_name = wandb_project or "molmo-controller"
            
            # Auto-generate run name if not provided
            if wandb_name is None:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                wandb_name = f"grpo-g{group_size}-lr{lr}-{timestamp}"
            
            # Initialize wandb
            wandb_dir = output_path / 'wandb'
            wandb_dir.mkdir(parents=True, exist_ok=True)
            
            wandb_run = wandb.init(
                dir=str(wandb_dir),
                project=project_name,
                entity=wandb_entity,
                name=wandb_name,
                config={
                    'model_path': model_path,
                    'output_dir': output_dir,
                    'dataset_names': dataset_names,
                    'device': device,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'lr': lr,
                    'group_size': group_size,
                    'train_split': train_split,
                    'seed': seed,
                    'latency_budget_min': latency_budget_min,
                    'latency_budget_max': latency_budget_max,
                    'importance_scores_file': importance_scores_file,
                    'use_multi_gpu': use_multi_gpu,
                },
                tags=['grpo', 'controller', 'joint-training'] + dataset_names,
            )
            log.info(f"Weights & Biases logging enabled: project={project_name}, run={wandb_name}")
            log.info(f"View run at: {wandb_run.url if hasattr(wandb_run, 'url') else 'https://wandb.ai'}")
        except ImportError:
            log.warning("wandb not installed. Install with: pip install wandb")
            log.warning("Continuing without wandb logging...")
            use_wandb = False
        except Exception as e:
            log.warning(f"Failed to initialize wandb: {e}")
            log.warning("Continuing without wandb logging...")
            use_wandb = False
    
    best_val_reward = float('-inf')
    
    for epoch in range(num_epochs):
        log.info(f"\n{'='*80}")
        log.info(f"Epoch {epoch+1}/{num_epochs}")
        log.info(f"{'='*80}")
        
        # Train
        trainer.one_stage_controller.train()
        
        epoch_metrics = {
            'loss': [],
            'reward_mean': [],
            'accuracy_mean': [],
            'latency_mean': [],
            'budget_violation_rate': [],
        }
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        log_interval = 50  # wandb logging interval (in steps) to surface metrics even if epoch doesn't finish
        for batch_idx, batch in enumerate(pbar):
            try:
                # Clear CUDA cache periodically to avoid memory issues
                if batch_idx % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                metrics = trainer.train_step(
                    batch=batch,
                    lang_extractor=lang_extractor,
                    budget_encoder=budget_encoder,
                )
                
                # Accumulate metrics
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{metrics.get('loss', 0):.4f}",
                    'reward': f"{metrics.get('reward_mean', 0):.4f}",
                    'acc': f"{metrics.get('accuracy_mean', 0):.4f}",
                })

                # Log a lightweight snapshot to wandb every log_interval steps
                if use_wandb and wandb_run is not None and (batch_idx + 1) % log_interval == 0:
                    try:
                        wandb_run.log({
                            'step/loss': metrics.get('loss', 0.0),
                            'step/reward_mean': metrics.get('reward_mean', 0.0),
                            'step/accuracy_mean': metrics.get('accuracy_mean', 0.0),
                            'step/latency_mean': metrics.get('latency_mean', 0.0),
                            'step': epoch * len(train_loader) + batch_idx + 1,
                            'epoch': epoch + 1,
                        })
                    except Exception as e:
                        log.debug(f"Step-level wandb log failed at step {batch_idx+1}: {e}")
                
            except RuntimeError as e:
                error_msg = str(e)
                if "CUDA" in error_msg or "device-side assert" in error_msg:
                    log.error(f"CUDA error in training step {batch_idx}: {e}")
                    log.error("This may be due to multi-GPU issues or index out of bounds.")
                    log.error("Attempting to clear CUDA cache and continue...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        try:
                            torch.cuda.synchronize()
                        except:
                            pass
                    continue
                else:
                    log.error(f"Error in training step {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            except Exception as e:
                log.error(f"Error in training step {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Log epoch metrics
        avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()}
        log.info(f"Train Epoch {epoch+1} Metrics:")
        log.info(f"  Loss: {avg_metrics['loss']:.4f}")
        log.info(f"  Reward: {avg_metrics['reward_mean']:.4f} (std: {np.std(epoch_metrics['reward_mean']) if epoch_metrics['reward_mean'] else 0:.4f})")
        log.info(f"  Accuracy: {avg_metrics['accuracy_mean']:.4f} (std: {np.std(epoch_metrics['accuracy_mean']) if epoch_metrics['accuracy_mean'] else 0:.4f})")
        log.info(f"  Latency: {avg_metrics['latency_mean']:.2f}ms (std: {np.std(epoch_metrics['latency_mean']) if epoch_metrics['latency_mean'] else 0:.2f}ms)")
        log.info(f"  Budget Violation Rate: {avg_metrics['budget_violation_rate']:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar('Train/Loss', avg_metrics['loss'], epoch + 1)
        writer.add_scalar('Train/Reward_Mean', avg_metrics['reward_mean'], epoch + 1)
        writer.add_scalar('Train/Reward_Std', np.std(epoch_metrics['reward_mean']) if epoch_metrics['reward_mean'] else 0.0, epoch + 1)
        writer.add_scalar('Train/Accuracy_Mean', avg_metrics['accuracy_mean'], epoch + 1)
        writer.add_scalar('Train/Accuracy_Std', np.std(epoch_metrics['accuracy_mean']) if epoch_metrics['accuracy_mean'] else 0.0, epoch + 1)
        writer.add_scalar('Train/Latency_Mean', avg_metrics['latency_mean'], epoch + 1)
        writer.add_scalar('Train/Latency_Std', np.std(epoch_metrics['latency_mean']) if epoch_metrics['latency_mean'] else 0.0, epoch + 1)
        writer.add_scalar('Train/Budget_Violation_Rate', avg_metrics['budget_violation_rate'], epoch + 1)
        
        # Log to Weights & Biases (if enabled)
        # Use run.log() instead of wandb.log() to match official wandb style
        if use_wandb and wandb_run is not None:
            try:
                # Log metrics with proper formatting for wandb
                # Use forward slashes for grouping (wandb will create groups automatically)
                metrics_to_log = {
                    'train/loss': avg_metrics['loss'],
                    'train/reward_mean': avg_metrics['reward_mean'],
                    'train/reward_std': np.std(epoch_metrics['reward_mean']) if epoch_metrics['reward_mean'] else 0.0,
                    'train/accuracy_mean': avg_metrics['accuracy_mean'],
                    'train/accuracy_std': np.std(epoch_metrics['accuracy_mean']) if epoch_metrics['accuracy_mean'] else 0.0,
                    'train/latency_mean': avg_metrics['latency_mean'],
                    'train/latency_std': np.std(epoch_metrics['latency_mean']) if epoch_metrics['latency_mean'] else 0.0,
                    'train/budget_violation_rate': avg_metrics['budget_violation_rate'],
                    'epoch': epoch + 1,
                }
                wandb_run.log(metrics_to_log, step=epoch + 1)
                log.debug(f"Logged to wandb: {list(metrics_to_log.keys())}")
            except Exception as e:
                log.warning(f"Failed to log to wandb: {e}")
                import traceback
                log.debug(traceback.format_exc())
        
        # Save training history to CSV
        import csv
        history_file = output_path / 'training_history.csv'
        file_exists = history_file.exists()
        
        with open(history_file, 'a', newline='') as f:
            csv_writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'reward_mean', 'reward_std', 'accuracy_mean', 
                                                   'accuracy_std', 'latency_mean', 'latency_std', 'budget_violation_rate'])
            if not file_exists:
                csv_writer.writeheader()
            csv_writer.writerow({
                'epoch': epoch + 1,
                'loss': avg_metrics['loss'],
                'reward_mean': avg_metrics['reward_mean'],
                'reward_std': np.std(epoch_metrics['reward_mean']) if epoch_metrics['reward_mean'] else 0.0,
                'accuracy_mean': avg_metrics['accuracy_mean'],
                'accuracy_std': np.std(epoch_metrics['accuracy_mean']) if epoch_metrics['accuracy_mean'] else 0.0,
                'latency_mean': avg_metrics['latency_mean'],
                'latency_std': np.std(epoch_metrics['latency_mean']) if epoch_metrics['latency_mean'] else 0.0,
                'budget_violation_rate': avg_metrics['budget_violation_rate'],
            })
        
        log.info(f"Training history saved to {history_file}")
        
        # Validate
        if val_loader is not None:
            val_metrics = trainer.validate(
                val_loader=val_loader,
                lang_extractor=lang_extractor,
                budget_encoder=budget_encoder,
            )
            
            log.info(f"Val Epoch {epoch+1} Metrics:")
            log.info(f"  Reward: {val_metrics.get('reward_mean', 0):.4f}")
            log.info(f"  Accuracy: {val_metrics.get('accuracy_mean', 0):.4f}")
            log.info(f"  Latency: {val_metrics.get('latency_mean', 0):.2f}ms")
            log.info(f"  Budget Violation Rate: {val_metrics.get('budget_violation_rate', 0):.4f}")
            
            # Log to TensorBoard
            writer.add_scalar('Val/Reward_Mean', val_metrics.get('reward_mean', 0), epoch + 1)
            writer.add_scalar('Val/Accuracy_Mean', val_metrics.get('accuracy_mean', 0), epoch + 1)
            writer.add_scalar('Val/Latency_Mean', val_metrics.get('latency_mean', 0), epoch + 1)
            writer.add_scalar('Val/Budget_Violation_Rate', val_metrics.get('budget_violation_rate', 0), epoch + 1)
            
            # Log to Weights & Biases (if enabled)
            # Use run.log() instead of wandb.log() to match official wandb style
            if use_wandb and wandb_run is not None:
                try:
                    # Log validation metrics with proper formatting for wandb
                    val_metrics_to_log = {
                        'val/reward_mean': val_metrics.get('reward_mean', 0),
                        'val/accuracy_mean': val_metrics.get('accuracy_mean', 0),
                        'val/latency_mean': val_metrics.get('latency_mean', 0),
                        'val/budget_violation_rate': val_metrics.get('budget_violation_rate', 0),
                        'epoch': epoch + 1,
                    }
                    wandb_run.log(val_metrics_to_log, step=epoch + 1)
                    log.debug(f"Logged validation metrics to wandb: {list(val_metrics_to_log.keys())}")
                except Exception as e:
                    log.warning(f"Failed to log validation metrics to wandb: {e}")
                    import traceback
                    log.debug(traceback.format_exc())
            
            # Append validation metrics to CSV
            import csv
            val_history_file = output_path / 'validation_history.csv'
            val_file_exists = val_history_file.exists()
            
            with open(val_history_file, 'a', newline='') as f:
                csv_writer = csv.DictWriter(f, fieldnames=['epoch', 'reward_mean', 'accuracy_mean', 
                                                       'latency_mean', 'budget_violation_rate'])
                if not val_file_exists:
                    csv_writer.writeheader()
                csv_writer.writerow({
                    'epoch': epoch + 1,
                    'reward_mean': val_metrics.get('reward_mean', 0),
                    'accuracy_mean': val_metrics.get('accuracy_mean', 0),
                    'latency_mean': val_metrics.get('latency_mean', 0),
                    'budget_violation_rate': val_metrics.get('budget_violation_rate', 0),
                })
            
            # Save best model
            val_reward = val_metrics.get('reward_mean', float('-inf'))
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                checkpoint = {
                    'epoch': epoch,
                    'knob1_state_dict': knob1_predictor.state_dict(),
                    'knob2_knob3_state_dict': knob2_knob3_predictor.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'val_reward': val_reward,
                    'val_metrics': val_metrics,
                }
                torch.save(checkpoint, output_path / 'best_joint_checkpoint.pt')
                log.info(f"Saved best model (val_reward={best_val_reward:.4f})")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'knob1_state_dict': knob1_predictor.state_dict(),
                'knob2_knob3_state_dict': knob2_knob3_predictor.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
            }
            torch.save(checkpoint, output_path / f'joint_checkpoint_epoch_{epoch+1}.pt')
            log.info(f"Saved checkpoint at epoch {epoch+1}")
    
    # Close TensorBoard writer
    writer.close()
    
    # Finish wandb run (if enabled)
    # Use run.finish() instead of wandb.finish() to match official wandb style
    if use_wandb and wandb_run is not None:
        wandb_run.finish()
        log.info("Weights & Biases run finished")
    
    log.info(f"\nJoint training completed! Best model saved to {output_path / 'best_joint_checkpoint.pt'}")
    log.info(f"TensorBoard logs saved to {tensorboard_dir}")
    log.info(f"To view TensorBoard, run: tensorboard --logdir={tensorboard_dir}")
    if use_wandb and wandb_run is not None:
        log.info(f"View W&B run at: {wandb_run.url if hasattr(wandb_run, 'url') else 'https://wandb.ai'}")
    return knob1_predictor, knob2_knob3_predictor


def main():
    parser = argparse.ArgumentParser(
        description="Train Joint Controller (Stage1 + Stage2)",
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
    # Training args
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/joint_controller",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,  # Increased for H100 (80GB), can go higher if memory allows
        help="Batch size (default: 64 for H100, can increase to 128+ if memory allows)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for controller"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=5,
        help="Group size for GRPO"
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
    parser.add_argument(
        "--load_stage1_checkpoint",
        type=str,
        default=None,
        help="Optional: Path to pre-trained Stage1 checkpoint (for fine-tuning)"
    )
    parser.add_argument(
        "--importance_scores_file",
        type=str,
        default="results/layer_importance_scores_exp3_recommended.json",
        help="Path to JSON file containing importance scores for transformer blocks"
    )
    parser.add_argument(
        "--use_multi_gpu",
        action="store_true",
        help="Use DataParallel for multi-GPU training (if multiple GPUs available)"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging (requires wandb to be installed)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (default: 'molmo-controller')"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team name (optional)"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated from config)"
    )
    
    args = parser.parse_args()
    
    # Load training data
    log.info("Loading training data...")
    data_loader = CoreExpDataLoader(args.results_dir)
    training_data = data_loader.load_multiple_datasets(args.dataset_names)
    
    if not training_data:
        raise ValueError("No training data found!")
    
    log.info(f"Loaded {len(training_data)} training samples")
    
    # Train joint controller
    log.info("=" * 80)
    log.info("Training One-Stage Controller with GRPO")
    log.info("=" * 80)
    
    train_joint_controller(
        training_data=training_data,
        model_path=args.model_path,
        output_dir=args.output_dir,
        dataset_names=args.dataset_names,
        device=args.device,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        group_size=args.group_size,
        train_split=args.train_split,
        seed=args.seed,
        load_stage1_checkpoint=args.load_stage1_checkpoint,
        latency_budget_min=170.0,
        latency_budget_max=380.0,
        importance_scores_file=args.importance_scores_file,
        use_multi_gpu=args.use_multi_gpu,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
    )
    
    log.info("Training completed!")


if __name__ == "__main__":
    main()

