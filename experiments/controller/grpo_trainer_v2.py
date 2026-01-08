"""
Improved GRPO trainer with proper group relative ranking.
Implements Group Relative Policy Optimization for controller training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import json
import time

log = logging.getLogger(__name__)


class ControllerDataset(Dataset):
    """
    Dataset for controller training.
    Each sample contains features and ground truth knob configuration with rewards.
    """
    
    def __init__(
        self,
        training_data: List[Dict],
        knob1_options: List[str] = None,
        knob2_options: List[int] = None,
        knob3_options: List[int] = None,
    ):
        """
        Args:
            training_data: List of training samples with features and configs
            knob1_options: List of tier options (default: ["low", "medium", "high"])
            knob2_options: List of top_k options (default: [4, 6, 8, 10, 12])
            knob3_options: List of num_active_blocks options (default: [8, 10, 12, 14, 16])
        """
        self.data = training_data
        
        # Set default options
        self.knob1_options = knob1_options or ["low", "medium", "high"]
        self.knob2_options = knob2_options or [4, 6, 8, 10, 12]
        self.knob3_options = knob3_options or [8, 10, 12, 14, 16]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Map config values to indices
        try:
            knob1_idx = self.knob1_options.index(item['tier'])
        except ValueError:
            # Default to medium if tier not found
            knob1_idx = 1
        
        try:
            knob2_idx = self.knob2_options.index(item['top_k'])
        except ValueError:
            # Find closest value
            knob2_idx = min(range(len(self.knob2_options)),
                          key=lambda i: abs(self.knob2_options[i] - item['top_k']))
        
        try:
            knob3_idx = self.knob3_options.index(item['num_active_blocks'])
        except ValueError:
            # Find closest value
            knob3_idx = min(range(len(self.knob3_options)),
                          key=lambda i: abs(self.knob3_options[i] - item['num_active_blocks']))
        
        return {
            'vision_feat': torch.tensor(item['vision_feat'], dtype=torch.float32),
            'lang_feat': torch.tensor(item['lang_feat'], dtype=torch.float32),
            'budget_feat': torch.tensor(item['budget_feat'], dtype=torch.float32),
            'latency_budget': torch.tensor(item['latency_budget'], dtype=torch.float32),
            'knob1_idx': torch.tensor(knob1_idx, dtype=torch.long),
            'knob2_idx': torch.tensor(knob2_idx, dtype=torch.long),
            'knob3_idx': torch.tensor(knob3_idx, dtype=torch.long),
            'accuracy': torch.tensor(item['accuracy'], dtype=torch.float32),
            'latency': torch.tensor(item['latency'], dtype=torch.float32),
            'reward': torch.tensor(item.get('reward', 0.0), dtype=torch.float32),
            'sample_id': item.get('sample_id', idx),
        }


class GRPOTrainerV2:
    """
    Improved GRPO trainer with proper group relative ranking.
    
    Key improvements:
    1. Proper group formation (by sample_id and latency_budget)
    2. Group relative ranking loss
    3. Better reward computation
    4. Comprehensive logging and checkpointing
    """
    
    def __init__(
        self,
        controller: nn.Module,
        reward_fn,
        device: str = "cuda",
        group_size: int = 5,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        Args:
            controller: Controller model
            reward_fn: Reward function
            device: Device to use
            group_size: Size of groups for GRPO
            lr: Learning rate
            weight_decay: Weight decay
            max_grad_norm: Maximum gradient norm for clipping
            temperature: Temperature for action sampling
        """
        self.controller = controller.to(device)
        self.reward_fn = reward_fn
        self.device = device
        self.group_size = group_size
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        
        # Optimizer
        self.optimizer = optim.Adam(
            controller.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        # Training history
        self.train_losses = []
        self.train_rewards = []
        self.val_metrics_history = []
    
    def form_groups(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Form groups from batch data.
        Groups are formed by (sample_id, latency_budget) pairs.
        
        Args:
            batch: Batch of data
        
        Returns:
            List of groups
        """
        batch_size = batch['vision_feat'].shape[0]
        
        # Group by (sample_id, latency_budget)
        groups_dict = {}
        sample_ids = batch.get('sample_id', [i for i in range(batch_size)])
        budgets = batch['latency_budget'].cpu().numpy()
        
        for i in range(batch_size):
            key = (sample_ids[i], float(budgets[i]))
            if key not in groups_dict:
                groups_dict[key] = []
            groups_dict[key].append(i)
        
        # Form groups
        groups = []
        for key, indices in groups_dict.items():
            if len(indices) >= 2:  # Need at least 2 samples for ranking
                group = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        group[k] = v[indices]
                    else:
                        group[k] = [v[i] for i in indices]
                groups.append(group)
        
        return groups
    
    def compute_grpo_loss(
        self,
        logits: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        groups: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss with group relative ranking.
        
        Args:
            logits: Controller logits
            actions: Action indices
            rewards: Reward values
            groups: List of groups
        
        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics
        """
        # Compute log probabilities
        log_probs = self.controller.compute_log_probs(logits, actions)  # (B,)
        
        # Group relative ranking loss
        total_loss = 0.0
        num_pairs = 0
        
        start_idx = 0
        group_losses = []
        
        for group in groups:
            group_size = group['vision_feat'].shape[0]
            group_log_probs = log_probs[start_idx:start_idx + group_size]
            group_rewards = rewards[start_idx:start_idx + group_size]
            
            # Sort by reward (descending)
            sorted_indices = torch.argsort(group_rewards, descending=True)
            sorted_log_probs = group_log_probs[sorted_indices]
            sorted_rewards = group_rewards[sorted_indices]
            
            # For each pair (i, j) where i > j in ranking
            group_loss = 0.0
            group_pairs = 0
            
            for i in range(group_size):
                for j in range(i + 1, group_size):
                    # Higher reward should have higher log-prob
                    log_prob_diff = sorted_log_probs[i] - sorted_log_probs[j]
                    reward_diff = sorted_rewards[i] - sorted_rewards[j]
                    
                    # Loss: -log(sigmoid(log_prob_diff * sign(reward_diff)))
                    # This encourages higher log-prob for higher reward
                    pair_loss = -F.logsigmoid(log_prob_diff * torch.sign(reward_diff))
                    group_loss += pair_loss
                    group_pairs += 1
            
            if group_pairs > 0:
                group_loss = group_loss / group_pairs
                group_losses.append(group_loss.item())
                total_loss += group_loss
                num_pairs += group_pairs
            
            start_idx += group_size
        
        if num_pairs > 0:
            loss = total_loss / len(groups)  # Average over groups
        else:
            # Fallback: standard policy gradient
            advantages = rewards - rewards.mean()
            loss = -(log_probs * advantages).mean()
        
        metrics = {
            'loss': loss.item(),
            'num_groups': len(groups),
            'num_pairs': num_pairs,
            'group_loss_mean': np.mean(group_losses) if group_losses else 0.0,
        }
        
        return loss, metrics
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Run one training step.
        
        Args:
            batch: Batch of data
        
        Returns:
            Metrics dictionary
        """
        self.controller.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_device[k] = v.to(self.device)
            else:
                batch_device[k] = v
        
        # Forward
        logits = self.controller(
            vision_feat=batch_device['vision_feat'],
            lang_feat=batch_device['lang_feat'],
            budget_feat=batch_device['budget_feat'],
        )
        
        # Compute rewards if not provided
        if 'reward' not in batch_device or batch_device['reward'].sum() == 0:
            # Map indices to actual values for reward computation
            knob1_options = ["low", "medium", "high"]
            knob2_options = [4, 6, 8, 10, 12]
            knob3_options = [8, 10, 12, 14, 16]
            
            config = {
                'max_crops': torch.tensor(
                    [3 if k == "low" else (6 if k == "medium" else 12) 
                     for k in [knob1_options[i] for i in batch_device['knob1_idx'].cpu()]],
                    device=self.device
                ),
                'top_k': torch.tensor(
                    [knob2_options[i] for i in batch_device['knob2_idx'].cpu()],
                    device=self.device
                ),
                'num_active_blocks': torch.tensor(
                    [knob3_options[i] for i in batch_device['knob3_idx'].cpu()],
                    device=self.device
                ),
            }
            
            rewards = self.reward_fn(
                accuracy=batch_device['accuracy'],
                latency=batch_device['latency'],
                latency_budget=batch_device['latency_budget'],
                config=config,
            )
        else:
            rewards = batch_device['reward']
        
        # Form groups
        groups = self.form_groups(batch_device)
        
        if not groups:
            # Fallback: use all samples as one group
            groups = [batch_device]
        
        # Compute GRPO loss
        actions = {
            'knob1_idx': batch_device['knob1_idx'],
            'knob2_idx': batch_device['knob2_idx'],
            'knob3_idx': batch_device['knob3_idx'],
        }
        
        loss, loss_metrics = self.compute_grpo_loss(logits, actions, rewards, groups)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Metrics
        metrics = {
            **loss_metrics,
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'accuracy_mean': batch_device['accuracy'].mean().item(),
            'latency_mean': batch_device['latency'].mean().item(),
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validation.
        
        Args:
            val_loader: Validation dataloader
        
        Returns:
            Validation metrics
        """
        self.controller.eval()
        
        all_rewards = []
        all_accuracies = []
        all_latencies = []
        all_budget_violations = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move to device
                batch_device = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_device[k] = v.to(self.device)
                    else:
                        batch_device[k] = v
                
                # Forward
                logits = self.controller(
                    vision_feat=batch_device['vision_feat'],
                    lang_feat=batch_device['lang_feat'],
                    budget_feat=batch_device['budget_feat'],
                )
                
                # Deterministic sampling (argmax)
                actions = self.controller.sample_actions(logits, deterministic=True)
                
                # Compute reward
                knob1_options = ["low", "medium", "high"]
                knob2_options = [4, 6, 8, 10, 12]
                knob3_options = [8, 10, 12, 14, 16]
                
                config = {
                    'max_crops': torch.tensor(
                        [3 if k == "low" else (6 if k == "medium" else 12) 
                         for k in actions['knob1']],
                        device=self.device
                    ),
                    'top_k': actions['knob2'],
                    'num_active_blocks': actions['knob3'],
                }
                
                rewards = self.reward_fn(
                    accuracy=batch_device['accuracy'],
                    latency=batch_device['latency'],
                    latency_budget=batch_device['latency_budget'],
                    config=config,
                )
                
                all_rewards.extend(rewards.cpu().numpy())
                all_accuracies.extend(batch_device['accuracy'].cpu().numpy())
                all_latencies.extend(batch_device['latency'].cpu().numpy())
                
                # Budget violations
                violations = (batch_device['latency'] > batch_device['latency_budget']).float()
                all_budget_violations.extend(violations.cpu().numpy())
        
        metrics = {
            'reward_mean': np.mean(all_rewards),
            'reward_std': np.std(all_rewards),
            'accuracy_mean': np.mean(all_accuracies),
            'latency_mean': np.mean(all_latencies),
            'budget_violation_rate': np.mean(all_budget_violations),
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 100,
        val_loader: Optional[DataLoader] = None,
        save_dir: Optional[str] = None,
        save_every: int = 10,
        log_every: int = 100,
    ):
        """
        Training loop.
        
        Args:
            train_loader: Training dataloader
            num_epochs: Number of epochs
            val_loader: Optional validation dataloader
            save_dir: Directory to save checkpoints
            save_every: Save every N epochs
            log_every: Log every N steps
        """
        best_val_reward = float('-inf')
        
        for epoch in range(num_epochs):
            # Train phase
            self.controller.train()
            epoch_metrics = {
                'loss': [],
                'reward_mean': [],
                'accuracy_mean': [],
                'latency_mean': [],
            }
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for step, batch in enumerate(pbar):
                metrics = self.train_step(batch)
                
                # Accumulate metrics
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])
                
                # Update progress bar
                if (step + 1) % log_every == 0:
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'reward': f"{metrics['reward_mean']:.4f}",
                    })
            
            # Epoch averages
            avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items() if v}
            log.info(f"Epoch {epoch+1}/{num_epochs}: {avg_metrics}")
            
            self.train_losses.append(avg_metrics.get('loss', 0.0))
            self.train_rewards.append(avg_metrics.get('reward_mean', 0.0))
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                log.info(f"Validation: {val_metrics}")
                
                self.val_metrics_history.append(val_metrics)
                
                # Update learning rate
                self.scheduler.step(val_metrics['reward_mean'])
                
                # Save best model
                if val_metrics['reward_mean'] > best_val_reward:
                    best_val_reward = val_metrics['reward_mean']
                    if save_dir:
                        self.save_checkpoint(save_dir, epoch, is_best=True, metrics=val_metrics)
            
            # Periodic save
            if save_dir and (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_dir, epoch, is_best=False)
    
    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        is_best: bool = False,
        metrics: Optional[Dict] = None,
    ):
        """Save checkpoint."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'controller_state_dict': self.controller.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'train_rewards': self.train_rewards,
            'val_metrics_history': self.val_metrics_history,
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        if is_best:
            path = save_path / 'best_checkpoint.pt'
        else:
            path = save_path / f'checkpoint_epoch_{epoch+1}.pt'
        
        torch.save(checkpoint, path)
        log.info(f"Saved checkpoint to {path}")
        
        # Also save metadata
        metadata = {
            'epoch': epoch,
            'is_best': is_best,
            'metrics': metrics,
            'timestamp': time.time(),
        }
        metadata_path = save_path / f'metadata_epoch_{epoch+1}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.controller.load_state_dict(checkpoint['controller_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_rewards = checkpoint.get('train_rewards', [])
        self.val_metrics_history = checkpoint.get('val_metrics_history', [])
        log.info(f"Loaded checkpoint from {checkpoint_path}")

