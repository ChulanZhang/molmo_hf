"""
GRPO trainer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)


class ControllerDataset(Dataset):
    """Controller training dataset."""
    
    def __init__(self, training_data: List[Dict]):
        self.data = training_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Map config values to indices
        max_crops_options = [2, 4, 6, 8, 10, 12]
        top_k_options = [4, 8, 12, 16, 20, 24, 28, 32]
        blocks_options = [8, 9, 10, 11, 12, 13, 14, 15, 16]
        
        max_crops_idx = max_crops_options.index(item['config']['max_crops'])
        top_k_idx = top_k_options.index(item['config']['top_k'])
        blocks_idx = blocks_options.index(item['config']['num_active_blocks'])
        
        return {
            'image_feature': item['image_feature'],
            'language_feature': item['language_feature'],
            'latency_budget': torch.tensor(item['latency_budget'], dtype=torch.float32),
            'max_crops_idx': torch.tensor(max_crops_idx, dtype=torch.long),
            'top_k_idx': torch.tensor(top_k_idx, dtype=torch.long),
            'blocks_idx': torch.tensor(blocks_idx, dtype=torch.long),
            'accuracy': torch.tensor(item['accuracy'], dtype=torch.float32),
            'latency': torch.tensor(item.get('latency', 0.0), dtype=torch.float32),
            'sample_id': item['sample_id'],
        }


class GRPOTrainer:
    """GRPO trainer."""
    
    def __init__(
        self,
        controller: nn.Module,
        reward_fn,
        device: str = "cuda",
        group_size: int = 8,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        self.controller = controller.to(device)
        self.reward_fn = reward_fn
        self.device = device
        self.group_size = group_size
        
        self.optimizer = optim.Adam(
            controller.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        self.train_losses = []
        self.train_rewards = []
    
    def group_trajectories(
        self,
        batch: Dict[str, torch.Tensor],
        group_size: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Group batch data.

        Args:
            batch: batch tensors
            group_size: group size

        Returns:
            list of grouped data dicts
        """
        batch_size = batch['image_feature'].shape[0]
        groups = []
        
        # Group by latency budget (same budget in one group)
        budget_values = batch['latency_budget'].cpu().numpy()
        unique_budgets = np.unique(budget_values)
        
        for budget in unique_budgets:
            mask = budget_values == budget
            indices = np.where(mask)[0]
            
            # Split indices into groups
            for i in range(0, len(indices), group_size):
                group_indices = indices[i:i+group_size]
                if len(group_indices) < 2:
                    continue  # skip groups that are too small
                
                group = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        group[key] = value[group_indices]
                    else:
                        group[key] = [value[idx] for idx in group_indices]
                groups.append(group)
        
        return groups
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        groups: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute per-group relative advantages.

        Args:
            rewards: (B,) reward values
            groups: grouped data

        Returns:
            advantages: (B,) advantage values
        """
        advantages = torch.zeros_like(rewards)
        
        start_idx = 0
        for group in groups:
            group_size = group['image_feature'].shape[0]
            group_rewards = rewards[start_idx:start_idx+group_size]
            
            # Mean reward per group
            group_mean = group_rewards.mean()
            
            # Relative advantage
            group_advantages = group_rewards - group_mean
            
            advantages[start_idx:start_idx+group_size] = group_advantages
            start_idx += group_size
        
        return advantages
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Run one training step.

        Args:
            batch: batch tensors

        Returns:
            training metrics
        """
        self.controller.train()
        self.optimizer.zero_grad()
        
        # Forward
        logits = self.controller(
            image_feat=batch['image_feature'].to(self.device),
            lang_feat=batch['language_feature'].to(self.device),
            budget=batch['latency_budget'].to(self.device),
        )
        
        # Compute reward
        actions = {
            'max_crops_idx': batch['max_crops_idx'].to(self.device),
            'top_k_idx': batch['top_k_idx'].to(self.device),
            'blocks_idx': batch['blocks_idx'].to(self.device),
        }
        
        # Map indices to actual values (for reward computation)
        max_crops_options = [2, 4, 6, 8, 10, 12]
        top_k_options = [4, 8, 12, 16, 20, 24, 28, 32]
        blocks_options = [8, 9, 10, 11, 12, 13, 14, 15, 16]
        
        config = {
            'max_crops': torch.tensor([max_crops_options[i] for i in actions['max_crops_idx'].cpu()], 
                                     device=self.device),
            'top_k': torch.tensor([top_k_options[i] for i in actions['top_k_idx'].cpu()], 
                                device=self.device),
            'num_active_blocks': torch.tensor([blocks_options[i] for i in actions['blocks_idx'].cpu()], 
                                            device=self.device),
        }
        
        rewards = self.reward_fn(
            accuracy=batch['accuracy'].to(self.device),
            latency=batch['latency'].to(self.device),
            latency_budget=batch['latency_budget'].to(self.device),
            config=config,
        )
        
        # Group trajectories and compute advantages
        groups = self.group_trajectories(batch, self.group_size)
        advantages = self.compute_advantages(rewards, groups)
        
        # Compute log probabilities
        log_probs = self.controller.compute_log_probs(logits, actions)
        
        # Loss (policy gradient)
        loss = -(log_probs * advantages).mean()
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Log metrics
        metrics = {
            'loss': loss.item(),
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 100,
        val_loader: Optional[DataLoader] = None,
        save_dir: Optional[str] = None,
        save_every: int = 10,
    ):
        """
        Training loop.

        Args:
            train_loader: training dataloader
            num_epochs: number of epochs
            val_loader: optional validation dataloader
            save_dir: directory to save checkpoints
            save_every: save every N epochs
        """
        best_val_reward = float('-inf')
        
        for epoch in range(num_epochs):
            # Train phase
            self.controller.train()
            epoch_metrics = {
                'loss': [],
                'reward_mean': [],
                'reward_std': [],
            }
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                metrics = self.train_step(batch)
                
                for key, value in metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key].append(value)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'reward': f"{metrics['reward_mean']:.4f}",
                })
            
            # Epoch averages
            avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
            log.info(f"Epoch {epoch+1}: {avg_metrics}")
            
            self.train_losses.append(avg_metrics['loss'])
            self.train_rewards.append(avg_metrics['reward_mean'])
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                log.info(f"Validation: {val_metrics}")
                
                if val_metrics['reward_mean'] > best_val_reward:
                    best_val_reward = val_metrics['reward_mean']
                    if save_dir:
                        self.save_checkpoint(save_dir, epoch, is_best=True)
            
            # Periodic save
            if save_dir and (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_dir, epoch, is_best=False)
    
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validation.

        Args:
            val_loader: validation dataloader

        Returns:
            validation metrics
        """
        self.controller.eval()
        
        all_rewards = []
        all_accuracies = []
        all_latencies = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                logits = self.controller(
                    image_feat=batch['image_feature'].to(self.device),
                    lang_feat=batch['language_feature'].to(self.device),
                    budget=batch['latency_budget'].to(self.device),
                )
                
                # Deterministic sampling (argmax actions)
                actions = self.controller.sample_actions(logits, deterministic=True)
                
                # Compute reward
                max_crops_options = [2, 4, 6, 8, 10, 12]
                top_k_options = [4, 8, 12, 16, 20, 24, 28, 32]
                blocks_options = [8, 9, 10, 11, 12, 13, 14, 15, 16]
                
                config = {
                    'max_crops': actions['max_crops'],
                    'top_k': actions['top_k'],
                    'num_active_blocks': actions['num_active_blocks'],
                }
                
                rewards = self.reward_fn(
                    accuracy=batch['accuracy'].to(self.device),
                    latency=batch['latency'].to(self.device),
                    latency_budget=batch['latency_budget'].to(self.device),
                    config=config,
                )
                
                all_rewards.extend(rewards.cpu().numpy())
                all_accuracies.extend(batch['accuracy'].numpy())
                all_latencies.extend(batch['latency'].numpy())
        
        metrics = {
            'reward_mean': np.mean(all_rewards),
            'reward_std': np.std(all_rewards),
            'accuracy_mean': np.mean(all_accuracies),
            'latency_mean': np.mean(all_latencies),
        }
        
        return metrics
    
    def save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        is_best: bool = False,
    ):
        """Save checkpoint."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'controller_state_dict': self.controller.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_rewards': self.train_rewards,
        }
        
        if is_best:
            path = os.path.join(save_dir, 'best_checkpoint.pt')
        else:
            path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        
        torch.save(checkpoint, path)
        log.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.controller.load_state_dict(checkpoint['controller_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_rewards = checkpoint.get('train_rewards', [])
        log.info(f"Loaded checkpoint from {checkpoint_path}")

